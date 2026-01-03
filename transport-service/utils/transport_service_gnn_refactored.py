"""
Refactored Transport Service with Temporal-Aware GNN
Predicts best transport method based on:
- Origin/Destination locations
- Date (regular, weekend, poya day, holiday)
- Time (early_morning, morning, day, evening, night, late_night)
- Available modes (bus, train, ridehailing)

Uses survey-based rules for reliability/crowding without service_conditions.csv
Uses HolidayDetector API instead of calendar.csv

Environment Variables:
- PREDICTION_MODE: 'hybrid' (default) or 'gnn_only'
  * hybrid: Use CSV data when available, GNN predictions for missing routes
  * gnn_only: Always use GNN predictions, ignore CSV data
- DATA_SOURCE: 'csv' (default) or 'mongodb'
- HOLIDAY_API_KEY: Optional HolidayAPI key for fetching holidays
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .data_repository import DataRepository, create_repository
from .temporal_gnn_model import TemporalTransportGNN, LegacyTemporalTransportGNN
from .holiday_detector import HolidayDetector
from .temporal_features import (
    get_reliability_score,
    get_crowding_score,
)


class TransportServiceGNNRefactored:
    """
    Main service class using temporal-aware GNN for transport recommendations.
    
    Key Changes from v1:
    - Uses TemporalTransportGNN instead of TransportGNN
    - Predicts best MODE (not edge ratings)
    - Incorporates temporal features (date/time)
    - Uses HolidayDetector instead of calendar.csv
    - Uses survey rules instead of service_conditions.csv
    """

    # Mode mappings
    # Support both spellings for ride hailing
    MODE_TO_ID = {"bus": 0, "train": 1, "ride_hail": 2, "ridehailing": 2}
    ID_TO_MODE = {0: "bus", 1: "train", 2: "ride_hail"}

    # Day type mappings
    DAY_TYPE_TO_ID = {"regular": 0, "weekend": 1, "poya": 2, "holiday": 3}
    ID_TO_DAY_TYPE = {0: "regular", 1: "weekend", 2: "poya", 3: "holiday"}

    # Time bucket mappings (finer granularity)
    TIME_BUCKET_TO_ID = {
        "late_night": 0,    # 00:00-02:59
        "early_morning": 1, # 03:00-05:59
        "morning_peak": 2,  # 06:00-08:59
        "mid_morning": 3,   # 09:00-11:59
        "midday": 4,        # 12:00-14:59
        "afternoon": 5,     # 15:00-17:59
        "evening": 6,       # 18:00-20:59
        "night": 7,         # 21:00-23:59
    }
    ID_TO_TIME_BUCKET = {v: k for k, v in TIME_BUCKET_TO_ID.items()}

    # Legacy coarse time-period mapping (for backward-compatible checkpoints)
    TIME_PERIOD_TO_ID = {
        "early_morning": 0,
        "morning": 1,
        "day": 2,
        "evening": 3,
        "night": 4,
        "late_night": 5,
    }
    ID_TO_TIME_PERIOD = {v: k for k, v in TIME_PERIOD_TO_ID.items()}

    def __init__(
        self,
        model_path: str,
        data_path: str = None,
        use_mongodb: bool = False,
    ):
        """
        Initialize the refactored transport service.

        Args:
            model_path: Path to trained temporal GNN model (.pth file)
            data_path: Path to transport data directory
            use_mongodb: If True, use MongoDB; otherwise use CSV
        """
        self.model = None
        self.artifacts = {}
        self.data_version = None
        self.holiday_detector = HolidayDetector(use_api=False)
        self.location_id_to_idx = {}
        self.day_of_week_to_id = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
        self.model_uses_time_bucket = True
        self.model_uses_day_of_week = True

        # Prediction mode: 'hybrid' (default) or 'gnn_only'
        # hybrid: use CSV data when available, GNN for missing routes
        # gnn_only: always use GNN predictions (ignore CSV routes)
        self.prediction_mode = os.getenv("PREDICTION_MODE", "hybrid").lower()
        if self.prediction_mode not in ["hybrid", "gnn_only"]:
            print(f"⚠️ Invalid PREDICTION_MODE '{self.prediction_mode}', defaulting to 'hybrid'")
            self.prediction_mode = "hybrid"
        print(f"🔧 Prediction mode: {self.prediction_mode}")

        # Initialize data repository (MongoDB or CSV)
        try:
            use_mongo = use_mongodb or os.getenv("DATA_SOURCE") == "mongodb"
            self.repository = create_repository(use_mongo=use_mongo)
            self.data_version = self.repository.get_data_version()
        except Exception as e:
            print(f"⚠️  Repository init failed: {e}")
            self.repository = None

        # Load model
        self._load_model(model_path)

        # Load data through repository
        self._load_data()

    def _ensure_services_df(self):
        """If services_df is missing, derive minimal metadata from edges."""
        if self.services_df is not None:
            return
        if self.edges_df is None:
            return
        # Derive a minimal services table from edges
        derived = self.edges_df[
            [
                "service_id",
                "mode",
                "operator",
                "fare_lkr",
                "duration_min",
                "distance_km",
            ]
        ].copy()
        derived = derived.rename(
            columns={
                "fare_lkr": "base_fare",
                "duration_min": "base_duration_min",
            }
        )
        self.services_df = derived.drop_duplicates(subset=["service_id"]).reset_index(drop=True)

    def _load_model(self, model_path: str):
        """Load the trained temporal GNN model and artifacts."""
        try:
            if not os.path.exists(model_path):
                print(f"⚠️ Model not found at {model_path}")
                print("   Service will use survey-based rules for recommendations")
                return

            checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")

            # Validate checkpoint
            if not isinstance(checkpoint, dict):
                print("⚠️ Incompatible checkpoint format. Expected dict with model artifacts.")
                self.model = None
                return

            required_keys = {"node_features", "edge_index", "model_state_dict"}
            if not required_keys.issubset(checkpoint.keys()):
                missing = required_keys.difference(checkpoint.keys())
                print(f"⚠️ Checkpoint missing keys: {missing}")
                self.model = None
                return

            # Detect legacy checkpoint (uses time_period_embedding instead of time_bucket/day_of_week)
            state_dict = checkpoint.get("model_state_dict", {})
            is_legacy_temporal = any(
                k.startswith("time_period_embedding") for k in state_dict.keys()
            ) and not any(k.startswith("time_bucket_embedding") for k in state_dict.keys())

            # Initialize model architecture with saved parameters
            node_features = checkpoint["node_features"]
            num_day_types = checkpoint.get("num_day_types", 2)
            num_modes = checkpoint.get("num_modes", 3)
            hidden_dim = checkpoint.get("hidden_dim", 64)

            if is_legacy_temporal:
                num_time_periods = checkpoint.get(
                    "num_time_periods", len(self.TIME_PERIOD_TO_ID)
                )
                self.model = LegacyTemporalTransportGNN(
                    node_features=node_features.shape[1],
                    num_day_types=num_day_types,
                    num_time_periods=num_time_periods,
                    num_modes=num_modes,
                    hidden_dim=hidden_dim,
                )
                self.model_uses_time_bucket = False
                self.model_uses_day_of_week = False
            else:
                num_time_buckets = checkpoint.get(
                    "num_time_buckets",
                    checkpoint.get("num_time_periods", len(self.TIME_BUCKET_TO_ID)),
                )
                num_day_of_week = checkpoint.get("num_day_of_week", len(self.day_of_week_to_id))
                self.model = TemporalTransportGNN(
                    node_features=node_features.shape[1],
                    num_day_types=num_day_types,
                    num_time_buckets=num_time_buckets,
                    num_day_of_week=num_day_of_week,
                    num_modes=num_modes,
                    hidden_dim=hidden_dim,
                )

            # Load trained weights
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            # Store artifacts
            self.artifacts = {
                "model": self.model,
                "node_features": checkpoint["node_features"],
                "edge_index": checkpoint["edge_index"],
            }

            # Load location_id_to_idx mapping from checkpoint
            if "location_id_to_idx" in checkpoint:
                self.location_id_to_idx = checkpoint["location_id_to_idx"]
                print(f"   Loaded location mapping: {len(self.location_id_to_idx)} locations")
            else:
                print("   ⚠️ Warning: location_id_to_idx not in checkpoint, creating from data")
                # Fallback: create from current data (may not match training)
                if hasattr(self, 'nodes_df') and self.nodes_df is not None:
                    self.location_id_to_idx = {
                        int(row.location_id): idx 
                        for idx, row in self.nodes_df.reset_index().iterrows()
                    }

            # Override class-level mappings with checkpoint mappings
            if "mode_to_id" in checkpoint:
                self.MODE_TO_ID = checkpoint["mode_to_id"]
                self.ID_TO_MODE = {v: k for k, v in self.MODE_TO_ID.items()}
                print(f"   Loaded mode mapping: {self.MODE_TO_ID}")
            
            if "day_type_to_id" in checkpoint:
                self.DAY_TYPE_TO_ID = checkpoint["day_type_to_id"]
                self.ID_TO_DAY_TYPE = {v: k for k, v in self.DAY_TYPE_TO_ID.items()}
                print(f"   Loaded day_type mapping: {self.DAY_TYPE_TO_ID}")
            
            if "time_period_to_id" in checkpoint:
                # Backward compatibility with old checkpoints (coarse periods)
                self.TIME_PERIOD_TO_ID = checkpoint["time_period_to_id"]
                self.ID_TO_TIME_PERIOD = {v: k for k, v in self.TIME_PERIOD_TO_ID.items()}
                print(f"   Loaded time_period mapping: {self.TIME_PERIOD_TO_ID}")
            if "time_bucket_to_id" in checkpoint:
                self.TIME_BUCKET_TO_ID = checkpoint["time_bucket_to_id"]
                self.ID_TO_TIME_BUCKET = {v: k for k, v in self.TIME_BUCKET_TO_ID.items()}
                print(f"   Loaded time_bucket mapping: {self.TIME_BUCKET_TO_ID}")

            if "day_of_week_to_id" in checkpoint:
                self.day_of_week_to_id = checkpoint["day_of_week_to_id"]
                print(f"   Loaded day_of_week mapping: {self.day_of_week_to_id}")

            print(f"✅ Temporal GNN model loaded successfully from {model_path}")

        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            import traceback

            traceback.print_exc()
            self.model = None

    def _load_data(self):
        """Load data from repository (MongoDB or CSV)."""
        try:
            if self.repository is None:
                raise RuntimeError("Data repository not initialized")

            self.nodes_df = self.repository.get_nodes()
            self.services_df = self.repository.get_services()
            self.edges_df = self.repository.get_edges()

            # Ensure services_df exists even if repository lacked services.csv
            self._ensure_services_df()

            print(
                f"✅ Data loaded: {len(self.nodes_df)} nodes, "
                f"{len(self.services_df)} services, {len(self.edges_df)} edges"
            )
            print(f"   Data version: {self.data_version}")
            print(f"   Available modes: {self.services_df['mode'].unique().tolist()}")

        except Exception as e:
            print(f"⚠️ Could not load data: {e}")
            import traceback

            traceback.print_exc()
            self.nodes_df = None
            self.services_df = None
            self.edges_df = None

    def _get_temporal_features(
        self, departure_date: Optional[str], departure_time: Optional[str]
    ) -> Dict:
        """
        Get temporal features from date/time using HolidayDetector.

        Args:
            departure_date: Date in YYYY-MM-DD format
            departure_time: Time in HH:MM format

        Returns:
            Dict with temporal features
        """
        features = {
            "date": departure_date,
            "time": departure_time,
            "day_type": "regular",
            "time_period": "day",
            "time_bucket": "midday",
            "day_of_week": "monday",
            "is_crowded_likely": False,
        }

        try:
            if departure_date:
                date_obj = datetime.strptime(departure_date, "%Y-%m-%d")
                temporal_features = self.holiday_detector.get_temporal_features(
                    date_obj, departure_time or "12:00"
                )
                features.update(temporal_features)
                # Day-of-week (0=Monday)
                features["day_of_week"] = date_obj.strftime("%A").lower()

            if departure_time:
                # Ensure time period is set
                hour = int(departure_time.split(":")[0])
                # Map hour to fine-grained buckets
                if 0 <= hour < 3:
                    features["time_bucket"] = "late_night"
                elif 3 <= hour < 6:
                    features["time_bucket"] = "early_morning"
                elif 6 <= hour < 9:
                    features["time_bucket"] = "morning_peak"
                elif 9 <= hour < 12:
                    features["time_bucket"] = "mid_morning"
                elif 12 <= hour < 15:
                    features["time_bucket"] = "midday"
                elif 15 <= hour < 18:
                    features["time_bucket"] = "afternoon"
                elif 18 <= hour < 21:
                    features["time_bucket"] = "evening"
                else:
                    features["time_bucket"] = "night"

                # Coarse time_period for rule-based scoring
                if 5 <= hour < 7:
                    features["time_period"] = "early_morning"
                elif 7 <= hour < 10:
                    features["time_period"] = "morning"
                elif 10 <= hour < 16:
                    features["time_period"] = "day"
                elif 16 <= hour < 19:
                    features["time_period"] = "evening"
                elif 19 <= hour < 23:
                    features["time_period"] = "night"
                else:
                    features["time_period"] = "late_night"

        except Exception as e:
            print(f"⚠️ Could not parse temporal features: {e}")

        return features

    def predict_best_mode(
        self,
        origin_id: int,
        dest_id: int,
        departure_date: Optional[str] = None,
        departure_time: Optional[str] = None,
        available_modes: List[str] = None,
        distance_km: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        Predict the best transport mode using GNN (if available) or rules.

        Args:
            origin_id: Origin location ID
            dest_id: Destination location ID
            departure_date: YYYY-MM-DD format
            departure_time: HH:MM format
            available_modes: List of available modes (default: all)

        Returns:
            Dict with prediction details
        """
        if available_modes is None:
            available_modes = ["bus", "train", "ride_hail"]

        # Get temporal features
        temporal_features = self._get_temporal_features(
            departure_date, departure_time
        )
        day_type = temporal_features["day_type"]
        time_period = temporal_features["time_period"]
        time_bucket = temporal_features["time_bucket"]
        day_of_week = temporal_features["day_of_week"]

        if self.model is None:
            # Use survey-based rules
            return self._predict_with_rules(
                available_modes, day_type, time_period
            )

        # Use GNN model
        return self._predict_with_model(
            origin_id=origin_id,
            dest_id=dest_id,
            day_type=day_type,
            time_bucket=time_bucket,
            day_of_week=day_of_week,
            available_modes=available_modes,
            time_period=time_period,
            distance_km=distance_km,
        )

    def _predict_with_rules(
        self,
        available_modes: List[str],
        day_type: str,
        time_period: str,
    ) -> Dict:
        """Predict using survey-based rules."""
        best_mode = None
        best_reliability = -1

        for mode in available_modes:
            reliability = get_reliability_score(mode, time_period, day_type)
            crowding = get_crowding_score(mode, time_period, day_type)

            # Score: reliability - crowding penalty
            score = reliability - (crowding * 0.3)

            if score > best_reliability:
                best_reliability = score
                best_mode = mode

        return {
            "method": "rules",
            "best_mode": best_mode,
            "score": best_reliability,
            "day_type": day_type,
            "time_period": time_period,
        }

    def _predict_with_model(
        self,
        origin_id: int,
        dest_id: int,
        day_type: str,
        time_bucket: str,
        day_of_week: str,
        available_modes: List[str],
        time_period: Optional[str] = None,
        distance_km: Optional[float] = None,
    ) -> Dict:
        """Predict using GNN model."""
        try:
            # Map external location IDs to tensor indices
            if origin_id not in self.location_id_to_idx or dest_id not in self.location_id_to_idx:
                raise ValueError("Location mapping missing for origin or destination")

            origin_idx = self.location_id_to_idx[origin_id]
            dest_idx = self.location_id_to_idx[dest_id]

            # Prepare tensors
            origin_ids = torch.tensor([origin_idx], dtype=torch.long)
            dest_ids = torch.tensor([dest_idx], dtype=torch.long)
            day_type_ids = torch.tensor(
                [self.DAY_TYPE_TO_ID.get(day_type, 0)], dtype=torch.long
            )
            # Temporal IDs depend on model capabilities
            if self.model_uses_time_bucket:
                time_bucket_idx = self.TIME_BUCKET_TO_ID.get(time_bucket)
                if time_bucket_idx is None:
                    fallback_bucket = "midday" if "midday" in self.TIME_BUCKET_TO_ID else list(self.TIME_BUCKET_TO_ID.keys())[0]
                    time_bucket_idx = self.TIME_BUCKET_TO_ID.get(fallback_bucket, 0)
                time_bucket_ids = torch.tensor([time_bucket_idx], dtype=torch.long)

                day_of_week_ids = torch.tensor(
                    [self.day_of_week_to_id.get(day_of_week, 0)], dtype=torch.long
                )
            else:
                # Legacy: coarse time_period only
                period_idx = self.TIME_PERIOD_TO_ID.get(time_period or "day", 2)
                time_period_ids = torch.tensor([period_idx], dtype=torch.long)

            node_features = self.artifacts["node_features"]
            edge_index = self.artifacts["edge_index"]

            best_mode = None
            best_score = -1

            with torch.no_grad():
                for mode in available_modes:
                    mode_ids = torch.tensor(
                        [self.MODE_TO_ID[mode]], dtype=torch.long
                    )

                    if self.model_uses_time_bucket:
                        score = self.model.forward(
                            node_features,
                            edge_index,
                            origin_ids,
                            dest_ids,
                            day_type_ids,
                            time_bucket_ids,
                            day_of_week_ids,
                            mode_ids,
                        )
                    else:
                        score = self.model.forward(
                            node_features,
                            edge_index,
                            origin_ids,
                            dest_ids,
                            day_type_ids,
                            time_period_ids,
                            mode_ids,
                        )

                    score_val = score.item()
                    # Heuristic bias for legacy models: late-night + short trips favor ride_hail, penalize bus
                    if (not self.model_uses_time_bucket) and time_period in {"night", "late_night"} and distance_km is not None:
                        if distance_km <= 30:
                            if mode == "ride_hail":
                                score_val += 0.12
                            elif mode == "bus":
                                score_val -= 0.12
                            score_val = max(0.0, min(1.0, score_val))

                    if score_val > best_score:
                        best_score = score_val
                        best_mode = mode

            return {
                "method": "gnn",
                "best_mode": best_mode,
                "score": best_score,
                "day_type": day_type,
                "time_period": time_period,
                "time_bucket": time_bucket if self.model_uses_time_bucket else None,
                "day_of_week": day_of_week if self.model_uses_day_of_week else None,
                "distance_km": distance_km,
            }

        except Exception as e:
            print(f"⚠️ GNN prediction failed: {e}, returning error")
            return {
                "method": "gnn_error",
                "error": str(e),
                "day_type": day_type,
                "time_period": time_period,
                "time_bucket": time_bucket if self.model_uses_time_bucket else None,
                "day_of_week": day_of_week if self.model_uses_day_of_week else None,
                "available_modes": available_modes,
            }

    def get_recommendations(
        self,
        origin: str,
        destination: str,
        departure_date: Optional[str] = None,
        departure_time: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict:
        """
        Get ranked transport recommendations.

        Args:
            origin: Origin location name
            destination: Destination location name
            departure_date: YYYY-MM-DD format
            departure_time: HH:MM format
            top_k: Number of recommendations

        Returns:
            Recommendations with temporal context
        """
        try:
            if self.nodes_df is None or self.edges_df is None:
                return {
                    "error": "Service data not loaded",
                    "available_locations": [],
                }

            # Ensure services_df present
            self._ensure_services_df()

            # Look up locations
            origin_matches = self.nodes_df[
                self.nodes_df["name"].str.contains(origin, case=False, na=False)
            ]
            dest_matches = self.nodes_df[
                self.nodes_df["name"].str.contains(destination, case=False, na=False)
            ]

            if len(origin_matches) == 0:
                return {
                    "error": f"Origin '{origin}' not found",
                    "available_locations": self.nodes_df["name"].tolist(),
                }

            if len(dest_matches) == 0:
                return {
                    "error": f"Destination '{destination}' not found",
                    "available_locations": self.nodes_df["name"].tolist(),
                }

            origin_id = origin_matches.iloc[0]["location_id"]
            dest_id = dest_matches.iloc[0]["location_id"]
            origin_name = origin_matches.iloc[0]["name"]
            dest_name = dest_matches.iloc[0]["name"]

            # Get temporal features
            temporal_features = self._get_temporal_features(
                departure_date, departure_time
            )

            # Calculate distance between nodes early for heuristic scoring
            origin_node = self.nodes_df[self.nodes_df["location_id"] == origin_id].iloc[0]
            dest_node = self.nodes_df[self.nodes_df["location_id"] == dest_id].iloc[0]

            from math import radians, sin, cos, sqrt, atan2

            lat1, lon1 = radians(origin_node["latitude"]), radians(origin_node["longitude"])
            lat2, lon2 = radians(dest_node["latitude"]), radians(dest_node["longitude"])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            distance_km = 6371 * 2 * atan2(sqrt(a), sqrt(1-a))

            # Find available services from CSV (skip if gnn_only mode)
            available_edges = pd.DataFrame()  # Empty by default
            if self.prediction_mode == "hybrid":
                available_edges = self.repository.get_edges_between(origin_id, dest_id)

            # Determine available modes
            if len(available_edges) > 0 and self.prediction_mode == "hybrid":
                # Use modes from existing services in CSV
                available_modes = [
                    self.services_df[
                        self.services_df["service_id"] == edge_row["service_id"]
                    ].iloc[0]["mode"]
                    for _, edge_row in available_edges.iterrows()
                ]
                available_modes = list(set(available_modes))
            else:
                # No CSV data - use GNN to predict for all possible modes
                available_modes = list(self.MODE_TO_ID.keys())

            # Predict best mode using GNN
            best_mode_prediction = self.predict_best_mode(
                origin_id,
                dest_id,
                departure_date,
                departure_time,
                available_modes=available_modes,
                distance_km=distance_km,
            )

            if best_mode_prediction.get("method") == "gnn_error":
                return {
                    "error": f"GNN prediction failed: {best_mode_prediction.get('error')}",
                    "method": "gnn_error",
                    "origin": origin_name,
                    "destination": dest_name,
                    "day_type": temporal_features.get("day_type"),
                    "time_period": temporal_features.get("time_period"),
                }

            # Build recommendations
            recommendations = []
            
            if len(available_edges) > 0 and self.prediction_mode == "hybrid":
                # Use actual services from CSV
                for _, edge_row in available_edges.iterrows():
                    service_id = edge_row["service_id"]
                    service = self.services_df[
                        self.services_df["service_id"] == service_id
                    ].iloc[0]
                    mode = service["mode"]

                    # Calculate reliability score
                    reliability = get_reliability_score(
                        mode,
                        temporal_features["time_period"],
                        temporal_features["day_type"],
                    )
                    crowding = get_crowding_score(
                        mode,
                        temporal_features["time_period"],
                        temporal_features["day_type"],
                    )

                    # Boost score for best mode
                    if mode == best_mode_prediction["best_mode"]:
                        reliability = min(1.0, reliability + 0.2)

                    recommendations.append(
                        {
                            "service_id": service_id,
                            "mode": mode,
                            "operator": service.get("operator"),
                            "duration_min": int(service.get("base_duration_min", 0)),
                            "fare_lkr": float(service.get("base_fare", 0)),
                            "distance_km": float(service.get("distance_km", 0)),
                            "reliability": float(reliability),
                            "crowding": float(crowding),
                            "is_recommended": mode == best_mode_prediction["best_mode"],
                            "reliability_stars": "⭐" * int(round(reliability * 5)),
                        }
                    )
            else:
                # Generate synthetic recommendations using GNN predictions
                for mode in available_modes:
                    # Get GNN reliability score for this mode
                    mode_prediction = self.predict_best_mode(
                        origin_id,
                        dest_id,
                        departure_date,
                        departure_time,
                        available_modes=[mode],
                    )
                    
                    gnn_score = mode_prediction.get("score", 0.5)
                    
                    # Calculate synthetic metrics based on mode
                    if mode == "bus":
                        avg_speed_kmh = 40
                        fare_per_km = 6.0
                        operator = "SLTB (Predicted)"
                    elif mode == "train":
                        avg_speed_kmh = 50
                        fare_per_km = 3.0
                        operator = "Sri Lanka Railways (Predicted)"
                    elif mode == "ride_hail":
                        avg_speed_kmh = 50
                        fare_per_km = 120.0
                        operator = "Uber/PickMe (Predicted)"
                    else:
                        avg_speed_kmh = 40
                        fare_per_km = 10.0
                        operator = "Unknown (Predicted)"
                    
                    duration_min = int((distance_km / avg_speed_kmh) * 60)
                    fare_lkr = distance_km * fare_per_km
                    
                    # Use survey-based baseline reliability
                    reliability = get_reliability_score(
                        mode,
                        temporal_features["time_period"],
                        temporal_features["day_type"],
                    )
                    crowding = get_crowding_score(
                        mode,
                        temporal_features["time_period"],
                        temporal_features["day_type"],
                    )
                    
                    # Blend GNN score with baseline (GNN has more weight)
                    reliability = 0.7 * gnn_score + 0.3 * reliability

                    # Heuristic: late-night & short trips favor ride_hail, penalize bus
                    if (not self.model_uses_time_bucket) and temporal_features["time_period"] in {"night", "late_night"}:
                        if distance_km <= 30:
                            if mode == "ride_hail":
                                reliability = min(1.0, reliability + 0.12)
                            elif mode == "bus":
                                reliability = max(0.0, reliability - 0.12)
                    
                    # Boost score for best mode
                    if mode == best_mode_prediction["best_mode"]:
                        reliability = min(1.0, reliability + 0.15)
                    
                    recommendations.append(
                        {
                            "service_id": f"GNN_PREDICTED_{mode.upper()}",
                            "mode": mode,
                            "operator": operator,
                            "duration_min": duration_min,
                            "fare_lkr": float(fare_lkr),
                            "distance_km": float(distance_km),
                            "reliability": float(reliability),
                            "crowding": float(crowding),
                            "is_recommended": mode == best_mode_prediction["best_mode"],
                            "reliability_stars": "⭐" * int(round(reliability * 5)),
                            "is_predicted": True,  # Flag to indicate this is GNN-generated
                        }
                    )

            # Sort by reliability
            recommendations.sort(
                key=lambda x: (x["is_recommended"], x["reliability"]), reverse=True
            )
            recommendations = recommendations[:top_k]

            return {
                "origin": origin_name,
                "destination": dest_name,
                "distance_km": (
                    recommendations[0]["distance_km"] if recommendations else None
                ),
                "departure_date": departure_date,
                "departure_time": departure_time,
                "temporal_context": temporal_features,
                "best_mode": best_mode_prediction,
                "total_services": len(recommendations),
                "recommendations": recommendations,
                "data_version": self.data_version,
                "prediction_mode": self.prediction_mode,
            }

        except Exception as e:
            print(f"Error in get_recommendations: {e}")
            import traceback

            traceback.print_exc()
            return {"error": f"Internal error: {str(e)}"}

    def get_all_services(
        self,
        departure_date: Optional[str] = None,
        departure_time: Optional[str] = None,
    ) -> Dict:
        """List all services with reliability/crowding scores for a given temporal context."""
        try:
            if self.edges_df is None:
                return {"error": "Service data not loaded"}

            self._ensure_services_df()

            temporal_features = self._get_temporal_features(departure_date, departure_time)
            day_type = temporal_features["day_type"]
            time_period = temporal_features["time_period"]

            services = []
            for _, svc in self.services_df.iterrows():
                reliability = get_reliability_score(svc["mode"], time_period, day_type)
                crowding = get_crowding_score(svc["mode"], time_period, day_type)
                services.append(
                    {
                        "service_id": svc["service_id"],
                        "mode": svc["mode"],
                        "operator": svc.get("operator"),
                        "duration_min": int(svc.get("base_duration_min", 0)),
                        "fare_lkr": float(svc.get("base_fare", 0)),
                        "distance_km": float(svc.get("distance_km", 0)),
                        "reliability": float(reliability),
                        "crowding": float(crowding),
                        "temporal_context": temporal_features,
                    }
                )

            services.sort(key=lambda x: x["reliability"], reverse=True)

            return {
                "total_services": len(services),
                "services": services,
                "temporal_context": temporal_features,
                "data_version": self.data_version,
            }

        except Exception as e:
            print(f"Error in get_all_services: {e}")
            return {"error": f"Internal error: {str(e)}"}


# Keep old class name for backward compatibility
TransportServiceGNN = TransportServiceGNNRefactored
