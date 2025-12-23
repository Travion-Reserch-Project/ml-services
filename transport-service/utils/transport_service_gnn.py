"""
Transport Service with GNN Integration
Combines NLP parsing with trained GNN model for transport recommendations
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from torch_geometric.nn import GCNConv
from .data_repository import DataRepository, create_repository


class TransportGNN(nn.Module):
    """GNN model architecture (must match training notebook)."""
    
    def __init__(self, node_features, edge_features, hidden_dim=64, output_dim=1):
        super(TransportGNN, self).__init__()
        
        # Graph convolution layers
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Edge prediction layers
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()  # Output 0-1 rating
        )
        
    def forward(self, x, edge_index, edge_attr, target_edges):
        # Message passing to learn node embeddings
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        
        # Predict ratings for target edges
        src_nodes = edge_index[0, target_edges]
        dst_nodes = edge_index[1, target_edges]
        
        # Concatenate source, destination embeddings and edge features
        edge_embeddings = torch.cat([
            x[src_nodes],
            x[dst_nodes],
            edge_attr[target_edges]
        ], dim=1)
        
        # Predict rating
        predictions = self.edge_mlp(edge_embeddings).squeeze()
        
        return predictions


class TransportServiceGNN:
    """
    Main service class integrating GNN model for transport recommendations.
    """
    
    def __init__(self, model_path: str, data_path: str = None, use_mongodb: bool = False):
        """
        Initialize the transport service.
        
        Args:
            model_path: Path to trained GNN model (.pth file)
            data_path: Path to transport data directory (used for CSV; ignored if use_mongodb=True)
            use_mongodb: If True, use MongoDB; otherwise use CSV
        """
        self.model = None
        self.artifacts = {}
        self.data_version = None
        
        # Initialize data repository (MongoDB or CSV)
        try:
            use_mongo = use_mongodb or os.getenv('DATA_SOURCE') == 'mongodb'
            self.repository = create_repository(use_mongo=use_mongo)
            self.data_version = self.repository.get_data_version()
        except Exception as e:
            print(f"⚠️  Repository init failed: {e}")
            self.repository = None
        
        # Load model
        self._load_model(model_path)
        
        # Load data through repository
        self._load_data()
    
    def _load_model(self, model_path: str):
        """Load the trained GNN model and artifacts."""
        try:
            if not os.path.exists(model_path):
                print(f"⚠️ Model not found at {model_path}")
                return
            
            # Load checkpoint; may be a dict bundle or a raw torch module
            checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')

            # Guard: ensure expected training bundle structure
            if not isinstance(checkpoint, dict):
                print("⚠️ Incompatible checkpoint format. Expected a dict with training artifacts (node_features, edge_index, edge_attr, model_state_dict).")
                print("   Tip: save with torch.save({'model_state_dict': model.state_dict(), 'node_features': ..., 'edge_index': ..., 'edge_attr': ...}, path)")
                self.model = None
                return
            required_keys = {'node_features', 'edge_attr', 'edge_index', 'model_state_dict'}
            if not required_keys.issubset(checkpoint.keys()):
                missing = required_keys.difference(checkpoint.keys())
                print(f"⚠️ Incompatible checkpoint: missing keys {missing}")
                self.model = None
                return
            
            # Initialize model architecture
            node_features = checkpoint['node_features']
            edge_attr = checkpoint['edge_attr']
            
            self.model = TransportGNN(
                node_features=node_features.shape[1],
                edge_features=edge_attr.shape[1],
                hidden_dim=64
            )
            
            # Load trained weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Store artifacts
            self.artifacts = {
                'model': self.model,
                'node_features': checkpoint['node_features'],
                'edge_index': checkpoint['edge_index'],
                'edge_attr': checkpoint['edge_attr'],
                'region_encoder': checkpoint.get('region_encoder'),
                'type_encoder': checkpoint.get('type_encoder'),
                'mode_encoder': checkpoint.get('mode_encoder'),
                'scaler': checkpoint.get('scaler')
            }
            
            print(f"✅ GNN model loaded successfully from {model_path}")
            
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
            
            print(f"✅ Data loaded: {len(self.nodes_df)} nodes, "
                  f"{len(self.services_df)} services, {len(self.edges_df)} edges")
            print(f"   Data version: {self.data_version}")
            
        except Exception as e:
            print(f"⚠️ Could not load data: {e}")
            import traceback
            traceback.print_exc()
            self.nodes_df = None
            self.services_df = None
            self.edges_df = None
    
    def predict_service_ratings(self, service_indices: List[int]) -> np.ndarray:
        """
        🤖 THIS IS WHERE THE MODEL MAKES PREDICTIONS 🤖
        
        The GNN model predicts quality ratings (1-5 stars) for transport services.
        It learned from historical data which routes/services perform better.
        
        Args:
            service_indices: List of edge indices (row numbers from edges.csv)
            
        Returns:
            Ratings on 1-5 scale (higher = better service quality)
            
        Note: This only predicts ratings. Service details (fare, operator, etc.) 
        come from CSV files in get_recommendations().
        """
        if self.model is None:
            return np.array([3.0] * len(service_indices))  # Default rating
        
        model = self.artifacts['model']
        node_features = self.artifacts['node_features']  # Pre-computed features from training
        edge_index = self.artifacts['edge_index']        # Graph structure from training
        edge_attr = self.artifacts['edge_attr']          # Edge features from training
        
        # Convert to tensor
        service_indices_tensor = torch.tensor(service_indices, dtype=torch.long)
        
        # 🔮 Make prediction using trained GNN
        model.eval()
        with torch.no_grad():
            predictions = model(node_features, edge_index, edge_attr, service_indices_tensor)
        
        # Handle scalar output (single prediction) vs. tensor (multiple)
        if isinstance(predictions, torch.Tensor):
            if predictions.dim() == 0:
                predictions = predictions.unsqueeze(0)
            predictions_array = predictions.numpy()
        else:
            predictions_array = np.array([predictions]) if np.isscalar(predictions) else predictions
        
        # Convert from 0-1 scale to 1-5 scale
        ratings = (predictions_array * 4) + 1
        
        # Ensure output is always array
        return np.atleast_1d(ratings)
    
    def get_recommendations(
        self, 
        origin: str, 
        destination: str,
        departure_date: Optional[str] = None,  # Format: "2025-12-25"
        departure_time: Optional[str] = None,  # Format: "14:30" or "2:30 PM"
        top_k: int = 5
    ) -> Dict:
        """
        Get ranked transport recommendations for origin → destination.
        
        Args:
            origin: Origin location name
            destination: Destination location name
            departure_date: Date in YYYY-MM-DD format (e.g., "2025-12-25")
            departure_time: Time in HH:MM format (e.g., "14:30")
            top_k: Number of recommendations to return
            
        Returns:
            Dictionary with recommendations sorted by rating
        """
        try:
            # 📍 Step 1: Look up locations in database (CSV)
            origin_matches = self.nodes_df[
                self.nodes_df['name'].str.contains(origin, case=False, na=False)
            ]
            dest_matches = self.nodes_df[
                self.nodes_df['name'].str.contains(destination, case=False, na=False)
            ]
            
            if len(origin_matches) == 0:
                return {
                    "error": f"Origin location '{origin}' not found",
                    "available_locations": self.nodes_df['name'].tolist()
                }
            
            if len(dest_matches) == 0:
                return {
                    "error": f"Destination location '{destination}' not found",
                    "available_locations": self.nodes_df['name'].tolist()
                }
            
            origin_id = origin_matches.iloc[0]['location_id']
            dest_id = dest_matches.iloc[0]['location_id']
            origin_name = origin_matches.iloc[0]['name']
            dest_name = dest_matches.iloc[0]['name']
            
            # 🚌 Step 2: Find which services connect these locations (from repository)
            available_edges = self.repository.get_edges_between(origin_id, dest_id)
            
            if len(available_edges) == 0:
                return {
                    "error": f"No direct services found from {origin_name} to {dest_name}",
                    "origin": origin_name,
                    "destination": dest_name
                }
            
            # Get edge indices for prediction
            edge_indices = available_edges.index.tolist()
            
            # Get temporal context
            temporal_context = self._get_temporal_context(departure_date, departure_time)
            
            # 🤖 Step 3: Use GNN model to predict quality ratings for these services
            # The model learned from historical data which services perform better
            ratings = self.predict_service_ratings(edge_indices)
            
            # ⏰ Step 4: Adjust ratings based on time/date (peak hours, holidays, etc.)
            if temporal_context:
                ratings = self._adjust_ratings_for_conditions(
                    ratings, edge_indices, temporal_context
                )
            
            # 📦 Step 5: Combine CSV data (service details) with model predictions (ratings)
            recommendations = []
            for i, edge_idx in enumerate(edge_indices):
                edge_row = available_edges.iloc[i]
                service_id = edge_row['service_id']
                # Get service details from CSV (operator, fare, etc.)
                service = self.services_df[self.services_df['service_id'] == service_id].iloc[0]
                
                recommendations.append({
                    "service_id": service_id,
                    "mode": service['mode'],
                    "operator": service['operator'],
                    "duration_min": int(service['base_duration_min']),
                    "fare_lkr": float(service['base_fare']),
                    "distance_km": float(service['distance_km']),
                    "rating": float(ratings[i]),
                    "rating_stars": "⭐" * int(round(ratings[i])),
                    "temporal_context": temporal_context if temporal_context else None
                })
            
            # Sort by rating (best first)
            recommendations.sort(key=lambda x: x['rating'], reverse=True)
            
            # Limit to top_k
            recommendations = recommendations[:top_k]
            
            return {
                "origin": origin_name,
                "destination": dest_name,
                "distance_km": recommendations[0]['distance_km'] if recommendations else None,
                "departure_date": departure_date,
                "departure_time": departure_time,
                "temporal_info": temporal_context,
                "total_services": len(recommendations),
                "recommendations": recommendations,
                "best_option": recommendations[0] if recommendations else None,
                "data_version": self.data_version,
                "note": "Ratings adjusted for time/date conditions" if temporal_context else "Base ratings (no temporal context)"
            }
            
        except Exception as e:
            print(f"Error in get_recommendations: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": f"Internal error: {str(e)}"
            }
    
    def query(self, natural_language_query: str) -> Dict:
        """
        Process a natural language query.
        
        Args:
            natural_language_query: e.g., "How can I go from Colombo to Anuradhapura?"
            
        Returns:
            Recommendations dictionary
        """
        # Simple NLP extraction (can be enhanced with your NLP parser)
        query_lower = natural_language_query.lower()
        
        # Extract origin/destination using simple pattern matching
        # (You can integrate your sophisticated NLP parser here)
        origin = None
        destination = None
        
        # Look for location names in the query
        for location in self.nodes_df['name'].tolist():
            if location.lower() in query_lower:
                if origin is None:
                    origin = location
                elif destination is None:
                    destination = location
        
        if origin is None or destination is None:
            return {
                "error": "Could not extract origin and destination from query",
                "hint": "Try: 'How to go from [origin] to [destination]?'",
                "available_locations": self.nodes_df['name'].tolist()
            }
        
        return self.get_recommendations(origin, destination)
    
    def get_all_services(self) -> Dict:
        """Get all available services with ratings."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Predict for all edges
            all_indices = list(range(len(self.edges_df)))
            ratings = self.predict_service_ratings(all_indices)
            
            services = []
            for i, (idx, edge_row) in enumerate(self.edges_df.iterrows()):
                service_id = edge_row['service_id']
                service = self.services_df[self.services_df['service_id'] == service_id].iloc[0]
                origin = self.nodes_df[self.nodes_df['location_id'] == edge_row['origin_id']].iloc[0]
                dest = self.nodes_df[self.nodes_df['location_id'] == edge_row['destination_id']].iloc[0]
                
                services.append({
                    "service_id": service_id,
                    "origin": origin['name'],
                    "destination": dest['name'],
                    "mode": service['mode'],
                    "operator": service['operator'],
                    "duration_min": int(service['base_duration_min']),
                    "fare_lkr": float(service['base_fare']),
                    "rating": float(ratings[i])
                })
            
            # Sort by rating
            services.sort(key=lambda x: x['rating'], reverse=True)
            
            return {
                "total_services": len(services),
                "services": services
            }
            
        except Exception as e:
            return {"error": f"Could not get services: {str(e)}"}
    
    def _get_temporal_context(self, departure_date: Optional[str], departure_time: Optional[str]) -> Optional[Dict]:
        """Extract temporal features from date/time."""
        if not departure_date and not departure_time:
            return None
        
        from datetime import datetime
        
        context = {}
        
        # Parse date
        if departure_date:
            try:
                date_obj = datetime.strptime(departure_date, "%Y-%m-%d")
                context['date'] = departure_date
                context['day_of_week'] = date_obj.strftime("%A")
                context['is_weekend'] = date_obj.weekday() >= 5
                
                # Check if poya day (simplified - would need calendar.csv lookup)
                # For now, assume full moon days
                day = date_obj.day
                context['is_poya'] = day in [8, 15, 23, 30]  # Approximate poya days
                
                context['day_type'] = 'poya' if context['is_poya'] else ('weekend' if context['is_weekend'] else 'regular')
            except:
                pass
        
        # Parse time
        if departure_time:
            try:
                # Handle both HH:MM and "H:MM AM/PM" formats
                if 'AM' in departure_time.upper() or 'PM' in departure_time.upper():
                    time_obj = datetime.strptime(departure_time, "%I:%M %p")
                else:
                    time_obj = datetime.strptime(departure_time, "%H:%M")
                
                hour = time_obj.hour
                context['time'] = departure_time
                context['hour'] = hour
                
                # Determine time period (matching service_conditions.csv)
                if 5 <= hour < 8:
                    context['time_period'] = 'early_morning'
                elif 8 <= hour < 12:
                    context['time_period'] = 'morning'  # Peak
                elif 12 <= hour < 17:
                    context['time_period'] = 'day'
                elif 17 <= hour < 21:
                    context['time_period'] = 'evening'  # Peak
                else:
                    context['time_period'] = 'night'
                
                context['is_peak_hour'] = context['time_period'] in ['morning', 'evening']
            except:
                pass
        
        return context if context else None
    
    def _adjust_ratings_for_conditions(
        self, 
        ratings: np.ndarray, 
        edge_indices: List[int],
        temporal_context: Dict
    ) -> np.ndarray:
        """
        Adjust ratings based on temporal conditions (crowdedness, delays).
        
        NOTE: This is a temporary workaround. Ideally, the GNN model should
        learn to predict ratings given temporal features as input.
        """
        try:
            # Load service conditions if available
            conditions_path = 'data/service_conditions.csv'
            if not hasattr(self, 'conditions_df'):
                if os.path.exists(conditions_path):
                    self.conditions_df = pd.read_csv(conditions_path)
                else:
                    return ratings
            
            adjusted_ratings = ratings.copy()
            time_period = temporal_context.get('time_period')
            day_type = temporal_context.get('day_type', 'regular')
            
            # Adjust each service based on conditions
            for i, edge_idx in enumerate(edge_indices):
                edge_row = self.edges_df.iloc[edge_idx]
                service_id = edge_row['service_id']
                
                # Find matching condition
                condition = self.conditions_df[
                    (self.conditions_df['service_id'] == service_id) &
                    (self.conditions_df['time_period'] == time_period) &
                    (self.conditions_df['day_type'] == day_type)
                ]
                
                if len(condition) > 0:
                    crowdedness = condition.iloc[0]['crowdedness_level']
                    typical_delay = condition.iloc[0]['typical_delay_min']
                    
                    # Penalize high crowdedness and delays
                    crowdedness_penalty = {
                        'low': 0.0,
                        'medium': -0.2,
                        'high': -0.5,
                        'very_high': -0.8
                    }.get(crowdedness, 0.0)
                    
                    delay_penalty = -0.1 * (typical_delay / 10)  # -0.1 per 10 min delay
                    
                    adjusted_ratings[i] = max(1.0, ratings[i] + crowdedness_penalty + delay_penalty)
            
            return adjusted_ratings
            
        except Exception as e:
            print(f"Warning: Could not adjust for conditions: {e}")
            return ratings

