"""
Temporal-Aware Transport GNN Model
Predicts best transport method based on:
- Origin/Destination locations
- Date (day type: regular, weekend, poya)
- Time (time period: early morning, morning, etc.)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from typing import Tuple


class LegacyTemporalTransportGNN(nn.Module):
    """
    Legacy temporal GNN that uses coarse time periods (no time buckets/day-of-week).
    This matches older checkpoints with `time_period_embedding` only.
    """

    def __init__(
        self,
        node_features: int,
        num_day_types: int = 2,
        num_time_periods: int = 5,
        num_modes: int = 3,
        hidden_dim: int = 64,
        output_dim: int = 1,
        use_attention: bool = False,
    ):
        super(LegacyTemporalTransportGNN, self).__init__()

        self.node_features = node_features
        self.num_day_types = num_day_types
        self.num_time_periods = num_time_periods
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # ===== LOCATION ENCODER =====
        if use_attention:
            self.conv1 = GATConv(node_features, hidden_dim, heads=2)
            self.conv2 = GATConv(hidden_dim * 2, hidden_dim, heads=2)
            self.conv3 = GATConv(hidden_dim * 2, hidden_dim)
        else:
            self.conv1 = GCNConv(node_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # ===== TEMPORAL EMBEDDINGS (legacy: day_type + time_period) =====
        temp_dim = hidden_dim // 2  # matches legacy checkpoints (e.g., 32 when hidden_dim=64)
        self.day_type_embedding = nn.Embedding(num_day_types, temp_dim)
        self.time_period_embedding = nn.Embedding(num_time_periods, temp_dim)

        # Fusion for legacy temporal embeddings
        self.temporal_fusion = nn.Sequential(
            nn.Linear(temp_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # ===== MODE EMBEDDING =====
        self.mode_embedding = nn.Embedding(num_modes, hidden_dim // 2)

        prediction_input_dim = hidden_dim * 2 + hidden_dim + hidden_dim // 2
        self.prediction_head = nn.Sequential(
            nn.Linear(prediction_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid(),
        )

        self.output_activation = nn.Identity()

    def encode_locations(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        return x

    def encode_temporal(self, day_type_ids: torch.Tensor, time_period_ids: torch.Tensor) -> torch.Tensor:
        day_emb = self.day_type_embedding(day_type_ids)
        time_emb = self.time_period_embedding(time_period_ids)
        temporal_combined = torch.cat([day_emb, time_emb], dim=1)
        return self.temporal_fusion(temporal_combined)

    def encode_mode(self, mode_ids: torch.Tensor) -> torch.Tensor:
        return self.mode_embedding(mode_ids)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        origin_ids: torch.Tensor,
        destination_ids: torch.Tensor,
        day_type_ids: torch.Tensor,
        time_period_ids: torch.Tensor,
        mode_ids: torch.Tensor,
    ) -> torch.Tensor:
        location_embeddings = self.encode_locations(x, edge_index)
        origin_emb = location_embeddings[origin_ids]
        dest_emb = location_embeddings[destination_ids]
        route_emb = torch.cat([origin_emb, dest_emb], dim=1)

        temporal_emb = self.encode_temporal(day_type_ids, time_period_ids)
        mode_emb = self.encode_mode(mode_ids)
        combined_features = torch.cat([route_emb, temporal_emb, mode_emb], dim=1)
        score = self.prediction_head(combined_features)
        return self.output_activation(score)


class TemporalTransportGNN(nn.Module):
    """
    GNN model with temporal awareness for transport recommendations.
    
    Architecture:
    1. Location embeddings (from GCN on transport network)
    2. Temporal embeddings (day type + time period)
    3. Service mode embeddings
    4. Combined prediction head: which service is best for this trip
    """

    def __init__(
        self,
        node_features: int,
        num_day_types: int = 4,  # regular, weekend, poya, holiday
        num_time_buckets: int = 8,  # finer-grained time buckets
        num_day_of_week: int = 7,  # Monday-Sunday
        num_modes: int = 3,  # bus, train, ridehailing
        hidden_dim: int = 64,
        output_dim: int = 1,
        use_attention: bool = False,
    ):
        """
        Initialize temporal GNN.

        Args:
            node_features: Feature dimension for location nodes
            num_day_types: Number of day type categories
            num_time_periods: Number of time period categories
            num_modes: Number of transport modes
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for rating, 3 for mode ranking)
            use_attention: If True, use GAT layers instead of GCN
        """
        super(TemporalTransportGNN, self).__init__()

        self.node_features = node_features
        self.num_day_types = num_day_types
        self.num_time_buckets = num_time_buckets
        self.num_day_of_week = num_day_of_week
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # ===== LOCATION ENCODER (GCN on transport network) =====
        if use_attention:
            self.conv1 = GATConv(node_features, hidden_dim, heads=2)
            self.conv2 = GATConv(hidden_dim * 2, hidden_dim, heads=2)
            self.conv3 = GATConv(hidden_dim * 2, hidden_dim)
        else:
            self.conv1 = GCNConv(node_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # ===== TEMPORAL EMBEDDINGS =====
        # Embed day type (regular, weekend, poya, holiday)
        temp_dim = hidden_dim // 3
        self.day_type_embedding = nn.Embedding(num_day_types, temp_dim)

        # Embed finer-grained time buckets (8 buckets)
        self.time_bucket_embedding = nn.Embedding(num_time_buckets, temp_dim)

        # Embed day of week (0-6)
        self.day_of_week_embedding = nn.Embedding(num_day_of_week, temp_dim)

        # Temporal fusion layer (concat of 3 embeds -> hidden_dim)
        self.temporal_fusion = nn.Sequential(
            nn.Linear(temp_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # ===== SERVICE MODE EMBEDDING =====
        self.mode_embedding = nn.Embedding(num_modes, hidden_dim // 2)

        # ===== PREDICTION HEAD =====
        # Takes: origin_emb (hidden_dim) + dest_emb (hidden_dim) + temporal_emb (hidden_dim) + mode_emb (hidden_dim//2)
        prediction_input_dim = hidden_dim * 2 + hidden_dim + hidden_dim // 2

        self.prediction_head = nn.Sequential(
            nn.Linear(prediction_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid(),
        )

        # Output activation kept as identity to avoid double-sigmoid during loading
        self.output_activation = nn.Identity()

    def encode_locations(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Encode location nodes using graph convolution.

        Args:
            x: Node features (num_nodes, node_features)
            edge_index: Graph edges (2, num_edges)

        Returns:
            Node embeddings (num_nodes, hidden_dim)
        """
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        return x

    def encode_temporal(
        self,
        day_type_ids: torch.Tensor,
        time_bucket_ids: torch.Tensor,
        day_of_week_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode temporal information (day type + time bucket + day of week).

        Args:
            day_type_ids: Day type IDs (batch_size,) - 0:regular, 1:weekend, 2:poya, 3:holiday
            time_bucket_ids: Time bucket IDs (batch_size,) - finer-grained buckets
            day_of_week_ids: Day of week IDs (batch_size,) - 0:Monday ... 6:Sunday

        Returns:
            Temporal embeddings (batch_size, hidden_dim)
        """
        day_emb = self.day_type_embedding(day_type_ids)
        time_emb = self.time_bucket_embedding(time_bucket_ids)
        dow_emb = self.day_of_week_embedding(day_of_week_ids)

        # Concatenate and fuse
        temporal_combined = torch.cat([day_emb, time_emb, dow_emb], dim=1)
        temporal_encoded = self.temporal_fusion(temporal_combined)

        return temporal_encoded

    def encode_mode(self, mode_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode service mode (bus, train, ridehailing).

        Args:
            mode_ids: Mode IDs (batch_size,) - 0:bus, 1:train, 2:ridehailing

        Returns:
            Mode embeddings (batch_size, hidden_dim//2)
        """
        return self.mode_embedding(mode_ids)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        origin_ids: torch.Tensor,
        destination_ids: torch.Tensor,
        day_type_ids: torch.Tensor,
        time_bucket_ids: torch.Tensor,
        day_of_week_ids: torch.Tensor,
        mode_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict suitability score for a transport option.

        Args:
            x: Node features
            edge_index: Graph edges
            origin_ids: Origin location IDs (batch_size,)
            destination_ids: Destination location IDs (batch_size,)
            day_type_ids: Day type IDs (batch_size,)
            time_period_ids: Time period IDs (batch_size,)
            mode_ids: Mode IDs (batch_size,)

        Returns:
            Suitability scores (batch_size, 1) - 0 to 1
        """
        # Encode locations
        location_embeddings = self.encode_locations(x, edge_index)  # (num_nodes, hidden_dim)

        # Get origin and destination embeddings
        origin_emb = location_embeddings[origin_ids]  # (batch, hidden_dim)
        dest_emb = location_embeddings[destination_ids]  # (batch, hidden_dim)

        # Combine origin and destination
        route_emb = torch.cat([origin_emb, dest_emb], dim=1)  # (batch, hidden_dim * 2)

        # Encode temporal context
        temporal_emb = self.encode_temporal(day_type_ids, time_bucket_ids, day_of_week_ids)  # (batch, hidden_dim)

        # Encode mode
        mode_emb = self.encode_mode(mode_ids)  # (batch, hidden_dim//2)

        # Concatenate all features
        combined_features = torch.cat([route_emb, temporal_emb, mode_emb], dim=1)

        # Predict
        score = self.prediction_head(combined_features)  # (batch, output_dim)
        score = self.output_activation(score)  # 0-1 rating

        return score

    def predict_best_mode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        origin_ids: torch.Tensor,
        destination_ids: torch.Tensor,
        day_type_ids: torch.Tensor,
        time_period_ids: torch.Tensor,
        available_modes: list = [0, 1, 2],  # bus, train, ridehailing
    ) -> Tuple[int, float]:
        """
        Predict the best transport mode for given origin/destination/time.

        Args:
            x: Node features
            edge_index: Graph edges
            origin_ids: Origin location IDs
            destination_ids: Destination location IDs
            day_type_ids: Day type IDs
            time_period_ids: Time period IDs
            available_modes: List of available mode IDs to consider

        Returns:
            Tuple of (best_mode_id, score)
        """
        with torch.no_grad():
            best_score = -1
            best_mode = None

            for mode_id in available_modes:
                mode_ids = torch.full_like(origin_ids, mode_id)
                score = self.forward(
                    x,
                    edge_index,
                    origin_ids,
                    destination_ids,
                    day_type_ids,
                    time_period_ids,
                    mode_ids,
                )
                score_val = score.mean().item()

                if score_val > best_score:
                    best_score = score_val
                    best_mode = mode_id

        return best_mode, best_score
