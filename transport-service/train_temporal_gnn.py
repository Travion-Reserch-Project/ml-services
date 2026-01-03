"""
Example: Training the Temporal GNN Model

This script shows how to train the new TemporalTransportGNN model
using the survey data rules and timetables.

Note: This is a guide. Adapt to your actual training data/labels.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from typing import Tuple, List

# Import refactored components
from utils.temporal_gnn_model import TemporalTransportGNN
from utils.holiday_detector import HolidayDetector
from utils.temporal_features import (
    get_reliability_score,
    get_crowding_score,
    create_survey_based_reliability_rules,
)


def create_transport_graph() -> Data:
    """Create PyTorch Geometric graph from nodes and edges."""
    # Load data
    nodes_df = pd.read_csv("data/nodes.csv")
    edges_df = pd.read_csv("data/edges.csv")
    
    num_nodes = len(nodes_df)
    
    # Create node features (simple: region one-hot + coordinates)
    regions = nodes_df['region'].unique()
    region_to_idx = {r: i for i, r in enumerate(regions)}
    
    region_one_hot = np.zeros((num_nodes, len(regions)))
    for i, region in enumerate(nodes_df['region']):
        region_to_idx_val = region_to_idx.get(region, 0)
        region_one_hot[i, region_to_idx_val] = 1
    
    # Normalize coordinates
    lats = nodes_df['latitude'].values
    lons = nodes_df['longitude'].values
    lat_norm = (lats - lats.min()) / (lats.max() - lats.min())
    lon_norm = (lons - lons.min()) / (lons.max() - lons.min())
    
    coords = np.column_stack([lat_norm, lon_norm])
    
    # Combine features
    node_features = np.hstack([region_one_hot, coords])
    x = torch.tensor(node_features, dtype=torch.float32)
    
    # Create edge index from routes
    origin_ids = edges_df['origin_id'].values - 1  # 0-indexed
    dest_ids = edges_df['destination_id'].values - 1
    
    edge_index = torch.tensor(
        np.vstack([origin_ids, dest_ids]),
        dtype=torch.long
    )
    
    data = Data(x=x, edge_index=edge_index)
    return data, nodes_df, edges_df


def create_training_samples(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    num_samples: int = 1000
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create training samples from survey data and timetables.
    
    Returns:
        origin_ids, dest_ids, day_types, time_periods, mode_ids, scores, weights
    """
    modes = ["bus", "train", "ridehailing"]
    mode_to_id = {m: i for i, m in enumerate(modes)}
    
    day_types = ["regular", "weekend", "poya", "holiday"]
    day_type_to_id = {d: i for i, d in enumerate(day_types)}
    
    time_periods = ["early_morning", "morning", "day", "evening", "night", "late_night"]
    time_period_to_id = {t: i for i, t in enumerate(time_periods)}
    
    # Generate random training samples
    num_nodes = len(nodes_df)
    
    origin_ids_list = []
    dest_ids_list = []
    day_type_ids_list = []
    time_period_ids_list = []
    mode_ids_list = []
    scores_list = []
    
    np.random.seed(42)
    
    for _ in range(num_samples):
        # Sample random origin/destination
        origin_id = np.random.randint(0, num_nodes)
        dest_id = np.random.randint(0, num_nodes)
        
        if origin_id == dest_id:
            continue
        
        # Sample random temporal features
        day_type = np.random.choice(day_types)
        time_period = np.random.choice(time_periods)
        
        # Sample random mode
        mode = np.random.choice(modes)
        
        # Get reliability score from survey data
        reliability = get_reliability_score(mode, time_period, day_type)
        
        # Add some noise to make training interesting
        score = reliability + np.random.normal(0, 0.05)
        score = np.clip(score, 0, 1)
        
        origin_ids_list.append(origin_id)
        dest_ids_list.append(dest_id)
        day_type_ids_list.append(day_type_to_id[day_type])
        time_period_ids_list.append(time_period_to_id[time_period])
        mode_ids_list.append(mode_to_id[mode])
        scores_list.append(score)
    
    # Convert to tensors
    origin_ids = torch.tensor(origin_ids_list, dtype=torch.long)
    dest_ids = torch.tensor(dest_ids_list, dtype=torch.long)
    day_type_ids = torch.tensor(day_type_ids_list, dtype=torch.long)
    time_period_ids = torch.tensor(time_period_ids_list, dtype=torch.long)
    mode_ids = torch.tensor(mode_ids_list, dtype=torch.long)
    scores = torch.tensor(scores_list, dtype=torch.float32)
    
    # Sample weights (uniform for now, can be adjusted)
    weights = torch.ones(len(origin_ids))
    
    return origin_ids, dest_ids, day_type_ids, time_period_ids, mode_ids, scores, weights


def train_model(
    model: TemporalTransportGNN,
    data: Data,
    origin_ids: torch.Tensor,
    dest_ids: torch.Tensor,
    day_type_ids: torch.Tensor,
    time_period_ids: torch.Tensor,
    mode_ids: torch.Tensor,
    scores: torch.Tensor,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001
):
    """Train the temporal GNN model."""
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    num_samples = len(origin_ids)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        # Shuffle data
        indices = torch.randperm(num_samples)
        origin_ids_shuffled = origin_ids[indices]
        dest_ids_shuffled = dest_ids[indices]
        day_type_ids_shuffled = day_type_ids[indices]
        time_period_ids_shuffled = time_period_ids[indices]
        mode_ids_shuffled = mode_ids[indices]
        scores_shuffled = scores[indices]
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            
            batch_origin = origin_ids_shuffled[start_idx:end_idx]
            batch_dest = dest_ids_shuffled[start_idx:end_idx]
            batch_day_type = day_type_ids_shuffled[start_idx:end_idx]
            batch_time_period = time_period_ids_shuffled[start_idx:end_idx]
            batch_mode = mode_ids_shuffled[start_idx:end_idx]
            batch_scores = scores_shuffled[start_idx:end_idx]
            
            # Forward pass
            predictions = model(
                data.x,
                data.edge_index,
                batch_origin,
                batch_dest,
                batch_day_type,
                batch_time_period,
                batch_mode
            )
            
            # Compute loss
            loss = loss_fn(predictions.squeeze(), batch_scores)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    print("✅ Training complete!")


def save_model(
    model: TemporalTransportGNN,
    data: Data,
    output_path: str = "model/transport_gnn_model.pth"
):
    """Save trained model and artifacts."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'node_features': data.x,
        'edge_index': data.edge_index,
    }, output_path)
    print(f"✅ Model saved to {output_path}")


def main():
    """Main training script."""
    
    print("=" * 60)
    print("Training Temporal Transport GNN Model")
    print("=" * 60)
    
    # Step 1: Create graph
    print("\n📊 Creating transport graph...")
    data, nodes_df, edges_df = create_transport_graph()
    print(f"   Nodes: {data.x.shape[0]}")
    print(f"   Edges: {data.edge_index.shape[1]}")
    print(f"   Node features: {data.x.shape[1]}")
    
    # Step 2: Create training samples
    print("\n📝 Creating training samples from survey data...")
    origin_ids, dest_ids, day_type_ids, time_period_ids, mode_ids, scores, weights = \
        create_training_samples(nodes_df, edges_df, num_samples=1000)
    print(f"   Training samples: {len(origin_ids)}")
    print(f"   Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    
    # Step 3: Initialize model
    print("\n🧠 Initializing TemporalTransportGNN...")
    model = TemporalTransportGNN(
        node_features=data.x.shape[1],
        num_day_types=4,        # regular, weekend, poya, holiday
        num_time_periods=6,     # early_morning, ..., late_night
        num_modes=3,            # bus, train, ridehailing
        hidden_dim=64,
        use_attention=False     # Set to True for GAT layers
    )
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 4: Train
    print("\n🚀 Training model...")
    train_model(
        model,
        data,
        origin_ids,
        dest_ids,
        day_type_ids,
        time_period_ids,
        mode_ids,
        scores,
        num_epochs=100,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Step 5: Save
    print("\n💾 Saving model...")
    save_model(model, data)
    
    # Step 6: Test
    print("\n✅ Testing trained model...")
    model.eval()
    with torch.no_grad():
        # Test prediction for Colombo → Anuradhapura on morning/regular day
        test_origin = torch.tensor([0])  # First location
        test_dest = torch.tensor([4])    # Another location
        test_day = torch.tensor([0])     # regular
        test_time = torch.tensor([1])    # morning
        
        for mode, mode_name in [(0, "bus"), (1, "train"), (2, "ridehailing")]:
            test_mode = torch.tensor([mode])
            score = model(data.x, data.edge_index, test_origin, test_dest, 
                         test_day, test_time, test_mode)
            print(f"   {mode_name}: {score.item():.3f}")
    
    print("\n" + "=" * 60)
    print("✅ Training complete! Model ready for inference.")
    print("=" * 60)


if __name__ == "__main__":
    main()
