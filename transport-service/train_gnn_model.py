"""
Train GNN Model on Transport Data

This script trains the GNN model using:
- edges.csv: Routes between locations with service info (mode, operator, distance, duration, fare)
- nodes.csv: Location information (name, type, coordinates, region)

The model learns to predict service ratings based on:
- Route characteristics (origin, destination, distance)
- Service attributes (mode, operator, fare, duration)
- Location features through graph neural network embeddings
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import Data
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from datetime import datetime


class TransportGNN(nn.Module):
    """GNN model for transport service reliability prediction."""
    
    def __init__(self, node_features, edge_features, hidden_dim=64, output_dim=1):
        super(TransportGNN, self).__init__()
        
        # Graph convolution layers for location embeddings
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # MLP for edge/service rating prediction
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()  # Output 0-1 reliability rating
        )
        
    def forward(self, x, edge_index, edge_attr, target_edges):
        """
        Forward pass through GNN.
        
        Args:
            x: Node features (num_nodes, node_features)
            edge_index: Graph edges (2, num_edges)
            edge_attr: Edge features (num_edges, edge_features)
            target_edges: Indices of edges to predict (num_targets,)
            
        Returns:
            Predictions for target edges (num_targets,)
        """
        # GCN layers to learn location embeddings
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        
        # Get source and destination node embeddings for target edges
        src_nodes = edge_index[0, target_edges]
        dst_nodes = edge_index[1, target_edges]
        
        # Concatenate embeddings with edge features
        edge_embeddings = torch.cat([
            x[src_nodes],
            x[dst_nodes],
            edge_attr[target_edges]
        ], dim=1)
        
        # Predict reliability rating
        predictions = self.edge_mlp(edge_embeddings).squeeze()
        
        return predictions


def load_and_prepare_data(data_path="data"):
    """Load and prepare training data from CSV files."""
    
    print("📊 Loading data files...")
    
    # Load base data
    nodes_df = pd.read_csv(os.path.join(data_path, "nodes.csv"))
    services_df = pd.read_csv(os.path.join(data_path, "services.csv"))
    edges_df = pd.read_csv(os.path.join(data_path, "edges.csv"))
    conditions_df = pd.read_csv(os.path.join(data_path, "service_conditions.csv"))
    timetables_df = pd.read_csv(os.path.join(data_path, "timetables.csv"))
    
    print(f"  ✓ Nodes: {len(nodes_df)}")
    print(f"  ✓ Services: {len(services_df)}")
    print(f"  ✓ Edges: {len(edges_df)}")
    print(f"  ✓ Conditions: {len(conditions_df)}")
    print(f"  ✓ Timetables: {len(timetables_df)}")
    
    return nodes_df, services_df, edges_df, conditions_df, timetables_df


def create_node_features(nodes_df):
    """Create feature vectors for nodes (locations)."""
    
    print("\n🔧 Creating node features...")
    
    # One-hot encode location type
    type_encoder = LabelEncoder()
    type_encoded = type_encoder.fit_transform(nodes_df['type'])
    type_one_hot = np.eye(len(type_encoder.classes_))[type_encoded]
    
    # One-hot encode region
    region_encoder = LabelEncoder()
    region_encoded = region_encoder.fit_transform(nodes_df['region'])
    region_one_hot = np.eye(len(region_encoder.classes_))[region_encoded]
    
    # Normalize coordinates
    coords = nodes_df[['latitude', 'longitude']].values
    coords_scaler = StandardScaler()
    coords_normalized = coords_scaler.fit_transform(coords)
    
    # Combine features
    node_features = np.hstack([
        type_one_hot,
        region_one_hot,
        coords_normalized
    ])
    
    print(f"  ✓ Node feature dim: {node_features.shape[1]}")
    print(f"  ✓ Type classes: {list(type_encoder.classes_)}")
    print(f"  ✓ Region classes: {list(region_encoder.classes_)}")
    
    return (
        torch.tensor(node_features, dtype=torch.float32),
        type_encoder,
        region_encoder,
        coords_scaler
    )


def create_edge_features(edges_df, services_df, conditions_df):
    """Create feature vectors for edges (routes/services)."""
    
    print("\n🔧 Creating edge features...")
    
    edge_features_list = []
    target_labels = []
    
    # Encode time periods and day types
    time_periods = conditions_df['time_period'].unique()
    day_types = conditions_df['day_type'].unique()
    
    time_period_encoder = LabelEncoder().fit(time_periods)
    day_type_encoder = LabelEncoder().fit(day_types)
    
    # For each edge, aggregate reliability from service_conditions
    for idx, edge in edges_df.iterrows():
        service_id = edge['service_id']
        distance = edge['distance_km']
        duration = edge['avg_duration_min']
        
        # Get service type
        service = services_df[services_df['service_id'] == service_id]
        if len(service) == 0:
            continue
        
        mode = service.iloc[0]['mode']
        mode_encoder = LabelEncoder().fit(['bus', 'train', 'ridehailing'])
        mode_encoded = mode_encoder.transform([mode])[0]
        
        # Get conditions for this service (average across time periods and day types)
        service_conditions = conditions_df[conditions_df['service_id'] == service_id]
        
        if len(service_conditions) > 0:
            avg_reliability = service_conditions['reliability'].mean()
            avg_crowding = service_conditions['is_crowded'].mean()
            availability = len(service_conditions['time_period'].unique()) / len(time_periods)
        else:
            avg_reliability = 0.5
            avg_crowding = 0.5
            availability = 0.5
        
        # Create feature vector
        features = np.array([
            distance / 100,  # Normalize distance
            duration / 100,  # Normalize duration
            mode_encoded / 2,  # Normalize mode
            avg_reliability,
            avg_crowding,
            availability
        ])
        
        edge_features_list.append(features)
        target_labels.append(avg_reliability)  # Target is average reliability
    
    edge_features = np.array(edge_features_list)
    target_labels = np.array(target_labels)
    
    print(f"  ✓ Edge feature dim: {edge_features.shape[1]}")
    print(f"  ✓ Training samples: {len(edge_features)}")
    print(f"  ✓ Target reliability range: [{target_labels.min():.3f}, {target_labels.max():.3f}]")
    
    return (
        torch.tensor(edge_features, dtype=torch.float32),
        torch.tensor(target_labels, dtype=torch.float32),
        time_period_encoder,
        day_type_encoder,
        mode_encoder
    )


def create_graph(nodes_df, edges_df):
    """Create PyTorch Geometric graph structure."""
    
    print("\n🔧 Creating graph structure...")
    
    # Map location IDs to indices
    location_id_to_idx = {loc_id: idx for idx, loc_id in enumerate(nodes_df['location_id'])}
    
    # Create edge index from edges_df
    edge_list = []
    for _, edge in edges_df.iterrows():
        src = location_id_to_idx[edge['origin_id']]
        dst = location_id_to_idx[edge['destination_id']]
        edge_list.append([src, dst])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    print(f"  ✓ Number of nodes: {len(nodes_df)}")
    print(f"  ✓ Number of edges: {edge_index.shape[1]}")
    
    return edge_index


def train_model(model, node_features, edge_index, edge_features, target_labels,
                num_epochs=100, learning_rate=0.001, batch_size=32):
    """Train the GNN model."""
    
    print("\n🚀 Training model...")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    # Prepare training data
    train_indices = torch.arange(len(target_labels))
    train_dataset = TensorDataset(train_indices, target_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    train_losses = []
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for batch_indices, batch_targets in train_loader:
            # Forward pass
            predictions = model(node_features, edge_index, edge_features, batch_indices)
            
            # Compute loss
            loss = loss_fn(predictions, batch_targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    print(f"✅ Training complete! Best loss: {best_loss:.6f}")
    
    return train_losses


def save_model(model, node_features, edge_index, edge_features,
               type_encoder, region_encoder, coords_scaler,
               output_path="model/transport_gnn_model.pth"):
    """Save trained model and artifacts."""
    
    print(f"\n💾 Saving model to {output_path}...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_attr': edge_features,
        'type_encoder': type_encoder,
        'region_encoder': region_encoder,
        'scaler': coords_scaler,
    }
    
    torch.save(checkpoint, output_path)
    print(f"✅ Model saved!")


def evaluate_model(model, node_features, edge_index, edge_features, target_labels):
    """Evaluate model on training data."""
    
    print("\n📈 Evaluating model...")
    
    model.eval()
    with torch.no_grad():
        all_indices = torch.arange(len(target_labels))
        predictions = model(node_features, edge_index, edge_features, all_indices)
    
    mse = ((predictions - target_labels) ** 2).mean().item()
    mae = (torch.abs(predictions - target_labels)).mean().item()
    
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"  Target range: [{target_labels.min():.3f}, {target_labels.max():.3f}]")
    
    return mse, mae


def main():
    """Main training pipeline."""
    
    print("=" * 70)
    print("🚂 TRANSPORT SERVICE GNN MODEL TRAINING")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Load data
    nodes_df, services_df, edges_df, conditions_df, timetables_df = load_and_prepare_data("data")
    
    # Step 2: Create features
    node_features, type_encoder, region_encoder, coords_scaler = create_node_features(nodes_df)
    edge_features, target_labels, time_period_encoder, day_type_encoder, mode_encoder = \
        create_edge_features(edges_df, services_df, conditions_df)
    
    # Step 3: Create graph
    edge_index = create_graph(nodes_df, edges_df)
    
    # Step 4: Initialize model
    print("\n🧠 Initializing model...")
    model = TransportGNN(
        node_features=node_features.shape[1],
        edge_features=edge_features.shape[1],
        hidden_dim=64,
        output_dim=1
    )
    print(f"  ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 5: Train
    train_losses = train_model(
        model, node_features, edge_index, edge_features, target_labels,
        num_epochs=100,
        learning_rate=0.001,
        batch_size=16
    )
    
    # Step 6: Evaluate
    evaluate_model(model, node_features, edge_index, edge_features, target_labels)
    
    # Step 7: Save
    save_model(
        model, node_features, edge_index, edge_features,
        type_encoder, region_encoder, coords_scaler,
        output_path="model/transport_gnn_model.pth"
    )
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext steps:")
    print("  1. Start service: uvicorn app:app --reload --port 8001")
    print("  2. Test: curl -X POST http://localhost:8001/api/recommend \\")
    print("            -d '{\"origin\":\"Colombo\",\"destination\":\"Anuradhapura\"}'")


if __name__ == "__main__":
    main()
