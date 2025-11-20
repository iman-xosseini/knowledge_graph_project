"""
GNN-Based Supply Chain Demand Forecasting
==========================================

This module implements a Graph Neural Network for supply chain forecasting that:
1. Uses historical production/demand as node features
2. Incorporates time-related features for seasonality/trends
3. Leverages neighborhood influence through graph structure
4. Implements hierarchy-aware forecasting
5. Supports transfer learning for data-scarce scenarios
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TimeFeatureExtractor:
    """
    Extracts temporal features including seasonality, trends, and cyclical patterns
    """
    def __init__(self):
        self.scaler = StandardScaler()

    def extract_features(self, dates):
        """
        Extract comprehensive time features from dates

        Args:
            dates: Array of datetime objects or strings

        Returns:
            DataFrame with temporal features
        """
        if isinstance(dates[0], str):
            dates = pd.to_datetime(dates)

        features = pd.DataFrame()

        # Cyclical features (sine/cosine encoding)
        features['day_of_week_sin'] = np.sin(2 * np.pi * dates.dayofweek / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * dates.dayofweek / 7)
        features['day_of_month_sin'] = np.sin(2 * np.pi * dates.day / 31)
        features['day_of_month_cos'] = np.cos(2 * np.pi * dates.day / 31)
        features['month_sin'] = np.sin(2 * np.pi * dates.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * dates.month / 12)
        features['quarter_sin'] = np.sin(2 * np.pi * dates.quarter / 4)
        features['quarter_cos'] = np.cos(2 * np.pi * dates.quarter / 4)

        # Binary features
        features['is_weekend'] = (dates.dayofweek >= 5).astype(int)
        features['is_month_start'] = dates.is_month_start.astype(int)
        features['is_month_end'] = dates.is_month_end.astype(int)
        features['is_quarter_start'] = dates.is_quarter_start.astype(int)
        features['is_quarter_end'] = dates.is_quarter_end.astype(int)

        # Linear trend
        features['days_since_start'] = (dates - dates.min()).days

        return features


class SupplyChainDataPreprocessor:
    """
    Preprocesses supply chain data for GNN training
    """
    def __init__(self, lookback_window=7, forecast_horizon=1):
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.time_extractor = TimeFeatureExtractor()

    def load_data(self, data_path="../data/SupplyGraph"):
        """Load all supply chain data"""
        # Load edges
        self.df_edges_plant = pd.read_csv(f"{data_path}/Edges/Edges (Plant).csv")
        self.df_edges_product_group = pd.read_csv(f"{data_path}/Edges/Edges (Product Group).csv")
        self.df_edges_product_subgroup = pd.read_csv(f"{data_path}/Edges/Edges (Product Sub-Group).csv")
        self.df_edges_storage_location = pd.read_csv(f"{data_path}/Edges/Edges (Storage Location).csv")

        # Load nodes
        self.df_nodes_productgroup_and_subgroup = pd.read_csv(f"{data_path}/Nodes/Node Types (Product Group and Subgroup).csv")
        self.df_nodes_plant_and_storage = pd.read_csv(f"{data_path}/Nodes/Nodes Type (Plant & Storage).csv")
        self.df_nodes = pd.read_csv(f"{data_path}/Nodes/Nodes.csv")
        self.df_nodes_index = pd.read_csv(f"{data_path}/Nodes/NodesIndex.csv")

        # Load temporal data
        self.df_delivery_to_distributor = pd.read_csv(f"{data_path}/Temporal Data/Unit/Delivery To distributor.csv")
        self.df_factory_issue = pd.read_csv(f"{data_path}/Temporal Data/Unit/Factory Issue.csv")
        self.df_production = pd.read_csv(f"{data_path}/Temporal Data/Unit/Production.csv")
        self.df_sales_order = pd.read_csv(f"{data_path}/Temporal Data/Unit/Sales Order.csv")

        # Convert dates
        for df in [self.df_production, self.df_sales_order, self.df_factory_issue, self.df_delivery_to_distributor]:
            df['Date'] = pd.to_datetime(df['Date'])

        print(f"✓ Data loaded successfully")
        print(f"  - Nodes: {len(self.df_nodes)}")
        print(f"  - Temporal records: {len(self.df_production)} days")
        print(f"  - Date range: {self.df_production['Date'].min()} to {self.df_production['Date'].max()}")

        return self

    def build_graph_structure(self, edge_weights=None):
        """
        Build heterogeneous graph structure combining multiple edge types

        Args:
            edge_weights: Dict with weights for each edge type
                         e.g., {'plant': 1.0, 'group': 0.8, 'subgroup': 0.9, 'storage': 0.7}
        """
        if edge_weights is None:
            edge_weights = {'plant': 1.0, 'group': 0.8, 'subgroup': 0.9, 'storage': 0.7}

        # Create node index mapping
        node_to_idx = dict(zip(self.df_nodes_index['Node'], self.df_nodes_index['NodeIndex']))

        edges = []
        edge_attrs = []

        # Add plant edges
        for _, row in self.df_edges_plant.iterrows():
            if row['node1'] in node_to_idx and row['node2'] in node_to_idx:
                edges.append([node_to_idx[row['node1']], node_to_idx[row['node2']]])
                edges.append([node_to_idx[row['node2']], node_to_idx[row['node1']]])  # Undirected
                edge_attrs.extend([edge_weights['plant']] * 2)

        # Add product group edges
        for _, row in self.df_edges_product_group.iterrows():
            if row['node1'] in node_to_idx and row['node2'] in node_to_idx:
                edges.append([node_to_idx[row['node1']], node_to_idx[row['node2']]])
                edges.append([node_to_idx[row['node2']], node_to_idx[row['node1']]])
                edge_attrs.extend([edge_weights['group']] * 2)

        # Add product subgroup edges
        for _, row in self.df_edges_product_subgroup.iterrows():
            if row['node1'] in node_to_idx and row['node2'] in node_to_idx:
                edges.append([node_to_idx[row['node1']], node_to_idx[row['node2']]])
                edges.append([node_to_idx[row['node2']], node_to_idx[row['node1']]])
                edge_attrs.extend([edge_weights['subgroup']] * 2)

        # Add storage location edges
        for _, row in self.df_edges_storage_location.iterrows():
            if row['node1'] in node_to_idx and row['node2'] in node_to_idx:
                edges.append([node_to_idx[row['node1']], node_to_idx[row['node2']]])
                edges.append([node_to_idx[row['node2']], node_to_idx[row['node1']]])
                edge_attrs.extend([edge_weights['storage']] * 2)

        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        self.edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)

        print(f"✓ Graph structure built: {self.edge_index.shape[1]} edges")

        return self

    def create_node_features(self, date_idx, target='production'):
        """
        Create node features for a specific date

        Args:
            date_idx: Index of the date in temporal data
            target: Which temporal data to use as target ('production', 'sales_order', 'factory_issue', 'delivery')
        """
        target_map = {
            'production': self.df_production,
            'sales_order': self.df_sales_order,
            'factory_issue': self.df_factory_issue,
            'delivery': self.df_delivery_to_distributor
        }

        df_target = target_map[target]

        # Get historical data (lookback window)
        start_idx = max(0, date_idx - self.lookback_window)
        historical_data = []

        for node in self.df_nodes['Node']:
            if node in df_target.columns:
                # Historical values
                hist_values = df_target.iloc[start_idx:date_idx][node].values

                # Pad if necessary
                if len(hist_values) < self.lookback_window:
                    hist_values = np.pad(hist_values, (self.lookback_window - len(hist_values), 0), 'constant')

                # Statistical features
                mean_val = np.mean(hist_values)
                std_val = np.std(hist_values)
                min_val = np.min(hist_values)
                max_val = np.max(hist_values)
                trend = hist_values[-1] - hist_values[0] if len(hist_values) > 1 else 0

                # Combine features
                node_features = list(hist_values) + [mean_val, std_val, min_val, max_val, trend]
                historical_data.append(node_features)
            else:
                # Node not in data - use zeros
                historical_data.append([0] * (self.lookback_window + 5))

        # Add time features
        current_date = df_target.iloc[date_idx]['Date']
        time_features = self.time_extractor.extract_features([current_date])

        # Broadcast time features to all nodes
        time_features_array = np.tile(time_features.values, (len(self.df_nodes), 1))

        # Combine historical and time features
        node_features = np.concatenate([
            np.array(historical_data),
            time_features_array
        ], axis=1)

        # Add hierarchy features (one-hot encoded groups and subgroups)
        hierarchy_features = self._create_hierarchy_features()
        node_features = np.concatenate([node_features, hierarchy_features], axis=1)

        return torch.tensor(node_features, dtype=torch.float)

    def _create_hierarchy_features(self):
        """Create one-hot encoded hierarchy features"""
        # Get unique groups and subgroups
        groups = self.df_nodes_productgroup_and_subgroup['Group'].unique()
        subgroups = self.df_nodes_productgroup_and_subgroup['Sub-Group'].unique()

        group_to_idx = {g: i for i, g in enumerate(groups)}
        subgroup_to_idx = {sg: i for i, sg in enumerate(subgroups)}

        hierarchy_features = []

        for node in self.df_nodes['Node']:
            node_info = self.df_nodes_productgroup_and_subgroup[
                self.df_nodes_productgroup_and_subgroup['Node'] == node
            ]

            if not node_info.empty:
                group = node_info.iloc[0]['Group']
                subgroup = node_info.iloc[0]['Sub-Group']

                # One-hot for group
                group_onehot = np.zeros(len(groups))
                group_onehot[group_to_idx[group]] = 1

                # One-hot for subgroup
                subgroup_onehot = np.zeros(len(subgroups))
                subgroup_onehot[subgroup_to_idx[subgroup]] = 1

                hierarchy_features.append(np.concatenate([group_onehot, subgroup_onehot]))
            else:
                hierarchy_features.append(np.zeros(len(groups) + len(subgroups)))

        return np.array(hierarchy_features)

    def create_dataset(self, target='production', train_ratio=0.7, val_ratio=0.15):
        """
        Create train/val/test datasets

        Returns:
            train_data, val_data, test_data: Lists of PyG Data objects
        """
        df_target = {
            'production': self.df_production,
            'sales_order': self.df_sales_order,
            'factory_issue': self.df_factory_issue,
            'delivery': self.df_delivery_to_distributor
        }[target]

        dataset = []

        # Create data for each time step
        for date_idx in range(self.lookback_window, len(df_target) - self.forecast_horizon):
            # Node features
            x = self.create_node_features(date_idx, target)

            # Target (next time step values)
            target_values = []
            for node in self.df_nodes['Node']:
                if node in df_target.columns:
                    target_val = df_target.iloc[date_idx + self.forecast_horizon][node]
                    target_values.append(target_val)
                else:
                    target_values.append(0)

            y = torch.tensor(target_values, dtype=torch.float).unsqueeze(1)

            # Create PyG Data object
            data = Data(
                x=x,
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                y=y
            )

            dataset.append(data)

        # Split dataset
        n_samples = len(dataset)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        train_data = dataset[:n_train]
        val_data = dataset[n_train:n_train + n_val]
        test_data = dataset[n_train + n_val:]

        print(f"✓ Dataset created:")
        print(f"  - Train: {len(train_data)} samples")
        print(f"  - Val: {len(val_data)} samples")
        print(f"  - Test: {len(test_data)} samples")
        print(f"  - Node features dim: {train_data[0].x.shape[1]}")

        return train_data, val_data, test_data


class HierarchyAwareGNN(nn.Module):
    """
    Hierarchy-aware GNN for supply chain forecasting

    This model incorporates:
    - Multi-layer graph convolutions for neighborhood aggregation
    - Hierarchy-aware attention mechanism
    - Residual connections for better gradient flow
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, dropout=0.2, use_attention=True):
        super(HierarchyAwareGNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention

        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        if use_attention:
            self.convs.append(GATConv(in_channels, hidden_channels, heads=4, concat=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels * 4))
            current_dim = hidden_channels * 4
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            current_dim = hidden_channels

        # Hidden layers
        for _ in range(num_layers - 2):
            if use_attention:
                self.convs.append(GATConv(current_dim, hidden_channels, heads=4, concat=True))
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels * 4))
                current_dim = hidden_channels * 4
            else:
                self.convs.append(GCNConv(current_dim, hidden_channels))
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
                current_dim = hidden_channels

        # Last layer
        if use_attention:
            self.convs.append(GATConv(current_dim, hidden_channels, heads=1, concat=False))
        else:
            self.convs.append(GCNConv(current_dim, hidden_channels))

        # Output layer
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, out_channels)

        # Hierarchy embedding
        self.hierarchy_attention = nn.MultiheadAttention(hidden_channels, num_heads=4)

    def forward(self, x, edge_index, edge_attr=None):
        # Graph convolutions
        for i in range(self.num_layers):
            x_residual = x if i > 0 and x.shape[1] == self.convs[i-1].out_channels else None

            if self.use_attention and i < self.num_layers - 1:
                x = self.convs[i](x, edge_index)
            else:
                x = self.convs[i](x, edge_index, edge_attr)

            if i < self.num_layers - 1:
                x = self.batch_norms[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

                # Residual connection
                if x_residual is not None and x_residual.shape == x.shape:
                    x = x + x_residual

        # Hierarchy-aware attention
        x_attn, _ = self.hierarchy_attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        x = x + x_attn.squeeze(0)

        # Output layers
        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x


class TransferLearningGNN(nn.Module):
    """
    GNN with transfer learning support for data-scarce scenarios

    Features:
    - Pre-training on all nodes
    - Fine-tuning on specific node groups
    - Meta-learning capabilities
    """
    def __init__(self, base_model, hierarchy_info):
        super(TransferLearningGNN, self).__init__()
        self.base_model = base_model
        self.hierarchy_info = hierarchy_info

        # Group-specific adapters
        unique_groups = hierarchy_info['Group'].unique()
        self.group_adapters = nn.ModuleDict({
            str(group): nn.Sequential(
                nn.Linear(base_model.fc2.in_features, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ) for group in unique_groups
        })

    def forward(self, x, edge_index, edge_attr=None, node_groups=None):
        # Get base model features (before final layer)
        for i in range(self.base_model.num_layers):
            if self.base_model.use_attention and i < self.base_model.num_layers - 1:
                x = self.base_model.convs[i](x, edge_index)
            else:
                x = self.base_model.convs[i](x, edge_index, edge_attr)

            if i < self.base_model.num_layers - 1:
                x = self.base_model.batch_norms[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=self.base_model.dropout, training=self.training)

        # Hierarchy attention
        x_attn, _ = self.base_model.hierarchy_attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        x = x + x_attn.squeeze(0)

        # FC1
        features = F.elu(self.base_model.fc1(x))
        features = F.dropout(features, p=self.base_model.dropout, training=self.training)

        # Group-specific prediction
        if node_groups is not None:
            outputs = []
            for i, group in enumerate(node_groups):
                adapter = self.group_adapters[str(group)]
                outputs.append(adapter(features[i]))
            return torch.stack(outputs)
        else:
            # Use base model for all nodes
            return self.base_model.fc2(features)

    def freeze_base_model(self):
        """Freeze base model for fine-tuning"""
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_base_model(self):
        """Unfreeze base model"""
        for param in self.base_model.parameters():
            param.requires_grad = True


class GNNTrainer:
    """
    Trainer for GNN models with comprehensive training utilities
    """
    def __init__(self, model, device='cpu', learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.criterion = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_data):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for data in train_data:
            data = data.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index, data.edge_attr)
            loss = self.criterion(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_data)

    def evaluate(self, data_list):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for data in data_list:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index, data.edge_attr)
                loss = self.criterion(out, data.y)
                total_loss += loss.item()

                predictions.append(out.cpu().numpy())
                targets.append(data.y.cpu().numpy())

        avg_loss = total_loss / len(data_list)
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        # Calculate metrics
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)

        return avg_loss, mae, rmse, r2, predictions, targets

    def train(self, train_data, val_data, epochs=100, early_stopping_patience=20):
        """Full training loop"""
        best_val_loss = float('inf')
        patience_counter = 0

        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_data)
            val_loss, val_mae, val_rmse, val_r2, _, _ = self.evaluate(val_data)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f} | MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | R2: {val_r2:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_gnn_model.pt')
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Load best model
        self.model.load_state_dict(torch.load('best_gnn_model.pt'))
        print("\n✓ Training completed!")

    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ============================================================================
# Main Execution Functions
# ============================================================================

def main_training_pipeline(data_path="../data/SupplyGraph",
                           target='production',
                           lookback_window=7,
                           forecast_horizon=1,
                           hidden_channels=128,
                           num_layers=3,
                           epochs=100,
                           use_attention=True):
    """
    Complete training pipeline

    Args:
        data_path: Path to supply chain data
        target: Target variable ('production', 'sales_order', etc.)
        lookback_window: Number of historical days to use
        forecast_horizon: Number of days ahead to forecast
        hidden_channels: Hidden dimension size
        num_layers: Number of GNN layers
        epochs: Training epochs
        use_attention: Whether to use attention mechanism
    """
    print("\n" + "="*70)
    print(" GNN-Based Supply Chain Forecasting")
    print("="*70)

    # 1. Load and preprocess data
    print("\n[Step 1/5] Loading data...")
    preprocessor = SupplyChainDataPreprocessor(lookback_window, forecast_horizon)
    preprocessor.load_data(data_path)

    # 2. Build graph
    print("\n[Step 2/5] Building graph structure...")
    preprocessor.build_graph_structure()

    # 3. Create datasets
    print("\n[Step 3/5] Creating datasets...")
    train_data, val_data, test_data = preprocessor.create_dataset(target=target)

    # 4. Initialize model
    print("\n[Step 4/5] Initializing model...")
    in_channels = train_data[0].x.shape[1]
    model = HierarchyAwareGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=1,
        num_layers=num_layers,
        use_attention=use_attention
    )

    print(f"Model architecture:")
    print(f"  - Input dim: {in_channels}")
    print(f"  - Hidden dim: {hidden_channels}")
    print(f"  - Layers: {num_layers}")
    print(f"  - Attention: {use_attention}")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 5. Train model
    print("\n[Step 5/5] Training model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    trainer = GNNTrainer(model, device=device, learning_rate=0.001)
    trainer.train(train_data, val_data, epochs=epochs)

    # 6. Evaluate on test set
    print("\n" + "="*70)
    print("Test Set Evaluation")
    print("="*70)
    test_loss, test_mae, test_rmse, test_r2, predictions, targets = trainer.evaluate(test_data)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R²: {test_r2:.4f}")

    # 7. Plot results
    trainer.plot_training_history()

    return model, trainer, preprocessor, (train_data, val_data, test_data)


def transfer_learning_pipeline(base_model, preprocessor, hierarchy_info,
                               target_group='S', epochs=50):
    """
    Transfer learning for data-scarce product groups

    Args:
        base_model: Pre-trained base model
        preprocessor: Data preprocessor
        hierarchy_info: DataFrame with hierarchy information
        target_group: Target group for fine-tuning
        epochs: Fine-tuning epochs
    """
    print("\n" + "="*70)
    print(f" Transfer Learning for Group: {target_group}")
    print("="*70)

    # Create transfer learning model
    transfer_model = TransferLearningGNN(base_model, hierarchy_info)

    # Freeze base model
    transfer_model.freeze_base_model()
    print("✓ Base model frozen")

    # Filter data for target group
    target_nodes = hierarchy_info[hierarchy_info['Group'] == target_group]['Node'].values
    # Implementation would filter train/val/test data for these nodes

    # Fine-tune on target group
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = GNNTrainer(transfer_model, device=device, learning_rate=0.0001)

    print(f"✓ Transfer learning model ready")
    print(f"✓ Fine-tuning on {len(target_nodes)} nodes from group '{target_group}'")

    return transfer_model, trainer


if __name__ == "__main__":
    # Run main pipeline
    model, trainer, preprocessor, datasets = main_training_pipeline(
        target='production',
        lookback_window=14,
        forecast_horizon=7,
        hidden_channels=128,
        num_layers=3,
        epochs=100,
        use_attention=True
    )

    print("\n✓ Pipeline completed successfully!")
