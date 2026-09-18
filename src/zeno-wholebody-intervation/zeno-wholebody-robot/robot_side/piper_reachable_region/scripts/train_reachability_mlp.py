#!/usr/bin/env python3
"""
MLP-based Implicit Reachability Map Training Script
====================================================

This script trains an MLP-based implicit function that maps 3D positions in the robot's base frame
to reachability probability and manipulability scores.

Input:
  - HDF5 file with reachability data (produced by reachability analysis)
  - Expected structure: /Spheres/sphere_dataset (N x 4 array with [x, y, z, score])

Output:
  - Trained PyTorch model (model.pth)
  - Configuration file with normalization parameters (config.json)

Usage:
  python train_reachability_mlp.py \
    --hdf5_path /path/to/reachability.hdf5 \
    --output_dir ./trained_model \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 128 \
    --num_layers 5

Author: Teleop Team
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ReachabilityDataset(Dataset):
    """
    PyTorch Dataset for reachability prediction.
    
    Returns:
      - xyz_normalized: 3D position in normalized space [-1, 1]^3
      - y_reach: binary reachability label (0 or 1)
      - y_manip: manipulability score (for reachable points only)
    """
    
    def __init__(self, xyz_normalized, y_reach, y_manip):
        """
        Args:
            xyz_normalized: (N, 3) array of normalized 3D positions
            y_reach: (N,) array of binary reachability labels
            y_manip: (N,) array of manipulability scores
        """
        self.xyz_normalized = torch.from_numpy(xyz_normalized).float()
        self.y_reach = torch.from_numpy(y_reach).float()
        self.y_manip = torch.from_numpy(y_manip).float()
        
    def __len__(self):
        return len(self.xyz_normalized)
    
    def __getitem__(self, idx):
        return self.xyz_normalized[idx], self.y_reach[idx], self.y_manip[idx]


class ReachabilityMLP(nn.Module):
    """
    MLP-based implicit reachability function with dual heads.
    
    Architecture:
      - Shared backbone: several fully-connected layers with ReLU activation
      - Head 1 (Reachability): outputs sigmoid-normalized probability [0, 1]
      - Head 2 (Manipulability): outputs non-negative manipulability score
    """
    
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=4, output_manip=True):
        """
        Args:
            input_dim: dimensionality of input (typically 3 for xyz)
            hidden_dim: hidden layer size
            num_layers: number of layers in the backbone
            output_manip: whether to output manipulability head
        """
        super(ReachabilityMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_manip = output_manip
        
        # Build shared backbone
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.backbone = nn.Sequential(*layers)
        
        # Reachability head: outputs probability [0, 1]
        self.reach_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Manipulability head: outputs non-negative score
        if self.output_manip:
            self.manip_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus()  # Softplus ensures output >= 0
            )
    
    def forward(self, xyz):
        """
        Args:
            xyz: (B, 3) tensor of normalized 3D positions
            
        Returns:
            p_reach: (B, 1) tensor of reachability probabilities
            m_pred: (B, 1) tensor of manipulability scores (or None if output_manip=False)
        """
        x = self.backbone(xyz)
        p_reach = self.reach_head(x)
        m_pred = self.manip_head(x) if self.output_manip else None
        return p_reach, m_pred


def load_hdf5_reachability_data(hdf5_path):
    """
    Load reachability data from HDF5 file.
    
    Expected structure:
      /Spheres/sphere_dataset: (N, 4) array [x, y, z, score]
    
    Returns:
        spheres: (N, 4) array of [x, y, z, score]
    """
    with h5py.File(hdf5_path, 'r') as f:
        if '/Spheres/sphere_dataset' in f:
            spheres = f['/Spheres/sphere_dataset'][()]
        else:
            # Try alternative paths
            print(f"Available keys in HDF5: {list(f.keys())}")
            raise KeyError("Could not find /Spheres/sphere_dataset in HDF5 file")
    
    print(f"Loaded {len(spheres)} reachable positions from HDF5")
    return spheres


def construct_training_data(spheres, num_negative_samples=None, negative_ratio=1.0, 
                             boundary_sampling=False, boundary_radius=0.2):
    """
    Construct positive and negative training samples from reachability data.
    
    Args:
        spheres: (N, 4) array of [x, y, z, score]
        num_negative_samples: if None, set to len(spheres) * negative_ratio
        negative_ratio: ratio of negative samples to positive samples
        boundary_sampling: whether to focus sampling near the reachability boundary
        boundary_radius: radius for boundary-focused sampling
        
    Returns:
        xyz_data: (M, 3) array of 3D positions
        y_reach: (M,) binary reachability labels
        y_manip: (M,) manipulability scores
        stats: dict with statistics
    """
    xyz_pos = spheres[:, :3]
    manip_pos = spheres[:, 3]
    
    # Normalize manipulability (optional: use log transform for better conditioning)
    manip_pos_norm = manip_pos / (np.max(manip_pos) + 1e-8)
    
    # Compute bounding box
    xyz_min = np.min(xyz_pos, axis=0)
    xyz_max = np.max(xyz_pos, axis=0)
    print(f"Bounding box: min={xyz_min}, max={xyz_max}")
    
    # Generate negative samples
    if num_negative_samples is None:
        num_negative_samples = int(len(spheres) * negative_ratio)
    
    # Simple approach: sample uniformly in the bounding box
    xyz_neg = np.random.uniform(xyz_min, xyz_max, size=(num_negative_samples, 3))
    
    # Check if negative samples are truly negative (far from reachable set)
    if boundary_sampling:
        # Keep only those far enough from any reachable point
        from scipy.spatial import cKDTree
        tree = cKDTree(xyz_pos)
        distances, _ = tree.query(xyz_neg)
        valid_neg = distances > boundary_radius
        xyz_neg = xyz_neg[valid_neg]
        print(f"After boundary filtering: {len(xyz_neg)} negative samples")
    
    # Combine positive and negative
    xyz_data = np.vstack([xyz_pos, xyz_neg])
    y_reach = np.hstack([np.ones(len(xyz_pos)), np.zeros(len(xyz_neg))])
    y_manip = np.hstack([manip_pos_norm, np.zeros(len(xyz_neg))])
    
    # Shuffle
    perm = np.random.permutation(len(xyz_data))
    xyz_data = xyz_data[perm]
    y_reach = y_reach[perm]
    y_manip = y_manip[perm]
    
    stats = {
        'num_positive': len(xyz_pos),
        'num_negative': len(xyz_neg),
        'xyz_min': xyz_min.tolist(),
        'xyz_max': xyz_max.tolist(),
        'manip_max': float(np.max(manip_pos)),
        'manip_min': float(np.min(manip_pos)),
    }
    
    print(f"Dataset: {stats['num_positive']} positive, {stats['num_negative']} negative samples")
    
    return xyz_data, y_reach, y_manip, stats


def normalize_xyz(xyz, xyz_min, xyz_max):
    """Normalize xyz to [-1, 1]^3 using the bounding box."""
    xyz_center = (xyz_min + xyz_max) / 2
    xyz_half_size = (xyz_max - xyz_min) / 2
    return (xyz - xyz_center) / (xyz_half_size + 1e-8)


def denormalize_xyz(xyz_norm, xyz_min, xyz_max):
    """Denormalize from [-1, 1]^3 back to original space."""
    xyz_center = (xyz_min + xyz_max) / 2
    xyz_half_size = (xyz_max - xyz_min) / 2
    return xyz_norm * (xyz_half_size + 1e-8) + xyz_center


def train_epoch(model, dataloader, optimizer, device, lambda_reach=1.0, lambda_manip=0.1):
    """
    Train for one epoch.
    
    Args:
        model: ReachabilityMLP instance
        dataloader: PyTorch DataLoader
        optimizer: PyTorch optimizer
        device: torch.device (cpu or cuda)
        lambda_reach: weight for reachability loss
        lambda_manip: weight for manipulability loss (applied only to positive samples)
        
    Returns:
        losses: dict with loss breakdown
    """
    model.train()
    total_loss = 0.0
    total_reach_loss = 0.0
    total_manip_loss = 0.0
    
    reach_criterion = nn.BCELoss()
    manip_criterion = nn.MSELoss()
    
    for xyz, y_reach, y_manip in dataloader:
        xyz = xyz.to(device)
        y_reach = y_reach.to(device).view(-1, 1)
        y_manip = y_manip.to(device).view(-1, 1)
        
        optimizer.zero_grad()
        
        # Forward pass
        p_reach, m_pred = model(xyz)
        
        # Reachability loss
        reach_loss = reach_criterion(p_reach, y_reach)
        
        # Manipulability loss: only on positive samples
        pos_mask = (y_reach > 0.5).squeeze()
        if pos_mask.sum() > 0:
            manip_loss = manip_criterion(m_pred[pos_mask], y_manip[pos_mask])
        else:
            manip_loss = torch.tensor(0.0, device=device)
        
        # Total loss
        loss = lambda_reach * reach_loss + lambda_manip * manip_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_reach_loss += reach_loss.item()
        total_manip_loss += manip_loss.item() if isinstance(manip_loss, torch.Tensor) else 0.0
    
    num_batches = len(dataloader)
    return {
        'total_loss': total_loss / num_batches,
        'reach_loss': total_reach_loss / num_batches,
        'manip_loss': total_manip_loss / num_batches,
    }


def evaluate(model, dataloader, device, lambda_reach=1.0, lambda_manip=0.1):
    """
    Evaluate model on a validation/test set.
    
    Returns:
        losses: dict with loss breakdown
        metrics: dict with additional metrics (accuracy, etc.)
    """
    model.eval()
    total_loss = 0.0
    total_reach_loss = 0.0
    total_manip_loss = 0.0
    
    correct = 0
    total = 0
    
    reach_criterion = nn.BCELoss()
    manip_criterion = nn.L1Loss()
    
    with torch.no_grad():
        for xyz, y_reach, y_manip in dataloader:
            xyz = xyz.to(device)
            y_reach = y_reach.to(device).view(-1, 1)
            y_manip = y_manip.to(device).view(-1, 1)
            
            p_reach, m_pred = model(xyz)
            
            reach_loss = reach_criterion(p_reach, y_reach)
            pos_mask = (y_reach > 0.5).squeeze()
            if pos_mask.sum() > 0:
                manip_loss = manip_criterion(m_pred[pos_mask], y_manip[pos_mask])
            else:
                manip_loss = torch.tensor(0.0, device=device)
            
            loss = lambda_reach * reach_loss + lambda_manip * manip_loss
            
            total_loss += loss.item()
            total_reach_loss += reach_loss.item()
            total_manip_loss += manip_loss.item() if isinstance(manip_loss, torch.Tensor) else 0.0
            
            # Accuracy
            pred_reach = (p_reach > 0.5).float()
            correct += (pred_reach == y_reach).sum().item()
            total += len(y_reach)
    
    num_batches = len(dataloader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return {
        'losses': {
            'total_loss': total_loss / num_batches,
            'reach_loss': total_reach_loss / num_batches,
            'manip_loss': total_manip_loss / num_batches,
        },
        'metrics': {
            'accuracy': accuracy,
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Train MLP-based implicit reachability map')
    # parser.add_argument('--hdf5_path', type=str, required=True, help='Path to HDF5 reachability data')
    parser.add_argument('--output_dir', type=str, default='./trained_model', help='Output directory for model and config')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension of MLP')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of layers in MLP backbone')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for Adam optimizer')
    parser.add_argument('--lambda_reach', type=float, default=1.0, help='Weight for reachability loss')
    parser.add_argument('--lambda_manip', type=float, default=0.1, help='Weight for manipulability loss')
    parser.add_argument('--negative_ratio', type=float, default=1.0, help='Ratio of negative to positive samples')
    parser.add_argument('--validation_split', type=float, default=0.1, help='Fraction of data for validation')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    args.hdf5_path = "/home/jeong/zeno/wholebody-teleop/teleop/src/zeno-wholebody-teleop/robot_side/piper_reachable_region/maps/3D_reach_map_gripper_base_0.05_2025-12-11-17-29-20.h5"
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading reachability data from {args.hdf5_path}...")
    spheres = load_hdf5_reachability_data(args.hdf5_path)
    
    # Construct training data
    print("\nConstructing training data...")
    xyz_data, y_reach, y_manip, stats = construct_training_data(
        spheres, 
        num_negative_samples=None,
        negative_ratio=args.negative_ratio,
        boundary_sampling=True,
        boundary_radius=0.05,
    )
    
    # Normalize xyz
    xyz_min = np.array(stats['xyz_min'])
    xyz_max = np.array(stats['xyz_max'])
    xyz_normalized = normalize_xyz(xyz_data, xyz_min, xyz_max)
    
    # Split into train/val
    num_samples = len(xyz_normalized)
    num_val = int(num_samples * args.validation_split)
    num_train = num_samples - num_val
    
    train_dataset = ReachabilityDataset(xyz_normalized[:num_train], y_reach[:num_train], y_manip[:num_train])
    val_dataset = ReachabilityDataset(xyz_normalized[num_train:], y_reach[num_train:], y_manip[num_train:])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train samples: {num_train}, Val samples: {num_val}")
    
    # Create model
    print(f"\nCreating model: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}")
    model = ReachabilityMLP(
        input_dim=3,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_manip=True,
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_losses = train_epoch(model, train_loader, optimizer, device, 
                                   lambda_reach=args.lambda_reach, lambda_manip=args.lambda_manip)
        
        val_result = evaluate(model, val_loader, device, 
                             lambda_reach=args.lambda_reach, lambda_manip=args.lambda_manip)
        val_losses = val_result['losses']
        val_metrics = val_result['metrics']
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"  Train: loss={train_losses['total_loss']:.6f}, reach={train_losses['reach_loss']:.6f}, manip={train_losses['manip_loss']:.6f}")
            print(f"  Val:   loss={val_losses['total_loss']:.6f}, reach={val_losses['reach_loss']:.6f}, manip={val_losses['manip_loss']:.6f}, acc={val_metrics['accuracy']:.2f}%")
        
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            best_epoch = epoch
    
    print(f"\nBest validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
    
    # Save model
    model_path = os.path.join(args.output_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save config
    config = {
        'model_architecture': {
            'input_dim': 3,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'output_manip': True,
        },
        'normalization': {
            'xyz_min': stats['xyz_min'],
            'xyz_max': stats['xyz_max'],
            'manip_max': stats['manip_max'],
            'manip_min': stats['manip_min'],
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'lambda_reach': args.lambda_reach,
            'lambda_manip': args.lambda_manip,
            'negative_ratio': args.negative_ratio,
            'validation_split': args.validation_split,
        },
        'data_stats': stats,
    }
    
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()

