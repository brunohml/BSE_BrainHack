import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import MSELoss
import torch.nn.utils as utils
from pathlib import Path
from glob import glob
import random
from Transformer import Transformer, ModelArgs
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt

def collate_fn(batch):
    """Custom collate function to handle sequences and labels."""
    sequences = []
    labels = []
    
    for item in batch:
        if isinstance(item, tuple):
            seq, label = item
            sequences.append(seq)
            labels.append(label)
        else:
            sequences.append(item)
    
    sequences = torch.stack(sequences)
    if labels:
        labels = torch.stack(labels)
        return sequences, labels
    return sequences

class BrainStateDataset(Dataset):
    def __init__(self, embeddings_files, sequence_length=10):
        self.sequence_length = sequence_length
        self.data = []
        self.seizure_labels = []
        self.embedding_dim = None  # Will be set from the data
        
        for file_path in embeddings_files:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            embeddings = data['patient_embeddings']  # Shape: (n_files, n_timepoints, n_features)
            
            # Set embedding dimension if not set
            if self.embedding_dim is None:
                self.embedding_dim = embeddings.shape[-1]
            elif self.embedding_dim != embeddings.shape[-1]:
                raise ValueError(f"Inconsistent embedding dimensions across files: {self.embedding_dim} vs {embeddings.shape[-1]}")
                
            seizure_labels = data.get('seizure_labels', None)
            
            # Flatten the first two dimensions (files and timepoints)
            n_files, n_timepoints, n_features = embeddings.shape
            embeddings_flat = embeddings.reshape(-1, n_features)
            
            # Get sequences of consecutive windows
            total_windows = len(embeddings_flat)
            for i in range(0, total_windows - sequence_length + 1):
                sequence = embeddings_flat[i:i + sequence_length]
                self.data.append(sequence)
                
                if seizure_labels is not None:
                    # Flatten seizure labels if they exist
                    seizure_labels_flat = np.array(seizure_labels).reshape(-1)
                    label_sequence = seizure_labels_flat[i:i + sequence_length]
                    # Convert to binary: 1 if any window in sequence has seizure
                    has_seizure = int(np.any(label_sequence == 1))
                    self.seizure_labels.append(has_seizure)
        
        self.data = [torch.FloatTensor(seq) for seq in self.data]
        if self.seizure_labels:
            self.seizure_labels = torch.LongTensor(self.seizure_labels)
        else:
            self.seizure_labels = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        if self.seizure_labels is not None:
            label = self.seizure_labels[idx]
            return sequence, label
        return sequence

def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def split_patients(data_dir, train_ratio=0.7, val_ratio=0.15):
    """Split patients into train/val/test sets."""
    # Get all unique patient directories under jackal
    patient_dirs = glob(os.path.join(data_dir, 'jackal', 'Epat*'))
    patient_ids = [os.path.basename(d) for d in patient_dirs]
    
    if not patient_ids:
        raise ValueError("No patient directories found in jackal subdirectory")
    
    # Shuffle patients
    random.shuffle(patient_ids)
    
    # Calculate split indices
    n_patients = len(patient_ids)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    # Split patient IDs
    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train + n_val]
    test_ids = patient_ids[n_train + n_val:]
    
    return train_ids, val_ids, test_ids

def get_embeddings_files(data_dir, patient_ids):
    """Get all embeddings files for given patient IDs."""
    files = []
    for pid in patient_ids:
        pattern = os.path.join(data_dir, 'jackal', pid, 'embeddings_*.pkl')
        files.extend(glob(pattern))
    return files

def check_nan_loss(loss, epoch, batch_idx):
    """Check if loss is NaN and log relevant information."""
    if torch.isnan(loss):
        logging.error(f"NaN loss detected at epoch {epoch}, batch {batch_idx}")
        return True
    return False

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        if isinstance(batch, tuple):
            sequences, labels = batch
        else:
            sequences = batch
            
        # Move data to device
        sequences = sequences.to(device)
        batch_size = sequences.size(0)
        
        # Forward pass
        output = model(sequences)
        
        # Calculate loss (predicting next timestep)
        loss = criterion(output[:, :-1, :], sequences[:, 1:, :])
        
        # Check for NaN loss
        if check_nan_loss(loss, epoch, batch_idx):
            return float('nan')
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        
        if batch_idx % 100 == 0:
            logging.info(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if isinstance(batch, tuple):
                sequences, labels = batch
            else:
                sequences = batch
            
            # Move data to device
            sequences = sequences.to(device)
            batch_size = sequences.size(0)
            
            # Debug logging
            if batch_idx == 0:  # Log first batch stats
                logging.info(f"Validation batch stats - Shape: {sequences.shape}, Mean: {sequences.mean():.6f}, Std: {sequences.std():.6f}")
                logging.info(f"Range - Min: {sequences.min():.6f}, Max: {sequences.max():.6f}")
            
            # Forward pass
            output = model(sequences)
            
            # Debug output stats
            if batch_idx == 0:  # Log first batch output stats
                logging.info(f"Model output stats - Shape: {output.shape}, Mean: {output.mean():.6f}, Std: {output.std():.6f}")
                logging.info(f"Output range - Min: {output.min():.6f}, Max: {output.max():.6f}")
            
            # Calculate loss (predicting next timestep)
            loss = criterion(output[:, :-1, :], sequences[:, 1:, :])
            
            if torch.isnan(loss):
                logging.error(f"NaN loss in validation batch {batch_idx}")
                logging.error(f"Loss components - MSE inputs:")
                logging.error(f"Pred shape: {output[:, :-1, :].shape}, Target shape: {sequences[:, 1:, :].shape}")
                logging.error(f"Pred stats - Mean: {output[:, :-1, :].mean():.6f}, Std: {output[:, :-1, :].std():.6f}")
                logging.error(f"Target stats - Mean: {sequences[:, 1:, :].mean():.6f}, Std: {sequences[:, 1:, :].std():.6f}")
                continue
            
            total_loss += loss.item() * batch_size
    
    return total_loss / len(val_loader.dataset)

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, is_best=False):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model if this is the best so far
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_losses(train_losses, val_losses, save_dir):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(save_dir, f'loss_plot_{timestamp}.png'))
    plt.close()

def main():
    # Training settings
    config = {
        'data_dir': 'output',
        'log_dir': 'logs',
        'checkpoint_dir': 'checkpoints',
        'plot_dir': 'training_plots',
        'sequence_length': 10,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
        'n_layers': 8,
        'n_heads': 8,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'max_grad_norm': 1.0
    }
    
    # Setup logging
    log_file = setup_logging(config['log_dir'])
    logging.info(f"Starting training with config: {json.dumps(config, indent=2)}")
    
    # Log device info
    logging.info(f"Using device: {config['device']}")
    if config['device'] == 'cuda':
        logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA device count: {torch.cuda.device_count()}")
    elif config['device'] == 'mps':
        logging.info("Metal Performance Shaders (MPS) backend is being used for GPU acceleration")
    else:
        logging.info("Running on CPU")
    
    # Split patients
    train_ids, val_ids, test_ids = split_patients(
        config['data_dir'], 
        config['train_ratio'], 
        config['val_ratio']
    )
    logging.info(f"Train patients: {len(train_ids)}, Val patients: {len(val_ids)}, Test patients: {len(test_ids)}")
    
    # Get data files
    train_files = get_embeddings_files(config['data_dir'], train_ids)
    val_files = get_embeddings_files(config['data_dir'], val_ids)
    test_files = get_embeddings_files(config['data_dir'], test_ids)
    
    # Create datasets
    train_dataset = BrainStateDataset(train_files, config['sequence_length'])
    val_dataset = BrainStateDataset(val_files, config['sequence_length'])
    test_dataset = BrainStateDataset(test_files, config['sequence_length'])
    
    # Get embedding dimension from dataset
    config['model_dim'] = train_dataset.embedding_dim
    logging.info(f"Using embedding dimension from data: {config['model_dim']}")
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                            shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                          collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                           collate_fn=collate_fn)
    
    # Initialize model
    model_args = ModelArgs(
        dim=config['model_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        max_batch_size=config['batch_size'],
        max_seq_len=config['sequence_length'],
        device=config['device']
    )
    
    model = Transformer(model_args).to(config['device'])
    
    # Log model parameters
    n_params = count_parameters(model)
    logging.info(f"Number of trainable parameters: {n_params:,}")
    
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config['num_epochs']):
        logging.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, 
                               config['device'], epoch, config['max_grad_norm'])
        
        # Check for NaN loss
        if np.isnan(train_loss):
            logging.error("Training stopped due to NaN loss")
            break
            
            
        logging.info(f"Training Loss: {train_loss:.6f}")
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, config['device'])
        logging.info(f"Validation Loss: {val_loss:.6f}")
        val_losses.append(val_loss)
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logging.info("New best model!")
        
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            config['checkpoint_dir'], is_best
        )
        
        # Plot losses every 5 epochs
        if (epoch + 1) % 5 == 0:
            plot_losses(train_losses, val_losses, config['plot_dir'])
    
    # Final loss plot
    plot_losses(train_losses, val_losses, config['plot_dir'])
    logging.info("Training completed!")

if __name__ == "__main__":
    main() 