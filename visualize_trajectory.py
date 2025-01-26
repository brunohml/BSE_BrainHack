import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from Transformer import Transformer, ModelArgs
import random
from glob import glob
import argparse
import pandas as pd
from pacmap import PaCMAP

def load_best_model(checkpoint_dir, model_args):
    """Load the best model from checkpoints directory."""
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")
    
    # Initialize model with same architecture
    model = Transformer(model_args)
    
    # Load checkpoint
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def get_random_sequence(embeddings, sequence_length=10, trajectory_length=50):
    """Get a random sequence of consecutive embeddings including context."""
    print(f"Embeddings shape before processing: {embeddings.shape}")
    
    # If embeddings are (n_timepoints, batch_size, feature_dim)
    # Select a random batch
    if len(embeddings.shape) == 3:
        n_timepoints, batch_size, feature_dim = embeddings.shape
        batch_idx = random.randint(0, batch_size - 1)
        embeddings = embeddings[:, batch_idx, :]  # Shape: (n_timepoints, feature_dim)
    
    print(f"Embeddings shape after selecting batch: {embeddings.shape}")
    
    # Need enough windows for both context and trajectory
    total_needed = sequence_length + trajectory_length
    if len(embeddings) < total_needed:
        raise ValueError(
            f"Not enough windows ({len(embeddings)}) for context ({sequence_length}) "
            f"+ trajectory ({trajectory_length}). Need at least {total_needed}."
        )
    
    # Random starting point that allows for context
    start_idx = random.randint(sequence_length, len(embeddings) - trajectory_length)
    
    # Get context (preceding states) and trajectory separately
    context = embeddings[start_idx - sequence_length:start_idx]  # states before
    trajectory = embeddings[start_idx:start_idx + trajectory_length]  # states to predict
    
    print(f"Context sequence shape: {context.shape}")
    print(f"Target trajectory shape: {trajectory.shape}")
    
    return context, trajectory, start_idx

def predict_trajectory(model, context_sequence, device, n_steps=50):
    """Generate predicted trajectory using the model."""
    model.eval()
    predictions = []
    
    # Convert context sequence to tensor and ensure correct shape
    sequence = torch.FloatTensor(context_sequence)
    print(f"Initial context sequence shape: {sequence.shape}")
    
    # Add batch dimension if needed
    if len(sequence.shape) == 2:
        sequence = sequence.unsqueeze(0)  # Add batch dimension
    
    print(f"Context sequence shape after reshape: {sequence.shape}")
    sequence = sequence.to(device)
    
    with torch.no_grad():
        for step in range(n_steps):
            # Get model prediction
            try:
                output = model(sequence)
                print(f"Model output shape at step {step}: {output.shape}")
            except Exception as e:
                print(f"Error at step {step}")
                print(f"Input sequence shape: {sequence.shape}")
                raise e
            
            # Get the last predicted state
            next_state = output[0, -1].cpu().numpy()
            predictions.append(next_state)
            
            # Update sequence: remove oldest state, add prediction
            # sequence[:, 1:] keeps all but first state
            # output[:, -1:] is the new predicted state
            sequence = torch.cat([sequence[:, 1:], 
                                output[:, -1:]], dim=1)
            
            if step % 10 == 0:  # Log every 10 steps to avoid too much output
                print(f"Updated sequence shape at step {step}: {sequence.shape}")
    
    predictions = np.array(predictions)
    print(f"Final predictions shape: {predictions.shape}")
    return predictions

def plot_trajectories(actual_trajectory, predicted_trajectory, manifold_data, start_time, end_time, original_embeddings):
    """Plot actual and predicted trajectories.
    
    Args:
        actual_trajectory: numpy array of shape (trajectory_length, feature_dim)
        predicted_trajectory: numpy array of shape (trajectory_length, feature_dim)
        manifold_data: dictionary containing manifold data including PACMAP parameters
        start_time: datetime object for the start of the trajectory
        end_time: datetime object for the end of the trajectory
        original_embeddings: numpy array of original embeddings to use as reference
    """
    print("Starting visualization process...")
    
    # Create PACMAP instance with fixed parameters
    pacmap = PaCMAP(
        n_components=2,
        MN_ratio=12.0,
        FP_ratio=1.0,
        distance='angular',
        verbose=True,
        lr=0.01
    )
    
    # Reshape original embeddings if needed
    if len(original_embeddings.shape) == 3:
        n_files, n_timepoints, n_features = original_embeddings.shape
        original_embeddings = original_embeddings.reshape(-1, n_features)
    print(f"Original embeddings shape: {original_embeddings.shape}")
    
    # Combine all data: original embeddings first, then trajectories
    all_data = np.vstack([original_embeddings, actual_trajectory, predicted_trajectory])
    print(f"Combined data shape: {all_data.shape}")
    
    # Fit and transform all data
    print("Performing PACMAP dimensionality reduction...")
    all_2d = pacmap.fit_transform(all_data)
    print("PACMAP reduction complete")
    
    print("Processing data for visualization...")
    # Split back into components
    n_orig = len(original_embeddings)
    n_traj = len(actual_trajectory)
    actual_2d = all_2d[n_orig:n_orig + n_traj]
    predicted_2d = all_2d[n_orig + n_traj:]
    background_2d = all_2d[:n_orig]
    
    # Create time points for x-axis
    time_points = pd.date_range(start=start_time, end=end_time, periods=len(actual_trajectory))
    
    print("Creating plots...")
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    print("Plotting background points...")
    # Plot background points in grey
    ax1.scatter(background_2d[:, 0], background_2d[:, 1], c='lightgrey', s=1, alpha=0.1, label='All States')
    
    print("Plotting trajectories...")
    # Plot actual trajectory in red and predicted in green
    ax1.plot(predicted_2d[:, 0], predicted_2d[:, 1], 'g-', label='Predicted Trajectory', alpha=0.7, linewidth=2)
    ax1.plot(actual_2d[:, 0], actual_2d[:, 1], 'r-', label='Actual Trajectory', alpha=0.7, linewidth=2)
    ax1.set_title('Brain State Trajectories in 2D Space')
    ax1.legend()
    
    print("Plotting time evolution...")
    # Plot trajectories over time
    ax2.plot(time_points, predicted_2d[:, 0], 'g-', label='Predicted X', alpha=0.7)
    ax2.plot(time_points, predicted_2d[:, 1], 'g--', label='Predicted Y', alpha=0.7)
    ax2.plot(time_points, actual_2d[:, 0], 'r-', label='Actual X', alpha=0.7)
    ax2.plot(time_points, actual_2d[:, 1], 'r--', label='Actual Y', alpha=0.7)
    ax2.set_title('Brain State Coordinates Over Time')
    ax2.legend()
    
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    plots_dir = 'training_plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Save plot with timestamp
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    plot_path = os.path.join(plots_dir, f'trajectory_visualization_{timestamp}.png')
    print(f"Saving plot to {plot_path}...")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualization complete. Plot saved.")

def get_test_patients(data_dir, train_ratio=0.7, val_ratio=0.15):
    """Get list of test patients based on the same split used in training."""
    # Get all patient directories
    patient_dirs = glob(os.path.join(data_dir, 'jackal', 'Epat*'))
    patient_ids = [os.path.basename(d).replace('Epat', '') for d in patient_dirs]
    
    if not patient_ids:
        raise ValueError("No patient directories found")
    
    # Sort to ensure same split as training
    patient_ids.sort()
    
    # Calculate split indices
    n_patients = len(patient_ids)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    # Get test patient IDs
    test_ids = patient_ids[n_train + n_val:]
    
    if not test_ids:
        raise ValueError("No test patients found with current split ratios")
    
    return test_ids

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize brain state trajectories using trained model')
    parser.add_argument('--model_path', type=str, 
                      help='Path to specific model checkpoint (default: checkpoints/best_model.pt)',
                      default='checkpoints/best_model.pt')
    parser.add_argument('--data_dir', type=str, default='output',
                      help='Directory containing patient data (default: output)')
    parser.add_argument('--patient_id', type=str,
                      help='Specific patient ID to process (must be a test patient)')
    parser.add_argument('--sequence_length', type=int, default=10,
                      help='Length of input sequence for model (default: 10)')
    parser.add_argument('--trajectory_length', type=int, default=50,
                      help='Length of trajectory to visualize (default: 50)')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                      help='Training set ratio used during model training')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                      help='Validation set ratio used during model training')
    parser.add_argument('--animal', type=str, default='jackal',
                      help='Animal name (default: jackal)')
    parser.add_argument('--window_length', type=int, default=60,
                      help='Window length in seconds (default: 60)')
    parser.add_argument('--stride_length', type=int, default=30,
                      help='Stride length in seconds (default: 30)')
    parser.add_argument('--data_type', type=str, default='train',
                      help='Data type (train/test/val) (default: train)')
    
    args = parser.parse_args()
    
    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Configuration
    config = {
        'data_dir': args.data_dir,
        'checkpoint_dir': os.path.dirname(args.model_path),
        'sequence_length': args.sequence_length,
        'trajectory_length': args.trajectory_length,
        'model_dim': 512,
        'n_layers': 8,
        'n_heads': 8,
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
        'batch_size': 1,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'animal': args.animal,
        'window_length': args.window_length,
        'stride_length': args.stride_length,
        'data_type': args.data_type
    }
    
    # Get test patient IDs
    test_ids = get_test_patients(config['data_dir'], 
                               config['train_ratio'], 
                               config['val_ratio'])
    print(f"Available test patients: {test_ids}")
    
    # Select patient
    if args.patient_id:
        if args.patient_id not in test_ids:
            raise ValueError(
                f"Patient {args.patient_id} is not in the test set. "
                f"Please choose from: {test_ids}"
            )
        patient_id = args.patient_id
    else:
        patient_id = random.choice(test_ids)
    
    patient_dir = os.path.join(config['data_dir'], config['animal'], f'Epat{patient_id}')
    print(f"\nProcessing test patient {patient_id}")
    
    # Load embeddings with correct file pattern
    embeddings_path = os.path.join(
        patient_dir, 
        f'embeddings_Epat{patient_id}_W{config["window_length"]}_S{config["stride_length"]}_{config["data_type"]}.pkl'
    )
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"Embeddings file not found at {embeddings_path}. "
            f"Expected file pattern: embeddings_Epat{patient_id}_W{config['window_length']}_S{config['stride_length']}_{config['data_type']}.pkl"
        )
        
    print(f"Loading embeddings from {embeddings_path}")
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    
    embeddings = data['patient_embeddings']
    print(f"Original embeddings shape: {embeddings.shape}")
    
    # Convert embeddings to numpy if they're not already
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()
    
    # Load manifold data with correct file pattern
    manifold_path = os.path.join(
        patient_dir, 
        f'manifold_Epat{patient_id}_MN12.0_FP1.0_LR0.01_NN0.pkl'
    )
    if not os.path.exists(manifold_path):
        raise FileNotFoundError(
            f"Manifold file not found at {manifold_path}. "
            "Please run create_manifold_split.py first to generate the PACMAP visualization."
        )
        
    print(f"Loading manifold data from {manifold_path}")
    with open(manifold_path, 'rb') as f:
        manifold_data = pickle.load(f)
    
    points_2d = manifold_data['transformed_points_2d']
    print(f"Loaded 2D points with shape: {points_2d.shape}")
    
    # Initialize model arguments
    model_args = ModelArgs(
        dim=config['model_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        max_batch_size=config['batch_size'],
        max_seq_len=config['sequence_length'],
        device=config['device']
    )
    
    # Load specified model checkpoint
    model = Transformer(model_args)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config['device'])
    model.eval()
    
    print(f"Loaded model from {args.model_path}")
    print(f"Model checkpoint was from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")
    
    # Get random sequence with context
    context, trajectory, start_idx = get_random_sequence(
        embeddings, 
        sequence_length=config['sequence_length'],
        trajectory_length=config['trajectory_length']
    )
    
    # Generate predictions
    predicted_sequence = predict_trajectory(
        model, context, config['device'], config['trajectory_length']
    )
    
    # Get the corresponding start and end times for this sequence
    start_time = pd.to_datetime(data['start_times'][start_idx])
    end_time = pd.to_datetime(data['stop_times'][start_idx + config['trajectory_length'] - 1])
    
    # Plot trajectories
    plot_trajectories(
        trajectory, predicted_sequence, manifold_data,
        start_time, end_time, embeddings
    )
    
    print(f"Visualization completed")

if __name__ == "__main__":
    main() 