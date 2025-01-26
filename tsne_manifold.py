import numpy as np
from sklearn.manifold import TSNE
import pickle
import hdbscan
import os
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from utils.unpickler_split import process_patient_files as unpack_single_patient
import re
import glob

def setup_output_directory(animal, patient_id):
    """Create output directory structure for the patient."""
    output_dir = os.path.join('output', animal, "tSNE", f"Epat{patient_id}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def find_embeddings_file(animal, patient_id, window_length=60, stride_length=30, data_type='train'):
    """Find the embeddings file for a patient. If not found, generate it."""
    # Convert integer to Epat format
    patient_id_str = f"Epat{patient_id}"
    
    output_dir = os.path.join('output', animal, patient_id_str)
    version_file = f'embeddings_{patient_id_str}_W{window_length}_S{stride_length}_{data_type}.pkl'
    version_path = os.path.join(output_dir, version_file)
    
    if os.path.exists(version_path):
        return version_path
        
    # If embeddings file doesn't exist, generate it using unpickler_split
    print(f"\nEmbeddings file not found for patient {patient_id_str}. Generating it now...")
    
    try:
        # Get all files for this patient
        pattern = os.path.join('source_pickles', animal, 'Epoch*',
                             f'{window_length}SecondWindow_{stride_length}SecondStride',
                             data_type, f'{patient_id_str}_*.pkl')
        patient_files = glob.glob(pattern)
        
        if not patient_files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")
            
        # Run unpickler to generate embeddings file
        version_path = unpack_single_patient(patient_files, animal, patient_id_str, 
                                           window_length, stride_length, data_type)
        print(f"Successfully generated embeddings file at {version_path}")
        return version_path
    except Exception as e:
        raise RuntimeError(f"Failed to generate embeddings file: {str(e)}")

def get_param_suffix(tsne_params):
    """Generate filename suffix based on parameters."""
    return f"_Ncomps{tsne_params['n_components']}_LR{tsne_params['lr']}_PP{tsne_params['pp']}_RNG{tsne_params['rng']}"

def apply_tsne(embeddings, tsne_params):
    """Apply t-SNE dimensionality reduction

    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        lr: learning rate - float or "auto". usually between [10.0, 1000.0]
        pp: perplexity - related to number of nearest neighbors; must be less than n_samples; maybe between 5 and 50
        rng: seed for random number generation
        
    Returns:
        TODO
    """
    # Build tSNE object
    tsne = TSNE(n_components=tsne_params['n_components'], 
                learning_rate=tsne_params['lr'], 
                perplexity=tsne_params['pp'],
                random_state=tsne_params['rng'])

    # Compute embeddings
    print(f"\nReducing to {tsne_params['n_components']} dimensions using t-SNE...")
    reduced_embeddings = tsne.fit_transform(embeddings)
    return reduced_embeddings

def process_single_patient(animal, patient_id, tsne_params, window_length=60, stride_length=30, 
                         data_type='train'):
    """Process embeddings for a single patient."""
    print("\n=== Processing Brain State Embeddings ===\n")
    output_dir = setup_output_directory(animal, patient_id)
    
    # Load embeddings data
    print("\nLoading embeddings from unpickler output...")
    embeddings_path = find_embeddings_file(animal, patient_id, window_length, 
                                         stride_length, data_type)
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    
    # Get the embeddings and reshape them
    embeddings_data = data['patient_embeddings']
    print(f"Original embeddings shape: {embeddings_data.shape}")  # (n_files, n_timepoints, n_features)
    
    # Reshape to (n_files*n_timepoints, n_features)
    n_files, n_timepoints, n_features = embeddings_data.shape
    embeddings_flat = embeddings_data.reshape(-1, n_features)  # Now (n_files*n_timepoints, n_features)
    print(f"Reshaped embeddings: {embeddings_flat.shape}")
    
    # Apply tsne
    reduced = apply_tsne(embeddings_flat, tsne_params)
    
    print(f"t-SNE output shape: {reduced.shape}")
    print(f"t-SNE range - X: [{reduced[:, 0].min():.2f}, {reduced[:, 0].max():.2f}], "
          f"Y: [{reduced[:, 1].min():.2f}, {reduced[:, 1].max():.2f}]")

    # Save visualization
    plt.figure(figsize=(12, 10))
    colors = np.tile(np.arange(n_timepoints), n_files)
    plt.scatter(reduced[:, 0], reduced[:, 1], 
                c=colors, cmap='viridis', s=1, alpha=0.5)
    plt.colorbar(label='Timepoint within window')
    plt.title(f't-SNE Brain State Embeddings for Patient {patient_id}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Generate parameter suffix for filenames
    param_suffix = get_param_suffix(tsne_params)
    
    # Save plot with parameters in filename
    plot_path = os.path.join(output_dir, f'pointcloud_Epat{patient_id}{param_suffix}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save processed data
    output_data = {
        'patient_id': patient_id,
        'transformed_points_2d': reduced,
        'file_indices': np.repeat(np.arange(n_files), n_timepoints),
        'window_indices': np.tile(np.arange(n_timepoints), n_files),
        'start_times': np.repeat(data['start_times'], n_timepoints),
        'stop_times': np.repeat(data['stop_times'], n_timepoints),
        'original_shape': embeddings_data.shape,
        'seizure_types': None,
        'seizure_events': None,
        'tsne_params': {
            'n_components': tsne_params['n_components'],
            'lr': tsne_params['lr'],
            'rng': tsne_params['rng'],
            'perplexity': tsne_params['pp'],
            'window_length': window_length,
            'stride_length': stride_length,
            'data_type': data_type
        }
    }
    
    # Save processed data with parameters in filename
    output_path = os.path.join(output_dir, f'manifold_Epat{patient_id}{param_suffix}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\nProcessing complete. Files saved to {output_dir}")
    return output_path, plot_path

def process_all_patients(animal, tsne_params, window_length=60, stride_length=30, data_type='train'):
    """Process all patients with embeddings files."""
    print("\n=== Processing All Patients ===\n")
    output_dir = os.path.join('output', animal, "tSNE")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    embeddings_files = glob.glob(os.path.join('output', animal, 'Epat*', f'embeddings_Epat*_W{window_length}_S{stride_length}_{data_type}.pkl'))

    print(f"\nFound {len(embeddings_files)} patients to process")
    for embeddings_file in embeddings_files:
        try:
            match = re.search(r'Epat(\d+)', embeddings_file)
            if not match:
                continue
            patient_num = int(match.group(1))

            print(f"\n=== Processing Epat{patient_num} ===")
            process_single_patient(animal, patient_num, tsne_params, window_length, stride_length, data_type)
        except Exception as e:
            print(f"Error processing patient {patient_num}: {e}")
            continue

def process_merged_patients(animal, patient_ids, tsne_params, window_length=60, stride_length=30, data_type='train'):
    """Process and merge embeddings from multiple patients."""
    print("\n=== Merging Patient Embeddings ===\n")
    output_dir = os.path.join('output', animal, "tSNE", '_'.join([f"Epat{pid}" for pid in patient_ids]))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    merged_data = []
    for patient_id in patient_ids:
        try:
            print(f"\nLoading data for patient {patient_id}...")
            embeddings_path = find_embeddings_file(animal, patient_id, window_length, stride_length, data_type)

            with open(embeddings_path, 'rb') as f:
                data = pickle.load(f)

            embeddings = data['patient_embeddings']
            n_files, n_timepoints, n_features = embeddings.shape
            embeddings_flat = embeddings.reshape(-1, n_features)
            merged_data.append(embeddings_flat)

            print(f"Added {n_files * n_timepoints} points for patient {patient_id}")
        except Exception as e:
            print(f"Error loading data for patient {patient_id}: {e}")

    merged_data = np.vstack(merged_data)
    print(f"\nTotal merged embeddings shape: {merged_data.shape}")

    reduced = apply_tsne(merged_data, tsne_params)

    plt.figure(figsize=(12, 10))
    colors = np.concatenate([np.full(embeddings.shape[0], i) for i, embeddings in enumerate(merged_data)])
    plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, cmap='tab10', s=1, alpha=0.5)
    plt.colorbar(label='Patient ID')
    plt.title(f't-SNE Brain State Embeddings for Merged Patients: {patient_ids}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    param_suffix = get_param_suffix(tsne_params)
    plot_path = os.path.join(output_dir, f'pointcloud_merged_{param_suffix}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nMerged processing complete. Files saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Process brain state embeddings using t-SNE.')
    parser.add_argument('--animal', type=str, required=True, help='Animal name (e.g., rhesusmonkey)')
    parser.add_argument('--patient_id', type=int, help='Patient ID (e.g., 37)')
    parser.add_argument('--all', action='store_true', help='Process all patients')
    parser.add_argument('--merge', type=int, nargs='+', help='List of patient IDs as integers (e.g., 37 38)')
    parser.add_argument('--window_length', type=int, default=60, help='Window length in seconds (default: 60)')
    parser.add_argument('--stride_length', type=int, default=30, help='Stride length in seconds (default: 30)')
    parser.add_argument('--data_type', type=str, default='train', choices=['train', 'valfinetune', 'valunseen'], help='Data type to process (default: train)')
    parser.add_argument('--n_components', type=int, default=2, help='Number of t-SNE dimensions (default: 2)')
    parser.add_argument('--lr', type=float, default=200.0, help='Learning rate for t-SNE optimization (default: 200.0)')
    parser.add_argument('--pp', type=float, default=30.0, help='Perplexity for t-SNE optimization (default: 30.0)')
    parser.add_argument('--rng', type=int, default=42, help='Random seed for t-SNE (default: 42)')

    args = parser.parse_args()

    tsne_params = {
        'n_components': args.n_components,
        'lr': args.lr,
        'pp': args.pp,
        'rng': args.rng
    }

    try: 
        if args.merge:
            process_merged_patients(args.animal, args.merge, tsne_params, args.window_length, args.stride_length, args.data_type)
        elif args.all:
            process_all_patients(args.animal, tsne_params, args.window_length, args.stride_length, args.data_type)
        elif args.patient_id:
            process_single_patient(args.animal, args.patient_id, tsne_params, args.window_length, args.stride_length, args.data_type)
        else:
            print("Error: Please specify either --patient_id, --all, or --merge")
            return
    except Exception as e:
        print(f"\nError: {e}")
        exit(1)


if __name__ == "__main__":
    main()