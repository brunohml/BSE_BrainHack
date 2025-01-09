import numpy as np
from pacmap import PaCMAP
from pacmap import sample_neighbors_pair
import pickle
import hdbscan
import os
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from utils.unpickler import process_single_patient as unpack_single_patient

def setup_output_directory(patient_id):
    """Create output directory structure for the patient."""
    output_dir = os.path.join('output', f"Epat{patient_id}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def find_embeddings_file(patient_id):
    """Find the embeddings file for a patient. If not found, generate it."""
    # Convert integer to Epat format
    patient_id_str = f"Epat{patient_id}"
    
    output_dir = os.path.join('output', patient_id_str)
    version_file = f'embeddings_{patient_id_str}.pkl'
    version_path = os.path.join(output_dir, version_file)
    
    if os.path.exists(version_path):
        return version_path
        
    # If embeddings file doesn't exist, generate it using unpickler
    print(f"\nEmbeddings file not found for patient {patient_id_str}. Generating it now...")
    pickle_file = 'source_pickles/raw_embeddings_1024d.pkl'
    
    if not os.path.exists(pickle_file):
        raise FileNotFoundError(
            f"Source pickle file not found at {pickle_file}. "
            "Please ensure the source pickle file exists in the source_pickles directory."
        )
    
    try:
        # Run unpickler to generate embeddings file
        version_path = unpack_single_patient(pickle_file, patient_id_str)
        print(f"Successfully generated embeddings file at {version_path}")
        return version_path
    except Exception as e:
        raise RuntimeError(f"Failed to generate embeddings file: {str(e)}")

def apply_pacmap_and_clustering(embeddings, do_10d=False, 
                              mn_ratio=12.0, fp_ratio=1.0, n_neighbors=None,
                              lr=0.01):
    """Apply PaCMAP dimensionality reduction and HDBSCAN clustering.
    
    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        do_10d: whether to also compute 10D reduction (used for clustering)
        mn_ratio: PaCMAP MN_ratio parameter
        fp_ratio: PaCMAP FP_ratio parameter
        n_neighbors: number of neighbors for PaCMAP (None for auto)
        lr: learning rate for PaCMAP optimization
        
    Returns:
        tuple of (2D embeddings, 10D embeddings if do_10d=True else None, cluster labels if do_10d=True else None)
    """
    # Prepare PaCMAP parameters
    pacmap_params = {
        'n_components': 2,
        'MN_ratio': mn_ratio,
        'FP_ratio': fp_ratio,
        'distance': 'angular',
        'verbose': True,
        'lr': lr
    }
    
    # Only add n_neighbors if it's provided
    if n_neighbors is not None:
        pacmap_params['n_neighbors'] = n_neighbors

    # Compute 10D embeddings if requested (used for clustering)
    dim10_space = None
    cluster_labels = None
    if do_10d:
        print("\nReducing to 10 dimensions using PaCMAP...")
        pacmap_10d = PaCMAP(**{**pacmap_params, 'n_components': 10})
        dim10_space = pacmap_10d.fit_transform(embeddings)
        
        print("\nPerforming HDBSCAN clustering...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=100)
        cluster_labels = clusterer.fit_predict(dim10_space)
    
    # Compute 2D embeddings
    print("\nReducing to 2 dimensions using PaCMAP...")
    pacmap_2d = PaCMAP(**pacmap_params)
    dim2_space = pacmap_2d.fit_transform(embeddings)
    
    return dim2_space, dim10_space, cluster_labels

def get_param_suffix(mn_ratio, fp_ratio, lr, n_neighbors):
    """Generate filename suffix based on parameters."""
    nn_str = f"NN{n_neighbors}" if n_neighbors is not None else "NN0"
    return f"_MN{mn_ratio}_FP{fp_ratio}_LR{lr}_{nn_str}"

def process_single_patient(patient_id, do_10d=False,
                         mn_ratio=12.0, fp_ratio=1.0, n_neighbors=None,
                         lr=0.01):
    """Process embeddings for a single patient."""
    print("\n=== Evaporating Pickles ===\n")
    output_dir = setup_output_directory(patient_id)
    
    # Load embeddings data
    print("\nLoading embeddings from unpickler output...")
    embeddings_path = find_embeddings_file(patient_id)
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    
    # Get the flattened embeddings
    flattened_data = data['patient_embeddings']
    print(f"Loaded embeddings shape: {flattened_data.shape}")
    
    # Apply PaCMAP and clustering
    dim2_space, dim10_space, cluster_labels = apply_pacmap_and_clustering(
        flattened_data, do_10d=do_10d,
        mn_ratio=mn_ratio, fp_ratio=fp_ratio, n_neighbors=n_neighbors,
        lr=lr
    )
    
    # Save visualization
    plt.figure(figsize=(10, 8))
    if do_10d:
        plt.scatter(dim2_space[:, 0], dim2_space[:, 1], c=cluster_labels, cmap='Spectral', s=0.0005)
        plt.colorbar(label='Cluster')
    else:
        plt.scatter(dim2_space[:, 0], dim2_space[:, 1], s=0.0005)
    plt.title(f'Brain State Embeddings for Patient {patient_id}\nMN={mn_ratio}, FP={fp_ratio}, n={n_neighbors}')
    
    # Generate parameter suffix for filenames
    param_suffix = get_param_suffix(mn_ratio, fp_ratio, lr, n_neighbors)
    
    # Save plot with parameters in filename
    plot_path = os.path.join(output_dir, f'pointcloud_Epat{patient_id}{param_suffix}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save processed data
    output_data = {
        'patient_id': patient_id,
        'transformed_points_2d': dim2_space,
        'transformed_points_10d': dim10_space if do_10d else None,
        'cluster_labels': cluster_labels if do_10d else None,
        'file_indices': data['file_indices'],
        'window_indices': data['window_indices'],
        'start_times': data['start_times'],
        'stop_times': data['stop_times'],
        'original_shape': data['original_shape'],
        'seizure_types': None,
        'seizure_events': None,
        'pacmap_params': {
            'mn_ratio': mn_ratio,
            'fp_ratio': fp_ratio,
            'n_neighbors': n_neighbors,
            'do_10d': do_10d
        }
    }
    
    # Save processed data with parameters in filename
    output_path = os.path.join(output_dir, f'manifold_Epat{patient_id}{param_suffix}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\nProcessing complete. Files saved to {output_dir}")
    return output_path, plot_path

def process_all_patients(do_10d=False, mn_ratio=12.0, fp_ratio=1.0, 
                        n_neighbors=None, lr=0.01):
    """Process all patients that have embeddings files."""
    print("\n=== Processing All Patients ===\n")
    
    # Check if output directory exists, create if not
    if not os.path.exists('output'):
        os.makedirs('output')
        print("Created output directory")
    
    # Check for source pickle file
    pickle_file = 'source_pickles/raw_embeddings_1024d.pkl'
    if not os.path.exists(pickle_file):
        raise FileNotFoundError(
            f"Source pickle file not found at {pickle_file}. "
            "Please ensure the source pickle file exists in the source_pickles directory."
        )
    
    # First, run unpickler for all patients if no embeddings exist
    patient_dirs = [d for d in os.listdir('output') 
                   if os.path.isdir(os.path.join('output', d))]
    
    if not patient_dirs:
        print("\nNo patient embeddings found. Running unpickler for all patients...")
        try:
            from utils.unpickler import process_all_patients as unpack_all_patients
            unpack_all_patients(pickle_file)
            # Refresh the list of patient directories
            patient_dirs = [d for d in os.listdir('output') 
                          if os.path.isdir(os.path.join('output', d))]
        except Exception as e:
            raise RuntimeError(f"Failed to unpack patient data: {str(e)}")
    
    print(f"\nFound {len(patient_dirs)} patients to process")
    
    for patient_dir in patient_dirs:
        try:
            # Extract patient number from directory name
            patient_num = int(patient_dir.replace('Epat', ''))
            print(f"\n=== Processing {patient_dir} ===")
            process_single_patient(patient_num, do_10d=do_10d,
                                mn_ratio=mn_ratio, fp_ratio=fp_ratio,
                                n_neighbors=n_neighbors, lr=lr)
        except Exception as e:
            print(f"Error processing {patient_dir}: {e}")
            continue

def process_merged_patients(patient_ids, do_10d=False,
                          mn_ratio=12.0, fp_ratio=1.0, n_neighbors=None,
                          lr=0.01):
    """Process and merge embeddings from multiple patients."""
    print("\n=== Merging Patient Embeddings ===\n")
    
    # Create output directory using concatenated IDs
    output_dir = os.path.join('output', '_'.join(patient_ids))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize lists to store merged data
    merged_data = {
        'patient_embeddings': [],
        'patient_ids': [],
        'file_indices': [],
        'window_indices': [],
        'start_times': [],
        'stop_times': []
    }
    
    # Load and merge data from each patient
    for patient_id in patient_ids:
        print(f"\nLoading data for patient {patient_id}...")
        embeddings_path = find_embeddings_file(patient_id)
        
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
            
        if data['patient_id'] != patient_id:
            raise ValueError(f"Patient ID mismatch in data file. Expected {patient_id}, found {data['patient_id']}")
        
        # Append data
        merged_data['patient_embeddings'].append(data['patient_embeddings'])
        merged_data['patient_ids'].extend([patient_id] * len(data['patient_embeddings']))
        merged_data['file_indices'].extend(data['file_indices'])
        merged_data['window_indices'].extend(data['window_indices'])
        merged_data['start_times'].extend(data['start_times'])
        merged_data['stop_times'].extend(data['stop_times'])
        
        print(f"Added {len(data['patient_embeddings'])} embeddings")
    
    # Convert lists to arrays where appropriate
    merged_data['patient_embeddings'] = np.vstack(merged_data['patient_embeddings'])
    print(f"\nTotal merged embeddings shape: {merged_data['patient_embeddings'].shape}")
    
    # Apply PaCMAP and clustering
    dim2_space, dim10_space, cluster_labels = apply_pacmap_and_clustering(
        merged_data['patient_embeddings'], do_10d=do_10d,
        mn_ratio=mn_ratio, fp_ratio=fp_ratio, n_neighbors=n_neighbors,
        lr=lr
    )
    
    # Save visualization
    plt.figure(figsize=(10, 8))
    if do_10d:
        plt.scatter(dim2_space[:, 0], dim2_space[:, 1], c=cluster_labels, cmap='Spectral', s=0.0005)
        plt.colorbar(label='Cluster')
    else:
        # Color points by patient
        unique_patients = sorted(set(merged_data['patient_ids']))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_patients)))
        
        for idx, (pat_id, color) in enumerate(zip(unique_patients, colors)):
            mask = np.array(merged_data['patient_ids']) == pat_id
            plt.scatter(dim2_space[mask, 0], 
                      dim2_space[mask, 1], 
                      color=color, 
                      label=pat_id,
                      s=0.5,
                      alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(f'Brain State Embeddings for Merged Patients: {", ".join(patient_ids)}\nMN={mn_ratio}, FP={fp_ratio}, n={n_neighbors}')
    
    # Generate parameter suffix for filenames
    param_suffix = get_param_suffix(mn_ratio, fp_ratio, lr, n_neighbors)
    
    merged_name = "_".join([f"Epat{num}" for num in patient_ids])
    plot_path = os.path.join(output_dir, f'pointcloud_{merged_name}{param_suffix}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save processed data
    output_data = {
        'patient_ids': merged_data['patient_ids'],
        'transformed_points_2d': dim2_space,
        'transformed_points_10d': dim10_space if do_10d else None,
        'cluster_labels': cluster_labels if do_10d else None,
        'file_indices': merged_data['file_indices'],
        'window_indices': merged_data['window_indices'],
        'start_times': merged_data['start_times'],
        'stop_times': merged_data['stop_times'],
        'seizure_types': None,
        'seizure_events': None,
        'pacmap_params': {
            'mn_ratio': mn_ratio,
            'fp_ratio': fp_ratio,
            'n_neighbors': n_neighbors,
            'do_10d': do_10d
        }
    }
    
    # Save processed data with parameters in filename
    output_path = os.path.join(output_dir, f'manifold_{merged_name}{param_suffix}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\nProcessing complete. Files saved to {output_dir}")
    return output_path, plot_path

def process_merged_sweep(patient_ids, mn_values, fp_values, bse_version=None):
    """Process merged embeddings with PaCMAP parameter sweep."""
    print("\n=== Merging Patient Embeddings and Performing Parameter Sweep ===\n")
    
    # Create output directory using concatenated IDs
    base_dir = os.path.join('output', '_'.join(patient_ids))
    sweep_dir = os.path.join(base_dir, 'pacmap_sweep')
    if not os.path.exists(sweep_dir):
        os.makedirs(sweep_dir)
    
    # Initialize lists to store merged data
    merged_data = {
        'patient_embeddings': [],
        'patient_ids': [],
        'file_indices': [],
        'window_indices': [],
        'start_times': [],
        'stop_times': []
    }
    
    # Load and merge data from each patient
    for patient_id in patient_ids:
        print(f"\nLoading data for patient {patient_id}...")
        embeddings_path = find_embeddings_file(patient_id)
        
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
            
        if data['patient_id'] != patient_id:
            raise ValueError(f"Patient ID mismatch in data file. Expected {patient_id}, found {data['patient_id']}")
        
        # Append data
        merged_data['patient_embeddings'].append(data['patient_embeddings'])
        merged_data['patient_ids'].extend([patient_id] * len(data['patient_embeddings']))
        merged_data['file_indices'].extend(data['file_indices'])
        merged_data['window_indices'].extend(data['window_indices'])
        merged_data['start_times'].extend(data['start_times'])
        merged_data['stop_times'].extend(data['stop_times'])
        
        print(f"Added {len(data['patient_embeddings'])} embeddings")
    
    # Convert lists to arrays where appropriate
    merged_data['patient_embeddings'] = np.vstack(merged_data['patient_embeddings'])
    print(f"\nTotal merged embeddings shape: {merged_data['patient_embeddings'].shape}")
    
    # Perform parameter sweep
    print("\nPerforming PaCMAP parameter sweep...")
    for mn_ratio in mn_values:
        for fp_ratio in fp_values:
            print(f"\nTrying MN_ratio={mn_ratio}, FP_ratio={fp_ratio}")
            
            # Apply PaCMAP with current parameters (no 10D reduction or clustering needed for sweep)
            dim2_space, _, _ = apply_pacmap_and_clustering(
                merged_data['patient_embeddings'],
                do_10d=False,
                mn_ratio=mn_ratio,
                fp_ratio=fp_ratio
            )
            
            # Create visualization
            plt.figure(figsize=(10, 8))
            # Color points by patient
            unique_patients = sorted(set(merged_data['patient_ids']))
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_patients)))
            
            for idx, (pat_id, color) in enumerate(zip(unique_patients, colors)):
                mask = np.array(merged_data['patient_ids']) == pat_id
                plt.scatter(dim2_space[mask, 0], 
                          dim2_space[mask, 1], 
                          color=color, 
                          label=pat_id,
                          s=0.5,
                          alpha=0.5)
            
            plt.title(f'Merged Brain State Embeddings\nMN={mn_ratio}, FP={fp_ratio}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Generate parameter suffix for filenames
            param_suffix = get_param_suffix(mn_ratio, fp_ratio, 0.01, None)  # Using default lr and no n_neighbors
            
            # Save plot with parameters in filename
            plot_path = os.path.join(sweep_dir, f'pointcloud_{merged_name}{param_suffix}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved plot to {plot_path}")
    
    print("\nParameter sweep complete")
    return sweep_dir

def main():
    parser = argparse.ArgumentParser(description='Process brain state embeddings for a specific patient.')
    parser.add_argument('--patient_id', type=int, help='Patient ID (e.g., 37)')
    parser.add_argument('--all', action='store_true', help='Process all patients')
    parser.add_argument('--merge', type=int, nargs='+', help='List of patient IDs as integers (e.g., 37 38)')
    parser.add_argument('--n_neighbors', type=int, help='Number of neighbors for PaCMAP (default: auto)')
    parser.add_argument('--do_10d', action='store_true', help='Perform 10D reduction and clustering')
    parser.add_argument('--mn_ratio', type=float, help='PaCMAP MN_ratio parameter (default: 12.0)')
    parser.add_argument('--fp_ratio', type=float, help='PaCMAP FP_ratio parameter (default: 1.0)')
    parser.add_argument('--lr', type=float, default=0.01,
                      help='Learning rate for PaCMAP optimization (default: 0.01)')
    
    args = parser.parse_args()
    
    # Get PaCMAP parameters
    pacmap_params = {
        'distance': 'angular',
        'lr': args.lr
    }
    if args.n_neighbors is not None:
        pacmap_params['n_neighbors'] = args.n_neighbors
    if args.mn_ratio is not None:
        pacmap_params['mn_ratio'] = args.mn_ratio
    if args.fp_ratio is not None:
        pacmap_params['fp_ratio'] = args.fp_ratio
    
    # Remove 'distance' from pacmap_params before passing to process functions
    process_params = {k: v for k, v in pacmap_params.items() if k != 'distance'}
    
    if args.merge:
        # No need to convert to Epat format here, process_merged_patients will handle it
        process_merged_patients(args.merge, do_10d=args.do_10d, **process_params)
    elif args.all or (args.patient_id and str(args.patient_id).lower() == 'all'):
        process_all_patients(do_10d=args.do_10d, **process_params)
    elif args.patient_id:
        # No need for try/except since --patient_id is already type=int
        process_single_patient(args.patient_id, do_10d=args.do_10d, **process_params)
    else:
        print("Error: Please specify either --patient_id, --all, --merge, or --load_pickle")
        return

if __name__ == "__main__":
    main()