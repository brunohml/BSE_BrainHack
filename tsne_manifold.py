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

# TODO: update for current params
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
    feature_names = tsne.get_feature_names_out()

    return reduced_embeddings, feature_names

# TODO: PARAMS
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
    reduced, labels = apply_tsne(embeddings_flat, tsne_params)
    
    print(f"t-SNE output shape: {reduced.shape}")
    print(f"t-SNE range - X: [{reduced[:, 0].min():.2f}, {reduced[:, 0].max():.2f}], "
          f"Y: [{reduced[:, 1].min():.2f}, {reduced[:, 1].max():.2f}]")

    # Save visualization
    plt.figure(figsize=(12, 10))
    # TODO: expand for more than 2D
    # if tsne_params['n_components']==2:
    # plt.scatter(reduced[:, 0], reduced[:, 1], cmap='Spectral', s=1)
    # plt.colorbar(label='Cluster')
    colors = np.tile(np.arange(n_timepoints), n_files)
    plt.scatter(reduced[:, 0], reduced[:, 1], 
                c=colors, cmap='viridis', s=1, alpha=0.5)
    plt.colorbar(label='Timepoint within window')
    plt.title(f't-SNE Brain State Embeddings for Patient {patient_id}') #{patient_id}\nMN={mn_ratio}, FP={fp_ratio}, n={n_neighbors}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    # if do_10d:
    #     plt.scatter(dim2_space[:, 0], dim2_space[:, 1], c=cluster_labels, cmap='Spectral', s=1)
    #     plt.colorbar(label='Cluster')
    # else:
    #     # Create a color gradient based on time points within each file
    #     colors = np.tile(np.arange(n_timepoints), n_files)
    #     plt.scatter(dim2_space[:, 0], dim2_space[:, 1], 
    #                c=colors, cmap='viridis', s=1, alpha=0.5)
    #     plt.colorbar(label='Timepoint within window')
    
    
    # Generate parameter suffix for filenames
    param_suffix = get_param_suffix(tsne_params)
    
    # Save plot with parameters in filename
    plot_path = os.path.join(output_dir, f'pointcloud_Epat{patient_id}{param_suffix}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # TODO: update saving processed data
    # # Save processed data
    # output_data = {
    #     'patient_id': patient_id,
    #     'transformed_points_2d': dim2_space,
    #     'transformed_points_10d': dim10_space if do_10d else None,
    #     'cluster_labels': cluster_labels if do_10d else None,
    #     'file_indices': np.repeat(np.arange(n_files), n_timepoints),
    #     'window_indices': np.tile(np.arange(n_timepoints), n_files),
    #     'start_times': np.repeat(data['start_times'], n_timepoints),
    #     'stop_times': np.repeat(data['stop_times'], n_timepoints),
    #     'original_shape': embeddings_data.shape,
    #     'seizure_types': None,
    #     'seizure_events': None,
    #     'pacmap_params': {
    #         'mn_ratio': mn_ratio,
    #         'fp_ratio': fp_ratio,
    #         'n_neighbors': n_neighbors,
    #         'do_10d': do_10d,
    #         'window_length': window_length,
    #         'stride_length': stride_length,
    #         'data_type': data_type
    #     }
    # }
    
    # # Save processed data with parameters in filename
    # output_path = os.path.join(output_dir, f'manifold_Epat{patient_id}{param_suffix}.pkl')
    # with open(output_path, 'wb') as f:
    #     pickle.dump(output_data, f)
    
    # print(f"\nProcessing complete. Files saved to {output_dir}")
    # return output_path, plot_path

def main():
    # Some settings for the tSNE based on arguments TODO
    animal = 'jackal'
    patient_id = '30'
    window_length = 60
    stride_length = 30
    data_type = 'train'

    tsne_params={
        'n_components': 2,
        'lr': 'auto',
        'rng': 15,
        'pp': 30,
    }
    # Common parameters for all processing functions
    common_params = {
        'window_length': window_length, #args.window_length,
        'stride_length': stride_length,#args.stride_length,
        'data_type': data_type,#args.data_type
    }
    
    # TODO: add process_all_patients and process_merged_patients
    # process_single_patient(args.animal, args.patient_id, tsne_params)
    try:
        process_single_patient(animal, patient_id, tsne_params, **common_params)
    except Exception as e:
        print(f"\nError: {e}")
        exit(1)


if __name__ == "__main__":
    main()