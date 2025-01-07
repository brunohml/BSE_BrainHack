import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import re

def setup_output_directory(patient_id):
    """Create output directory structure for the patient."""
    output_dir = os.path.join('output', patient_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def extract_epoch_number(pickle_file_path):
    """Extract epoch number from pickle filename."""
    filename = os.path.basename(pickle_file_path)
    # Try to find epoch number in the format 'epoch425' or just '425'
    match = re.search(r'epoch(\d+)|_(\d+)_', filename)
    if match:
        # Return the first non-None group (either from 'epoch\d+' or '_\d+_')
        return next(num for num in match.groups() if num is not None)
    return None

def process_single_patient(pickle_file_path, patient_id):
    """Process embeddings for a single patient."""
    print("\n=== Evaporating Pickle ===\n")
    
    output_dir = setup_output_directory(patient_id)
    
    # Load and extract data
    print("\nLoading embeddings from pickle file...")
    with open(pickle_file_path, 'rb') as f:
        data_tuple = pickle.load(f)
    
    try:
        patient_index = data_tuple[0].index(patient_id)
    except ValueError:
        print("\nAvailable patient IDs:")
        for i, pid in enumerate(data_tuple[0]):
            print(f"{i}: {pid}")
        raise ValueError(f"Patient ID '{patient_id}' not found in dataset")
    
    # Extract patient data
    patient_embeddings = data_tuple[1][patient_index]
    patient_start_times = data_tuple[2][patient_index]
    patient_stop_times = data_tuple[3][patient_index]
    
    # Process embeddings
    print("\nProcessing embeddings...")
    all_latent_data = np.array([np.array(embedding) for embedding in patient_embeddings])
    print(f"Original shape of latent data: {all_latent_data.shape}")
    
    # Reshape data
    reshaped_data = all_latent_data.transpose(0, 2, 1)
    flattened_data = reshaped_data.reshape(-1, 1024)
    print(f"Shape after flattening: {flattened_data.shape}")
    
    # Create parallel arrays
    expanded_start_times = []
    expanded_stop_times = []
    expanded_file_indices = []
    expanded_window_indices = []

    for file_idx in range(len(patient_start_times)):
        file_start = patient_start_times[file_idx]
        
        for window_idx in range(203):
            window_offset = window_idx * 5
            window_start = file_start + pd.Timedelta(seconds=window_offset)
            window_stop = window_start + pd.Timedelta(seconds=10)  # 10 second window
            
            expanded_start_times.append(window_start)
            expanded_stop_times.append(window_stop)
            expanded_file_indices.append(file_idx)
            expanded_window_indices.append(window_idx)
    
    print(f"start times shape: {len(expanded_start_times)}")
    print(f"stop times shape: {len(expanded_stop_times)}")
    print(f"file indices shape: {len(expanded_file_indices)}")
    print(f"window indices shape: {len(expanded_window_indices)}")
    
    # Save processed data
    output_data = {
        'patient_id': patient_id,
        'patient_embeddings': flattened_data,
        'file_indices': expanded_file_indices,
        'window_indices': expanded_window_indices,
        'start_times': expanded_start_times,
        'stop_times': expanded_stop_times,
        'original_shape': all_latent_data.shape,
        'sleep_labels': None
    }
    
    output_path = os.path.join(output_dir, f'embeddings_{patient_id}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\nProcessing complete. Files saved to {output_dir}")
    return output_path

def process_all_patients(pickle_file_path):
    """Process embeddings for all patients in the dataset."""
    print("\n=== Processing All Patients ===\n")
    
    # Load data
    print("\nLoading embeddings from pickle file...")
    with open(pickle_file_path, 'rb') as f:
        data_tuple = pickle.load(f)
    
    patient_ids = data_tuple[0]
    print(f"\nFound {len(patient_ids)} patients to process")
    
    processed_files = []
    for patient_id in patient_ids:
        try:
            print(f"\n=== Processing {patient_id} ===")
            output_path = process_single_patient(pickle_file_path, patient_id)
            processed_files.append(output_path)
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            continue
    
    print(f"\nProcessing complete. Processed {len(processed_files)} patients successfully.")
    return processed_files

def read_pickle_structure(pickle_file_path):
    """Read and display the structure of the pickle file contents."""
    print("\n=== Reading Pickle Structure ===\n")
    
    with open(pickle_file_path, 'rb') as f:
        data_tuple = pickle.load(f)
    
    print("Data structure:")
    print(f"Type: {type(data_tuple)}")
    print(f"Length: {len(data_tuple)} elements")
    
    for i, item in enumerate(data_tuple):
        print(f"\nElement {i}:")
        print(f"  Type: {type(item)}")
        
        if isinstance(item, list):
            print(f"  Length: {len(item)} items")
            if len(item) > 0:
                print(f"  First item type: {type(item[0])}")
                if isinstance(item[0], (str, int, float)):
                    print(f"  First few items: {item[:5]}")
        elif isinstance(item, np.ndarray):
            print(f"  Shape: {item.shape}")
            print(f"  dtype: {item.dtype}")
        else:
            print(f"  Value: {item}")

def main():
    parser = argparse.ArgumentParser(description='Process brain state embeddings for specific patients.')
    parser.add_argument('--patient_id', type=str, nargs='+',
                      help='One or more Patient IDs as integers (e.g., 37 38) or "all" to process all patients')
    parser.add_argument('--pickle_file', type=str, required=True,
                      help='Path to source pickle file')
    parser.add_argument('--read', action='store_true',
                      help='Read and display the structure of the pickle file without processing')
    
    args = parser.parse_args()
    
    try:
        if args.read:
            read_pickle_structure(args.pickle_file)
            return
            
        if len(args.patient_id) == 1 and args.patient_id[0].lower() == 'all':
            process_all_patients(args.pickle_file)
        else:
            for patient_str in args.patient_id:
                try:
                    patient_num = int(patient_str)
                    patient_id = f"Epat{patient_num}"
                    print(f"\n=== Processing {patient_id} ===")
                    process_single_patient(args.pickle_file, patient_id)
                except ValueError:
                    print(f"\nError: Patient ID '{patient_str}' is not a valid integer")
                    continue
    except ValueError as e:
        print(f"\nError: {e}")
        exit(1)

if __name__ == "__main__":
    main()