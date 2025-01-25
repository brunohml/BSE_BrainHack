import pickle
import os

def filter_patient_data(source_pickle, target_patients):
    """
    Filter pickle data to only include specified patients.
    
    Args:
        source_pickle: Path to source pickle file
        target_patients: List of patient IDs to keep
    """
    print(f"Loading source pickle file: {source_pickle}")
    with open(source_pickle, 'rb') as f:
        data_tuple = pickle.load(f)
    
    # Get indices of target patients
    patient_indices = []
    for patient_id in target_patients:
        try:
            idx = data_tuple[0].index(patient_id)
            patient_indices.append(idx)
        except ValueError:
            print(f"Warning: Patient {patient_id} not found in dataset")
    
    # Create new filtered data tuple
    filtered_data = (
        [data_tuple[0][i] for i in patient_indices],  # patient IDs
        [data_tuple[1][i] for i in patient_indices],  # embeddings
        [data_tuple[2][i] for i in patient_indices],  # start times
        [data_tuple[3][i] for i in patient_indices]   # stop times
    )
    
    return filtered_data

def main():
    # Define paths
    source_path = os.path.join('source_pickles', 'raw_embeddings_1024d.pkl')
    output_path = '1024d_embeddings.pkl'
    
    # Define target patients
    target_patients = ['Epat30', 'Epat31', 'Epat33', 'Epat37']
    
    # Filter data
    print(f"Filtering data for patients: {', '.join(target_patients)}")
    filtered_data = filter_patient_data(source_path, target_patients)
    
    # Save filtered data
    print(f"Saving filtered data to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(filtered_data, f)
    
    print("Done!")

if __name__ == "__main__":
    main()
