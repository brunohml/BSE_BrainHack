import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os
import argparse

def get_merged_patient_dir(patient_nums):
    """Create directory name for merged patients."""
    if isinstance(patient_nums[0], str) and patient_nums[0].startswith('Epat'):
        # If we're passed already-formatted IDs, just sort and join them
        return "_".join(sorted(patient_nums))
    else:
        # If we're passed numbers, format them as Epat IDs
        patient_ids = [f"Epat{num}" for num in sorted(patient_nums)]
        return "_".join(patient_ids)

def setup_output_directory(patient_id):
    """Create output directory structure for the patient."""
    if isinstance(patient_id, list):
        # For multiple patients, use merged directory name
        dir_name = get_merged_patient_dir(patient_id)
        # Use concatenated IDs for manifold filename
        manifold_path = os.path.join('output', dir_name, f'manifold_{dir_name}.pkl')
    else:
        dir_name = f"Epat{patient_id}"
        # Check for individual patient's manifold file
        manifold_path = os.path.join('output', dir_name, f'manifold_{dir_name}.pkl')
    
    if not os.path.exists(manifold_path):
        raise FileNotFoundError(
            f"No manifold file found at {manifold_path}. "
            "Please run pickle_to_cloud.py first."
        )
    
    # Create output directory (now just the patient directory)
    output_dir = os.path.join('output', dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return output_dir, manifold_path

def load_sleep_metadata(excel_path):
    """Load and process sleep metadata from Excel file."""
    print("Loading sleep metadata...")
    sleep_data = pd.read_excel(excel_path)
    
    # Convert datetime columns
    sleep_data['OnsetDatetime'] = pd.to_datetime(sleep_data['OnsetDatetime'])
    sleep_data['OffsetDatetime'] = pd.to_datetime(sleep_data['OffsetDatetime'])
    
    return sleep_data

def find_sleep_stage(start_time, stop_time, sleep_data, patient_id, certainty_threshold):
    """Find sleep stage for a given time window and patient."""
    patient_sleep = sleep_data[
        (sleep_data['PatID'] == patient_id) & 
        (sleep_data['AvgCertainty'] >= certainty_threshold)
    ]
    
    if len(patient_sleep) == 0:
        return 'unknown'
    
    # Check for overlapping sleep stages
    overlapping_stages = patient_sleep[
        (patient_sleep['OnsetDatetime'] <= stop_time) & 
        (patient_sleep['OffsetDatetime'] >= start_time)
    ]
    
    if len(overlapping_stages) > 0:
        sleep_stage = overlapping_stages.iloc[0]['SleepCat']
        # Group N2 and N3 into N
        return 'N' if sleep_stage in ['N2', 'N3'] else sleep_stage
    return 'unknown'

def create_visualization(points_2d, sleep_stages, patient_id, output_dir, filename):
    """Create and save point cloud visualization."""
    plt.figure(figsize=(15, 10))
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot background (unknown) points
    background_mask = np.array(sleep_stages) == 'unknown'
    plt.scatter(points_2d[background_mask, 0], 
               points_2d[background_mask, 1],
               color='gray',
               alpha=0.1,
               s=10,
               label='Unclassified')
    
    # Plot sleep stage points with new colors and labels
    sleep_colors = {
        'W': '#1f77b4',  # NIZ blue
        'N': '#d62728',  # SOZ red
        'R': '#ff7f0e'   # PZ orange
    }
    sleep_labels = {
        'W': 'Wake',
        'N': 'NREM',
        'R': 'REM'
    }
    
    for stage, color in sleep_colors.items():
        mask = np.array(sleep_stages) == stage
        if np.any(mask):
            plt.scatter(points_2d[mask, 0], 
                       points_2d[mask, 1],
                       color=color,
                       alpha=0.75,
                       s=20,
                       label=sleep_labels[stage])
    
    plt.title(f'PaCMAP 2D Projection for Patient {patient_id}\nSleep Stages Highlighted')
    plt.xlabel('PaCMAP Dimension 1')
    plt.ylabel('PaCMAP Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., markerscale=2)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def tag_points(patient_id, sleep_data, certainty_threshold):
    """Tag transformed points with sleep stage metadata."""
    output_dir, patient_data_path = setup_output_directory(patient_id)
    
    # Load data
    print("\nLoading patient data...")
    with open(patient_data_path, 'rb') as f:
        data = pickle.load(f)
    
    print("\nProcessing patient data...")
    sleep_stages = []
    
    # Get unique patient IDs from the data
    if isinstance(patient_id, list):
        unique_patients = sorted(patient_id)
    else:
        unique_patients = [patient_id]
    print(f"Found {len(unique_patients)} patients in data")
    
    # Process each patient's points
    for pat_id in unique_patients:
        print(f"\nTagging sleep stages for Epat{pat_id}")
        
        # Get indices for this patient's points
        if len(unique_patients) > 1:
            # For merged patients, we need to identify which points belong to which patient
            pat_mask = np.array([f"Epat{pat_id}" == pid for pid in data['patient_id']])
        else:
            # For single patient, all points belong to them
            pat_mask = np.ones(len(data['start_times']), dtype=bool)
        
        # Get this patient's sleep metadata
        pat_sleep = sleep_data[
            (sleep_data['PatID'] == f"Epat{pat_id}") & 
            (sleep_data['AvgCertainty'] >= certainty_threshold)
        ]
        
        if len(pat_sleep) == 0:
            print(f"No sleep events found for Epat{pat_id}")
            # Tag all points as unknown for this patient
            pat_stages = ['unknown'] * sum(pat_mask)
        else:
            print(f"Found {len(pat_sleep)} sleep events")
            # Tag each point for this patient
            pat_stages = []
            for start, stop in zip(
                np.array(data['start_times'])[pat_mask],
                np.array(data['stop_times'])[pat_mask]
            ):
                stage = find_sleep_stage(start, stop, sleep_data, f"Epat{pat_id}", certainty_threshold)
                pat_stages.append(stage)
        
        sleep_stages.extend(pat_stages)
    
    # Print summary statistics
    stage_counts = pd.Series(sleep_stages).value_counts()
    print("\nSleep stage distribution:")
    for stage, count in stage_counts.items():
        print(f"{stage}: {count}")
    
    # Save tagged data
    tagged_data = {
        'patient_id': [f"Epat{id}" for id in unique_patients] * (len(sleep_stages) // len(unique_patients)),
        'transformed_points_2d': data['transformed_points_2d'],
        'sleep_stages': sleep_stages,
        'start_times': data['start_times'],
        'stop_times': data['stop_times']
    }
    
    # Determine output filename
    display_name = "_".join([f"Epat{num}" for num in unique_patients])
    tagged_data_path = os.path.join(output_dir, 
        f'tagged_manifold_{display_name}.pkl')
    
    with open(tagged_data_path, 'wb') as f:
        pickle.dump(tagged_data, f)
    
    # Create visualization
    plot_path = create_visualization(
        data['transformed_points_2d'],
        sleep_stages,
        display_name,
        output_dir,
        f'tagged_pointcloud_{display_name}.png'
    )
    
    return tagged_data_path, plot_path

def process_patient(patient_id, sleep_data, certainty_threshold, force_reprocess=False):
    """Process a single patient's or merged patients' data."""
    # Handle both single patient and merged patients cases
    if isinstance(patient_id, list):
        # For merged patients, check sleep data for all patients
        patient_ids = [f"Epat{num}" for num in patient_id]
        patient_sleep = sleep_data[sleep_data['PatID'].isin(patient_ids)]
        if len(patient_sleep) == 0:
            print(f"No sleep events found for any patients in {patient_ids}")
            return
        print(f"\nFound {len(patient_sleep)} total sleep events for merged patients")
    else:
        # Single patient case
        patient_sleep = sleep_data[sleep_data['PatID'] == f"Epat{patient_id}"]
        if len(patient_sleep) == 0:
            print(f"No sleep events found for patient Epat{patient_id}")
            return
        print(f"\nFound {len(patient_sleep)} sleep events for Epat{patient_id}")
    
    # Setup output directory and check for existing data
    output_dir, _ = setup_output_directory(patient_id)
    
    # Determine filename based on single vs merged
    if isinstance(patient_id, list):
        merged_name = "_".join([f"Epat{num}" for num in patient_id])
        tagged_data_path = os.path.join(output_dir, 
            f'tagged_sleep_stages_{merged_name}_certainty{certainty_threshold:.2f}.pkl')
    else:
        tagged_data_path = os.path.join(output_dir, 
            f'tagged_sleep_stages_Epat{patient_id}_certainty{certainty_threshold:.2f}.pkl')
    
    # Check for existing data and process
    if os.path.exists(tagged_data_path) and not force_reprocess:
        print(f"Found existing tagged data at {tagged_data_path}")
        with open(tagged_data_path, 'rb') as f:
            data = pickle.load(f)
            
        # Handle both old and new data formats
        if 'sleep_stages' not in data:
            print("Found data in old format. Reprocessing...")
            tagged_data_path, plot_path = tag_points(patient_id, sleep_data, certainty_threshold)
        else:
            # Create visualization with appropriate name
            display_name = merged_name if isinstance(patient_id, list) else f"Epat{patient_id}"
            plot_path = create_visualization(
                data['transformed_points_2d'],
                data['sleep_stages'],
                display_name,
                output_dir,
                f'tagged_pointcloud_{display_name}.png'
            )
    else:
        print(f"Running tagging process...")
        tagged_data_path, plot_path = tag_points(patient_id, sleep_data, certainty_threshold)
    
    print(f"Processing complete! Files saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Tag brain state embeddings with sleep stage metadata.')
    parser.add_argument('--patient_id', type=int, nargs='+',
                      help='One or more Patient IDs as integers (e.g., 37 38) or "all" to process all patients')
    parser.add_argument('--metadata', type=str, 
                      default='metadata/cleaned_sleep.xlsx',
                      help='Path to Excel file containing sleep metadata')
    parser.add_argument('--force', action='store_true',
                      help='Force reprocessing of existing tagged data')
    parser.add_argument('--certainty', type=float, default=0.6,
                      help='Minimum certainty threshold for including sleep stages')
    
    args = parser.parse_args()
    
    try:
        # Load sleep metadata first
        sleep_data = load_sleep_metadata(args.metadata)
        
        if len(args.patient_id) == 1 and str(args.patient_id[0]).lower() == 'all':
            # Look for patient directories in output
            if not os.path.exists('output'):
                raise FileNotFoundError("output directory not found. Please run pickle_to_cloud.py first.")
            
            patient_dirs = [d for d in os.listdir('output') 
                          if os.path.isdir(os.path.join('output', d))]
            
            print(f"\nProcessing all {len(patient_dirs)} patients...")
            for patient_dir in patient_dirs:
                try:
                    # Extract patient number from directory name
                    patient_num = int(patient_dir.replace('Epat', ''))
                    print(f"\n=== Processing {patient_dir} ===")
                    process_patient(patient_num, sleep_data, args.certainty, args.force)
                except Exception as e:
                    print(f"Error processing {patient_dir}: {e}")
                    continue
        else:
            # Process either as individual patients or merged group
            if len(args.patient_id) > 1:
                print(f"\n=== Processing merged patients {', '.join(f'Epat{num}' for num in args.patient_id)} ===")
                process_patient(args.patient_id, sleep_data, args.certainty, args.force)
            else:
                patient_id = args.patient_id[0]
                print(f"\n=== Processing Epat{patient_id} ===")
                process_patient(patient_id, sleep_data, args.certainty, args.force)
            
    except Exception as e:
        print(f"\nError: {e}")
        exit(1)

if __name__ == "__main__":
    main()