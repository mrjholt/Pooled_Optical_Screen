def scan_folder_for_images(folder_path, recursive=False):
    supported_formats = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    file_paths = {}
    bit_depths = {}

    if recursive:
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                # Skip files with "preprocessed" in their name
                if "preprocessed" in file_name:
                    continue
                full_path = os.path.join(root, file_name)
                channel, path, bit_depth = process_image_info(full_path, supported_formats)
                if channel:
                    if channel not in file_paths:
                        file_paths[channel] = []
                        bit_depths[channel] = []
                    file_paths[channel].append(path)
                    bit_depths[channel].append(bit_depth)
    else:
        for file_name in os.listdir(folder_path):
            # Skip files with "preprocessed" in their name
            if "preprocessed" in file_name:
                continue
            full_path = os.path.join(folder_path, file_name)
            channel, path, bit_depth = process_image_info(full_path, supported_formats)
            if channel:
                if channel not in file_paths:
                    file_paths[channel] = []
                    bit_depths[channel] = []
                file_paths[channel].append(path)
                bit_depths[channel].append(bit_depth)

    return file_paths, bit_depths

def process_experiment(parent_folder, device='cuda'):
    experiments = os.listdir(parent_folder)
    results_df = pd.DataFrame()
    preprocessed_directory = '/home/jesseh/p65/preprocessed'
    
    if not os.path.exists(preprocessed_directory):
        os.makedirs(preprocessed_directory, exist_ok=True)

    # Gather all image paths and bit depths for all image files, organized by number channel
    all_file_paths, all_bit_depths = scan_folder_for_images(parent_folder, recursive=True)

    # Compute global metrics for each channel
    global_metrics = compute_global_metrics_per_channel(all_file_paths, all_bit_depths, device)
    
    for experiment_name in tqdm(experiments, desc="Processing Experiments"):
        experiment_path = os.path.join(parent_folder, experiment_name)
        image_folder = os.path.join(experiment_path, '0')
        coords_path = os.path.join(image_folder, 'coordinates.csv')
        
        if os.path.isdir(image_folder) and os.path.exists(coords_path):
            coords_df = pd.read_csv(coords_path)
            coords_df.set_index(['i', 'j', 'k'], inplace=True)

            preprocessed_loaders = {}
            # For each channel in experiment, apply preprocessing with global metrics
            for channel, metrics in global_metrics.items():
                if channel in all_file_paths:  # Ensure channel is present in this experiment
                    # Preprocess images for this channel using global metrics
                    preprocessed_loader = preprocess_images(all_file_paths[channel], 
                                                            all_bit_depths[channel], 
                                                            metrics, 
                                                            device, 
                                                            preprocessed_directory, 
                                                            channel)
                    preprocessed_loaders[channel] = preprocessed_loader

            segmentation_channel = '405'
            condition_results = segment_and_measure_intensity(preprocessed_loaders, coords_df, device, segmentation_channel)
            condition_results['condition'] = "TNFa" if "+tnfa" in experiment_name.lower() else "Control"
            results_df = pd.concat([results_df, condition_results], ignore_index=True)
    
    return results_df
