device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.Cellpose(gpu=True, model_type='nuclei')


class ImageDataset(Dataset):
    def __init__(self, file_paths, bit_depths=None):
        self.file_paths = file_paths
        self.bit_depths = bit_depths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        if self.bit_depths is not None:
            bit_depth = self.bit_depths[idx]
            max_pixel_value = float(2**bit_depth - 1)
            binned_image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_AREA)
            norm_image = binned_image / max_pixel_value
            tensor_image = torch.tensor(norm_image, dtype=torch.float32).unsqueeze(0)
        else:
            # If bit_depths are not provided, convert the image to a tensor
            if len(image.shape) == 2:  # If grayscale
                image = image[:, :, np.newaxis]
            image = image.transpose((2, 0, 1))  # Convert from HWC to CHW format required by PyTorch
            tensor_image = torch.tensor(image, dtype=torch.float32).div(255)  # Normalize to [0, 1]

        return tensor_image, file_path

def process_image_info(file_path, supported_formats):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in supported_formats:
        file_name = os.path.basename(file_path)
        parts = file_name.split('_')
        channel = parts[5] if len(parts) > 5 else 'unknown_channel'
        
        image = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
        if image.dtype == np.uint8:
            bit_depth = 8
        elif image.dtype == np.uint16:
            bit_depth = 16
        elif image.dtype == np.float32:
            bit_depth = 32
        else:
            print(f"Unsupported image data type for file: {file_name}")
            return None, None, None

        return channel, file_path, bit_depth
    return None, None, None

def segment_and_measure_intensity(preprocessed_loaders, coords_df, device, segmentation_channel='405'):
    results = []

    # Check if the segmentation channel exists
    if segmentation_channel not in preprocessed_loaders:
        print(f"Segmentation channel {segmentation_channel} not found in processed loaders.")
        return pd.DataFrame(results)  # Return empty DataFrame if segmentation channel is missing

    # Get masks from the segmentation channel
    segmentation_dataloader = preprocessed_loaders[segmentation_channel]
    
    for seg_images, seg_paths in segmentation_dataloader:
        seg_images = seg_images.to(device)

        for seg_idx, seg_image in enumerate(seg_images):
            seg_path = seg_paths[seg_idx]
            # Segment nuclei from the seg_image
            mask = segment_nuclei(seg_image)
            
            # Metadata extraction from the segmentation image path
            metadata = extract_metadata_from_filename(os.path.basename(seg_path))
            x, y, z = map_to_xyz(metadata['i'], metadata['j'], metadata['k'], coords_df)
                
            # Measure intensities in other channels based on the mask
            for channel, channel_dataloader in preprocessed_loaders.items():
                if channel == segmentation_channel:
                    continue  # Skip the segmentation channel

                # Extract the corresponding intensity image
                intensity_images, intensity_paths = next(iter(channel_dataloader))
                intensity_image = intensity_images[seg_idx].to(device)  # Get the corresponding intensity image for this segment
                intensity_path = intensity_paths[seg_idx]
                
                # Measure intensity
                intensity_values = measure_intensity(mask, intensity_image)  # Needs direct tensor processing
                
                for intensity in intensity_values:  # Assuming multiple intensities per field
                    results.append({
                        'h': metadata['h'],
                        'i': metadata['i'],
                        'j': metadata['j'],
                        'k': metadata['k'],
                        'channel': channel,
                        'intensity': intensity,
                        'x': x,
                        'y': y,
                        'z': z,
                        'image_name': os.path.basename(intensity_path) 
                    })

    return pd.DataFrame(results)

def segment_nuclei(image_tensor):
    """Segment nuclei in an image tensor using Cellpose."""
    # Convert PyTorch tensor to numpy array
    img_np = image_tensor.cpu().numpy()
    
    # If the tensor is in CHW format, convert it to HWC format expected by Cellpose
    if img_np.ndim == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
        
    # If the image has a channel dimension at the end (e.g., grayscale with a singleton dimension), squeeze it
    if img_np.shape[-1] == 1:
        img_np = np.squeeze(img_np, axis=-1)

    # Scale the image from 0-1 to 0-255 if necessary
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    
    # Segment using Cellpose
    masks, _, _, _ = model.eval(img_np, diameter=None, channels=[0, 0])

    return masks

def measure_intensity(mask, image_tensor):
    """Measure fluorescence intensity within each mask area individually."""
    # Ensure that the mask and image tensor are on CPU and convert them to numpy for processing
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        # Assume mask is already a numpy array
        mask_np = mask
    
    # Similarly, check and convert the image tensor
    if isinstance(image_tensor, torch.Tensor):
        image_np = image_tensor.cpu().numpy().squeeze()  # Squeeze to ensure we have 2D array
    else:
        image_np = image_tensor.squeeze()

    # Initialize a list to store mean intensity values for each unique region in the mask
    intensities = []

    # Iterate through each unique region in the mask (excluding 0, which represents the background)
    unique_masks = np.unique(mask_np)
    for unique_mask in unique_masks:
        if unique_mask == 0:  # Skip background
            continue

        # Calculate the mean intensity for pixels within the current mask region
        masked_intensity = image_np[mask_np == unique_mask]
        mean_intensity = np.mean(masked_intensity)
        intensities.append(mean_intensity)

    # Return a list of mean intensities for each masked region
    return intensities

def extract_metadata_from_filename(filename):
    # Assuming the filename pattern is: channel_h_i_j_k_Fluorescence_channel_nm_Ex.tiff
    # Split the filename by '_' and extract relevant parts
    parts = filename.split('_')
    h = parts[1]
    i = int(parts[2])
    j = int(parts[3])
    k = int(parts[4])
    channel = parts[0] 

    # Return the extracted metadata as a dictionary
    return {'h': h, 'i': i, 'j': j, 'k': k, 'channel': channel}

def map_to_xyz(i, j, k, coords_df):
    try:
        # Use the multi-index for fast lookup
        row = coords_df.loc[(i, j, k)]
        # Extract the x, y, z coordinates
        x, y, z = row['x (mm)'], row['y (mm)'], row['z (um)']
        return x, y, z
    except KeyError:
        # Handle the case where the (i, j, k) combination does not exist
        print(f"No matching coordinates found for indices: i={i}, j={j}, k={k}")
        return None, None, None

