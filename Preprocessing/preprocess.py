def apply_gaussian_smoothing(image, window_fraction):
    """
    What do:
        Apply Gaussian smoothing to an image tensor with a dynamically sized kernel
    Args:
        image (Tensor): The input image tensor. Can be 2D (H, W) for grayscale images or 3D (C, H, W) for multi-channel images
        window_fraction (float): Fraction of the image dimensions to determine the kernel size for Gaussian smoothing
    Returns:
        Tensor: The smoothed image tensor
    Raises:
        ValueError: If the input image tensor does not have 2 or 3 dimensions
    """
    # print("Applying Gaussian smoothing...")

    # Handle the dimensionality of the input image tensor to accommodate both grayscale and color images
    if image.dim() == 2:  # Grayscale image with dimensions (Height, Width)
        height, width = image.shape
    elif image.dim() == 3:  # Color image with dimensions (Channels, Height, Width)
        _, height, width = image.shape
    else:
        # Raise an error if the image tensor does not match expected dimensions
        raise ValueError("Image tensor has an unsupported number of dimensions")
    
    # Calculate the kernel size as a fraction of the image's dimensions, ensuring it is odd and at least 3 pixels
    window_size = int(min(height, width) * window_fraction)  # Determine kernel size based on the fraction of smaller dimension
    window_size = max(window_size + 1 if window_size % 2 == 0 else window_size, 3)  # Ensure kernel size is odd and at least 3

    # Initialize the GaussianBlur transformation with the calculated kernel size and sigma
    gaussian_blur = transforms.GaussianBlur(
        kernel_size=(window_size, window_size),  # Kernel size for the Gaussian blur
        sigma=(window_size / 6.0, window_size / 6.0)  # Sigma value calculated as a function of kernel size
    )

    # If the image is grayscale, add a channel dimension to make it compatible with GaussianBlur, which expects a 3D tensor
    if image.dim() == 2:
        image = image.unsqueeze(0)  # Adds a channel dimension without changing the data

    # Apply the Gaussian blur transformation to the image tensor
    # The unsqueeze(0) adds a batch dimension needed for processing, and squeeze(0) removes it afterward
    smoothed_image = gaussian_blur(image.unsqueeze(0)).squeeze(0)
    # print("Gaussian smoothing applied.")

    # Return the smoothed image tensor
    return smoothed_image

def calc_mean_image(dataloader):
    """
    What do:
        Calculate the mean image across an entire dataset
        This function iterates through a DataLoader object that provides batches of images,
        and calculates the mean image by summing all images and then dividing by the number
        of images in the dataset
    Args:
        dataloader (DataLoader): A PyTorch DataLoader object that provides batches of images
    Returns:
        Tensor: A tensor representing the mean image of the dataset
    """
    # Initialize a tensor of zeros with the same shape as the first batch of images.
    sum_images = torch.zeros(next(iter(dataloader))[0][0].shape).to(device)
    # Iterate over all batches of images provided by the dataloader.
    for images, _ in dataloader:
        # sum of all images seen so far
        sum_images += images.to(device).sum(axis=0)
    # calculate the mean image by dividing the sum_images tensor by number of images in the dataset
    mean_image = sum_images / len(dataloader.dataset)
    return mean_image

def compute_global_metrics_per_channel(all_file_paths, all_bit_depths, device='cuda'):
    global_metrics = {}

    for channel, paths in all_file_paths.items():
        print(f"Processing channel: {channel}, Number of Images: {len(paths)}")
        # Create a dataset and dataloader for the current channel
        channel_dataset = ImageDataset(paths, all_bit_depths[channel])
        channel_dataloader = DataLoader(channel_dataset, 
                                        batch_size=4, 
                                        shuffle=False, 
                                        num_workers=os.cpu_count(), 
                                        pin_memory=True)
        
        # Compute the mean image for the current channel
        mean_image = calc_mean_image(channel_dataloader)

        # Apply Gaussian smoothing to the mean image
        smoothed_mean_image = apply_gaussian_smoothing(mean_image, window_fraction=0.05)

        # Compute global percentiles for the current channel
        global_lower, global_upper = calc_global_percentiles(channel_dataloader)

        # Store smoothed mean image and global metrics for the current channel
        global_metrics[channel] = {
            'mean_image': smoothed_mean_image,
            'global_lower': global_lower,
            'global_upper': global_upper
        }

    return global_metrics

def correct_and_scale_images(images, correction_function, global_lower, global_upper):
    """
    What do:
        Correct and scale images using global percentile intensities
    Args:
        dataloader (DataLoader): DataLoader providing batches of images
        correction_function (Tensor): A tensor representing the correction function, typically the smoothed mean image
        global_lower (float): Lower percentile value used for scaling, derived from global dataset statistics
        global_upper (float): Upper percentile value used for scaling, derived from global dataset statistics
    Returns:
        list: A list of tensors representing corrected and scaled images
    """
    # Initialize an empty list to store the corrected and scaled images
    corrected_images = []
    
    # Temporarily disables gradient calculation to improve performance and reduce memory usage
    with torch.no_grad():
        # Iterate over all batches of images provided by the dataloader
        for image in images:
            # Transfer images to the designated device (CPU or GPU) for computation
            image = image.to(device)
                
            # Correct each image by dividing it by the correction function and then normalizing
            # by the mean of the correction function to maintain intensity scale.
            corrected = (image / correction_function) * torch.mean(correction_function)
            
            if torch.isnan(corrected).any() or torch.isinf(corrected).any():
                print("Warning: `corrected` contains nan or inf values")
            
            # Iterate over each corrected image in the batch.
            for image in corrected:
                # Scale the image intensities to be within 0 and 1 based on the global percentile values
                # This operation also ensures intensities outside this range are clipped appropriately
                scaled = torch.clip((image - global_lower) / (global_upper - global_lower), 0, 1)
                # print(f"scaled min: {scaled.min()}, scaled max: {scaled.max()}")

                if torch.isnan(scaled).any() or torch.isinf(scaled).any():
                    print("Warning: `scaled` contains nan or inf values")
                
                image_byte = (scaled * 255).to(torch.uint8)
                
                # Convert scaled image to an 8-bit unsigned integer format, suitable for image visualization and storage
                corrected_images.append(image_byte)    
    
    # Return the list of corrected and scaled images.
    return corrected_images

def calc_global_percentiles_exact(dataloader, min_percentile=0.1, max_percentile=99.9):
    """
    What do:
        Calculate global percentile values for intensity across all images in a dataset.
        Downside, very memory intensive.
    Args:
        dataloader (DataLoader): DataLoader providing batches of images from the dataset
        min_percentile (float): The lower percentile to calculate (value between 0 and 100)
        max_percentile (float): The upper percentile to calculate (value between 0 and 100)
    Returns:
        tuple: lower and upper global percentile intensity values as floats
    """
    # Initialize an empty tensor on the same device as the dataloader images to accumulate all intensity values.
    all_intensities = torch.tensor([], device=device)

    # Iterate over all batches of images in the dataloader.
    for images, _ in dataloader:
        # Ensure the images tensor is on the same device as all_intensities before flattening and concatenating.
        images_on_device = images.to(device)
        all_intensities = torch.cat((all_intensities, images_on_device.view(-1)), dim=0)

    # Calculate the specified lower and upper percentiles from the aggregated intensity values.
    # The percentiles are specified as a fraction (hence division by 100.0), and the calculation
    # is done on the device (GPU or CPU) where the data resides to leverage potential computational efficiency.
    lower_percentile, upper_percentile = torch.quantile(all_intensities, torch.tensor([min_percentile, max_percentile], device=device) / 100.0)

    # Convert the tensor percentile values to Python floats and return them.
    return lower_percentile.item(), upper_percentile.item()

def calc_global_percentiles(dataloader, min_percentile=0.1, max_percentile=99.9, device='cuda'):
    """
    Estimate global percentile values for intensity across all images in a dataset
    in a memory-efficient manner.
    """
    batch_percentiles_lower = []
    batch_percentiles_upper = []

    for images, _ in dataloader:
        images_on_device = images.to(device)
        batch_intensities = images_on_device.view(-1)
        lower, upper = torch.quantile(batch_intensities, torch.tensor([min_percentile, max_percentile], device=device) / 100.0)
        batch_percentiles_lower.append(lower.item())
        batch_percentiles_upper.append(upper.item())

    # Estimate global quantiles from batch quantiles
    global_lower = np.mean(batch_percentiles_lower)
    global_upper = np.mean(batch_percentiles_upper)

    return global_lower, global_upper

def save_processed_images_to_disk(preprocessed_images, save_directory, channel, file_paths):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    paths = []

    original_filenames = [os.path.basename(file_path) for file_path in file_paths]
    
    for img_tensor, original_filename in zip(preprocessed_images, original_filenames):
        # Ensure the tensor is on the CPU and convert it to a numpy array
        img_numpy = img_tensor.cpu().numpy()

        # If the tensor has a channel dimension (1, H, W), remove it to get (H, W) for OpenCV
        if img_numpy.ndim == 3 and img_numpy.shape[0] == 1:
            img_numpy = img_numpy.squeeze(0)

        # Convert numpy array to the right type (uint8) if necessary
        if img_numpy.dtype != 'uint8':
            img_numpy = (img_numpy * 255).astype('uint8')

        base_filename = os.path.splitext(original_filename)[0]
        new_filename = f"{channel}_{base_filename}_preprocessed.tiff"
        path = os.path.join(save_directory, new_filename)

        # Use OpenCV to save the image
        cv2.imwrite(path, img_numpy)
        paths.append(path)

    return paths

def preprocess_images(file_paths, bit_depths, metrics, device, save_directory, channel):
    # Create an ImageDataset instance for the current channel
    channel_dataset = ImageDataset(file_paths, bit_depths)
    channel_dataloader = DataLoader(channel_dataset, 
                                    batch_size=4, 
                                    shuffle=False, 
                                    num_workers=os.cpu_count(), 
                                    pin_memory=True)

    # Extract the global metrics for the current channel
    mean_image = metrics['mean_image']
    global_lower = metrics['global_lower']
    global_upper = metrics['global_upper']

    # Initialize a list to hold processed images
    preprocessed_images = []

    # Iterate over the DataLoader and apply corrections
    for images, _ in channel_dataloader:
        corrected_and_scaled_images = correct_and_scale_images(images, mean_image, global_lower, global_upper)
        preprocessed_images.extend(corrected_and_scaled_images)
    
    # Save processed images to disk and get their paths
    preprocessed_image_paths = save_processed_images_to_disk(preprocessed_images, save_directory, channel, file_paths)

    # Create a dataset from the saved image paths
    preprocessed_dataset = ImageDataset(preprocessed_image_paths)
    preprocessed_dataloader = DataLoader(preprocessed_dataset, 
                                         batch_size=4, 
                                         shuffle=False, 
                                         num_workers=os.cpu_count(), 
                                         pin_memory=True)
    
    # Return a DataLoader for the processed dataset    
    return preprocessed_dataloader