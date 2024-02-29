
import cv2
import gc  # Import Python's garbage collector
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import psutil
import pyclesperanto_prototype as cle
from skimage import img_as_uint, exposure, img_as_ubyte, img_as_float
from skimage.io import imread
from skimage.filters import median
from skimage.morphology import disk, ball, white_tophat
from scipy.ndimage import gaussian_filter
from skimage.restoration import rolling_ball
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms

memory = psutil.virtual_memory()
total_memory = memory.total / (1024 * 1024 * 1024)  # Convert bytes to GB
print(f"Total Memory: {total_memory:.2f} GB")
# Available memory
available_memory = memory.available / (1024 * 1024 * 1024)  # Convert bytes to GB
print(f"Available Memory: {available_memory:.2f} GB")

# Used memory
used_memory = memory.used / (1024 * 1024 * 1024)  # Convert bytes to GB
print(f"Used Memory: {used_memory:.2f} GB")

# Memory utilization percentage
memory_percent = memory.percent
print(f"Memory Utilization: {memory_percent}%")

# Check if CUDA (GPU support) is available
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)

# If CUDA is available, print the number and name of GPUs available
if cuda_available:
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs Available:", num_gpus)
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU available.")

def print_memory_usage(device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Current memory allocated: {torch.cuda.memory_allocated(device) / 1e9} GB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated(device) / 1e9} GB")
    print(f"Current memory cached: {torch.cuda.memory_reserved(device) / 1e9} GB")
    print(f"Max memory cached: {torch.cuda.max_memory_reserved(device) / 1e9} GB")

def clear_cuda_cache():
    # Trigger Python garbage collection
    gc.collect()
    # Clear PyTorch's CUDA cache
    torch.cuda.empty_cache()
    # After clearing, check memory usage again
    print_memory_usage()

print_memory_usage()  # Before clearing cache
# clear_cuda_cache()  # Clear cache and print memory usage after clearing

# Setup Distributed Environment
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

class ImageFinder:
    def __init__(self, folder_path):
        self.folder_path = folder_path
    
    def find_images(self, marker):
        """
        Find TIFF images containing the specified marker.
        """
        images = []
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith(".tif") and marker in file:
                    images.append(os.path.join(root, file))
        return images

class ImageDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load an image and convert it to a PyTorch tensor
        image = cv2.cvtColor(cv2.imread(self.file_paths[idx]), cv2.COLOR_BGR2RGB)
        # Bin the image by 2x2, effectively reducing its height and width by half
        binned_image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_AREA)
        # Convert the binned image to a PyTorch tensor, normalize to [0, 1]
        tensor_image = torch.tensor(binned_image, dtype=torch.float32) / 255
        # Reorder dimensions to CxHxW
        tensor_image = tensor_image.permute(2, 0, 1)
        return tensor_image

def calc_mean_image(dataloader, device):
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
    sum_images = torch.zeros(next(iter(dataloader))[0].shape).to(device)
    # Iterate over all batches of images provided by the dataloader.
    for images in tqdm(dataloader, desc='Calculating mean image'):
        # sum of all images seen so far
        sum_images += images.to(device).sum(axis=0)
    # calculate the mean image by dividing the sum_images tensor by number of images in the dataset
    mean_image = sum_images / len(dataloader.dataset)
    return mean_image

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
    print("Applying Gaussian smoothing...")
    
    # Handle the dimensionality of the input image tensor to accommodate both grayscale and color images
    if image.dim() == 2: # Grayscale image with dimensions (Height, Width)
        height, width = image.shape
    elif image.dim() == 3: # Color image with dimensions (Channels, Height, Width)
        _, height, width = image.shape
    else:
         # Raise an error if the image tensor does not match expected dimensions
        raise ValueError("Unsupported number of dimensions")

    # Calculate the kernel size as a fraction of the image's dimensions, ensuring it is odd and at least 3 pixels
    window_size = int(min(height, width) * window_fraction) # Determine kernel size based on the fraction of smaller dimension
    window_size = max(window_size + 1 if window_size % 2 == 0 else window_size, 3) # Ensure kernel size is odd and at least 3
    
    # Initialize the GaussianBlur transformation with the calculated kernel size and sigma
    gaussian_blur = transforms.GaussianBlur(
        kernel_size=(window_size, window_size), # Kernel size for the Gaussian blur
        sigma=(window_size / 6.0, window_size / 6.0)  # Sigma value calculated as a function of kernel size
        )

    # If the image is grayscale, add a channel dimension to make it compatible with GaussianBlur, which expects a 3D tensor
    if image.dim() == 2:
        image = image.unsqueeze(0) # Adds a channel dimension without changing the data

    # Apply the Gaussian blur transformation to the image tensor
    # The unsqueeze(0) adds a batch dimension needed for processing, and squeeze(0) removes it afterward
    smoothed_image = gaussian_blur(image.unsqueeze(0)).squeeze(0)
    return smoothed_image

def estimate_global_percentiles(dataloader, min_percentile=0.1, max_percentile=99.9):
    """
    Calculate global percentile values for intensity across all images in a dataset
    in a memory-efficient manner.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    percentile_values = []

    # Process each batch to calculate percentile values
    for images in tqdm(dataloader, desc='Gathering intensities'):
        images_on_device = images.to(device)
        # Flatten the images tensor and calculate percentiles for the current batch
        batch_percentiles = torch.quantile(images_on_device.view(-1), torch.tensor([min_percentile, max_percentile], device=device) / 100.0)
        percentile_values.append(batch_percentiles.cpu().numpy())  # Move results to CPU and convert to NumPy for efficiency

    # Convert list of percentile values to a NumPy array for further processing
    percentile_values = np.array(percentile_values)

    # Calculate the mean of the lower and upper percentiles across all batches
    global_lower = np.mean(percentile_values[:, 0])
    global_upper = np.mean(percentile_values[:, 1])

    return global_lower, global_upper

def calc_global_percentiles(dataloader, min_percentile=0.1, max_percentile=99.9):
    """
    What do:
        Calculate global percentile values for intensity across all images in a dataset
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
    for images in tqdm(dataloader, desc='Gathering intensities'):
        # Ensure the images tensor is on the same device as all_intensities before flattening and concatenating.
        images_on_device = images.to(device)
        all_intensities = torch.cat((all_intensities, images_on_device.view(-1)), dim=0)

    # Calculate the specified lower and upper percentiles from the aggregated intensity values.
    # The percentiles are specified as a fraction (hence division by 100.0), and the calculation
    # is done on the device (GPU or CPU) where the data resides to leverage potential computational efficiency.
    lower_percentile, upper_percentile = torch.quantile(all_intensities, torch.tensor([min_percentile, max_percentile], device=device) / 100.0)

    # Convert the tensor percentile values to Python floats and return them.
    return lower_percentile.item(), upper_percentile.item()


def correct_and_scale_images(dataloader, correction_function, global_lower, global_upper):
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
    
    print("Correcting and scaling images with global percentiles...")
    
    # Iterate over all batches of images provided by the dataloader
    for images in tqdm(dataloader, desc='Processing images'):
        # Transfer images to the designated device (CPU or GPU) for computation
        images = images.to(device)
        
        # Temporarily disables gradient calculation to improve performance and reduce memory usage
        with torch.no_grad():
            # Correct each image by dividing it by the correction function and then normalizing
            # by the mean of the correction function to maintain intensity scale.
            corrected = (images / correction_function) * torch.mean(correction_function)
            
            # Iterate over each corrected image in the batch.
            for image in corrected:
                # Scale the image intensities to be within 0 and 1 based on the global percentile values
                # This operation also ensures intensities outside this range are clipped appropriately
                scaled = torch.clip((image - global_lower) / (global_upper - global_lower), 0, 1)
                
                # Convert scaled image to an 8-bit unsigned integer format, suitable for image visualization and storage
                corrected_images.append((scaled * 255).byte())
    
    print("Correction and scaling completed")
    
    # Return the list of corrected and scaled images.
    return corrected_images

def main(rank, world_size):
    setup(rank, world_size)
    total_cpu_cores = os.cpu_count()
    cpu_cores_per_gpu = max(1, total_cpu_cores // world_size)  # Ensure at least 1 worker per GPU

    folder_path = '/home/jesseh/Jay/23_12_12_D21NPCD33_Image_Series/'
    
    image_finder = ImageFinder(folder_path)
    
    af488_image_list = image_finder.find_images("AF488")
    af555_images_list = image_finder.klfind_images("AF555")
    af647_image_list = image_finder.find_images("AF647")
    dapi_image_list = image_finder.find_images("DAPI")

    # Initialize custom dataset with the list of image file paths
    af488_dataset = ImageDataset(af488_image_list)

    # Initialize a DistributedSampler for the dataset. Ensures that data loading is optimized for distributed training
    af488_sampler = DistributedSampler(af488_dataset, # The dataset to sample from. Ensures each process gets a subset of the dataset
                                        num_replicas=world_size, # Total number of processes participating in the distributed training
                                        rank=rank, # The rank of the current process. Each process is assigned a unique rank within the distributed setup
                                        shuffle=False # Setting this to False ensures that the dataset's order is preserved across epochs
                                        )


    # Create a DataLoader from the dataset which provides batches of images for efficient processing
    # 'batch_size=' determines how many images are processed at once, reducing memory usage
    # 'num_workers=os.cpu_count()' leverages parallel processing, speeding up data loading
    af488_dataloader = DataLoader(af488_dataset, 
                                batch_size=4, 
                                sampler=af488_sampler, 
                                num_workers=cpu_cores_per_gpu, 
                                pin_memory=True # Enable pin_memory for faster data transfer to CUDA devices
                                )

    device = torch.device(f'cuda:{rank}')
    
    # Calculate global intensity percentiles from plate-level intensity histograms
    # This step finds the lower and upper bounds for pixel intensity normalization, ensuring consistent scaling
    # global_lower, global_upper = calc_global_percentiles(af488_dataloader)
    global_lower, global_upper = calc_global_percentiles(af488_dataloader)

    # Calculate the mean image across the dataset for normalization
    af488_mean_image = calc_mean_image(af488_dataloader, device)
    # Apply Gaussian smoothing to the mean image
    smoothed_af488_mean_image = apply_gaussian_smoothing(af488_mean_image, 0.25)
    # Correct and scale images based on global intensity percentiles and the smoothed mean image
    af488_corrected_images = correct_and_scale_images(af488_dataloader, smoothed_af488_mean_image, global_lower, global_upper)

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
