import json
import os
from typing import Counter
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from PIL import Image

from ml4floods.models.worldfloods_model import WorldFloodsModel
from model_trainer import floodmodel

def save_config(config, model_folder):
    os.makedirs(model_folder, exist_ok=True) # check the folder exists

    config_path = os.path.join(model_folder, "config.json") # def path to save the config

    # save the configuration as JSON
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_path}")
    return config_path

# Add this block before calling get_dataset
def inspect_raw_dataset(dataset_path):
    print("\nInspecting raw dataset before processing:")
    for data_type in ["train", "val", "test"]:
        img_folder = os.path.join(dataset_path, data_type, "S2")
        mask_folder = os.path.join(dataset_path, data_type, "gt")

        if not os.path.exists(img_folder) or not os.path.exists(mask_folder):
            print(f"{data_type} folder missing!")
            continue

        img_files = [f for f in os.listdir(img_folder) if f.endswith(".tif")]
        mask_files = [f for f in os.listdir(mask_folder) if f.endswith(".tif")]

        if img_files:
            img_path = os.path.join(img_folder, img_files[0])
            with rasterio.open(img_path) as img_src:
                print(f"{data_type} - Sample Image: {img_path}, Shape: ({img_src.count}, {img_src.height}, {img_src.width})")

        if mask_files:
            mask_path = os.path.join(mask_folder, mask_files[0])
            with rasterio.open(mask_path) as mask_src:
                print(f"{data_type} - Sample Mask: {mask_path}, Shape: ({mask_src.count}, {mask_src.height}, {mask_src.width})")


def inspect_processed_dataset(dataset):
    print("\nInspecting processed dataset after calling get_dataset:")

    # Train dataset
    if hasattr(dataset, "train_dataset"):
        print("Inspecting train dataset...")
        train_dataset = dataset.train_dataset
        for idx in range(len(train_dataset)):
            sample = train_dataset[idx]  # Access one sample
            if isinstance(sample, (tuple, list)):
                image, mask = sample
            elif isinstance(sample, dict):
                image = sample['image']
                mask = sample['mask']
            else:
                raise ValueError("Unexpected dataset structure.")

            print(f"Sample {idx + 1}:")
            print(f"  Image shape: {image.shape if isinstance(image, torch.Tensor) else type(image)}")
            print(f"  Mask shape: {mask.shape if isinstance(mask, torch.Tensor) else type(mask)}")
            break

    # Validation dataset
    if hasattr(dataset, "val_dataset"):
        print("Inspecting validation dataset...")
        val_dataset = dataset.val_dataset
        for idx in range(len(val_dataset)):
            sample = val_dataset[idx]
            if isinstance(sample, (tuple, list)):
                image, mask = sample
            elif isinstance(sample, dict):
                image = sample['image']
                mask = sample['mask']
            else:
                raise ValueError("Unexpected dataset structure.")

            print(f"Sample {idx + 1}:")
            print(f"  Image shape: {image.shape if isinstance(image, torch.Tensor) else type(image)}")
            print(f"  Mask shape: {mask.shape if isinstance(mask, torch.Tensor) else type(mask)}")
            break

    # Test dataset
    if hasattr(dataset, "test_dataset"):
        print("Inspecting test dataset...")
        test_dataset = dataset.test_dataset
        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]
            if isinstance(sample, (tuple, list)):
                image, mask = sample
            elif isinstance(sample, dict):
                image = sample['image']
                mask = sample['mask']
            else:
                raise ValueError("Unexpected dataset structure.")

            print(f"Sample {idx + 1}:")
            print(f"  Image shape: {image.shape if isinstance(image, torch.Tensor) else type(image)}")
            print(f"  Mask shape: {mask.shape if isinstance(mask, torch.Tensor) else type(mask)}")
            break

def crop_image_to_single_tile(input_path, output_folder, tile_size=(256, 256), prefix=""):
    """
    Crop the top-left 256x256 corner of an image and save it as a new file.

    Args:
        input_path (str): Path to the input image.
        output_folder (str): Directory to save the cropped image.
        tile_size (tuple): The target size (height, width) for the cropped image.
        prefix (str): Prefix for the output filenames.
    """
    os.makedirs(output_folder, exist_ok=True)

    with rasterio.open(input_path) as src:
        img_height, img_width = src.height, src.width
        tile_height, tile_width = tile_size

        # Skip if the image is smaller than the target size
        if img_height < tile_height or img_width < tile_width:
            print(f"Skipping {input_path} as it is smaller than {tile_size}.")
            return

        # Define the top-left 256x256 window
        window = Window(0, 0, tile_width, tile_height)
        transform = src.window_transform(window)
        meta = src.meta.copy()
        meta.update({
            "height": tile_height,
            "width": tile_width,
            "transform": transform
        })

        # Save the cropped tile
        output_path = os.path.join(output_folder, f"{prefix}_{os.path.basename(input_path)}")
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(src.read(window=window))
        print(f"Saved cropped image to {output_path}.")



def process_and_save_images(input_dir, output_dir, crop_size=(256, 256), data_type="", img_folder="S2", mask_folder="gt"):
    """
    Processes and crops images and masks to the desired size without adding prefixes.
    Args:
        input_dir (str): Input directory containing images and masks.
        output_dir (str): Output directory for cropped images and masks.
        crop_size (tuple): Desired crop size (height, width).
        data_type (str): Dataset type ('train', 'val', or 'test').
        img_folder (str): Subdirectory name for images.
        mask_folder (str): Subdirectory name for masks.
    """
    img_input_path = os.path.join(input_dir, data_type, img_folder)
    mask_input_path = os.path.join(input_dir, data_type, mask_folder)
    img_output_path = os.path.join(output_dir, data_type, img_folder)
    mask_output_path = os.path.join(output_dir, data_type, mask_folder)

    os.makedirs(img_output_path, exist_ok=True)
    os.makedirs(mask_output_path, exist_ok=True)

    img_files = [f for f in os.listdir(img_input_path) if f.endswith(".tif")]
    mask_files = [f for f in os.listdir(mask_input_path) if f.endswith(".tif")]

    for img_file in img_files:
        mask_file = img_file  # Use the same filename for masks
        img_path = os.path.join(img_input_path, img_file)
        mask_path = os.path.join(mask_input_path, mask_file)

        # Ensure corresponding mask exists
        if not os.path.exists(mask_path):
            print(f"Warning: Missing mask for {img_file}. Skipping...")
            continue

        # Load image and mask
        with rasterio.open(img_path) as img_src, rasterio.open(mask_path) as mask_src:
            if img_src.width < crop_size[1] or img_src.height < crop_size[0]:
                print(f"Skipping {img_file} as it is smaller than {crop_size}.")
                continue

            # Crop image and mask
            img_cropped = img_src.read(window=Window(0, 0, crop_size[1], crop_size[0]))
            mask_cropped = mask_src.read(window=Window(0, 0, crop_size[1], crop_size[0]))

            # Save cropped image
            img_meta = img_src.meta.copy()
            img_meta.update({"width": crop_size[1], "height": crop_size[0], "transform": img_src.window_transform(Window(0, 0, crop_size[1], crop_size[0]))})
            img_output_file = os.path.join(img_output_path, img_file)
            with rasterio.open(img_output_file, "w", **img_meta) as dst:
                dst.write(img_cropped)
            print(f"Saved cropped image to {img_output_file}.")

            # Save cropped mask
            mask_meta = mask_src.meta.copy()
            mask_meta.update({"width": crop_size[1], "height": crop_size[0], "transform": mask_src.window_transform(Window(0, 0, crop_size[1], crop_size[0]))})
            mask_output_file = os.path.join(mask_output_path, mask_file)
            with rasterio.open(mask_output_file, "w", **mask_meta) as dst:
                dst.write(mask_cropped)
            print(f"Saved cropped mask to {mask_output_file}.")


def find_unmatched_files(image_dir, mask_dir):
    """
    Combs through the directories to find any file that doesn't have a corresponding match.
    
    Args:
        image_dir (str): Path to the directory containing images (S2).
        mask_dir (str): Path to the directory containing masks (gt).
    
    Returns:
        unmatched_in_images (list): List of image files without a matching mask.
        unmatched_in_masks (list): List of mask files without a matching image.
    """
    # Get all files in the image and mask directories
    image_files = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.tif')}
    mask_files = {os.path.splitext(f)[0] for f in os.listdir(mask_dir) if f.endswith('.tif')}
    
    # Find unmatched files
    unmatched_in_images = image_files - mask_files
    unmatched_in_masks = mask_files - image_files
    
    return unmatched_in_images, unmatched_in_masks

def check_class_distribution(dataloader):
    class_counts = Counter()
    for batch in dataloader:
        _, mask = batch["image"], batch["mask"]
        unique, counts = torch.unique(mask, return_counts=True)
        for cls, count in zip(unique.cpu().numpy(), counts.cpu().numpy()):
            class_counts[cls] += count
    print("Class distribution:", class_counts)

def load_trained_model(model_path, model_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = floodmodel(None, model_params).to(device)  # Initialize the model
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()  # Set to evaluation mode (important for batch normalization/dropout layers)
    return model