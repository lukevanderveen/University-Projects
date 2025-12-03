import argparse
import glob
import cv2
import pkg_resources
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from ml4floods.models.config_setup import get_default_config
from ml4floods.models.dataset_setup import get_dataset
from ml4floods.models.utils.configuration import AttrDict
from preprocessing import Preprocessor
from model_trainer import *
from inference import process_inference
#from evaluation import *
from utils import *
import torch

 
if __name__ == "__main__":
    """##### PATHS ####"""
    dataset_path = r"D:\Documents\coding\300\ml4floods\worldfloods_v1_0_sample"
    config_path = pkg_resources.resource_filename("ml4floods","models/configurations/worldfloods_template.json")
    model_folder = "D:/Documents/coding/300/ml4floods/models"  # Path to save the trained model
    #resized_folder = r"D:\Documents\coding\300\ml4floods\rapid_testing_samples"
    big_folder_path = r"D:\Documents\coding\300\ml4floods\WORLDFLOODS\WORLDFLOODS"

    image_dir = r"D:\Documents\coding\300\ml4floods\WORLDFLOODS\WORLDFLOODS\train\S2"
    mask_dir = r"D:\Documents\coding\300\ml4floods\WORLDFLOODS\WORLDFLOODS\train\gt"

    # Find unmatched files
    unmatched_in_images, unmatched_in_masks = find_unmatched_files(image_dir, mask_dir)
    
    # Print unmatched files
    print("Unmatched files in S2 (images):")
    for file in unmatched_in_images:
        print(f"{file}.tif")
    
    print("\nUnmatched files in gt (masks):")
    for file in unmatched_in_masks:
        print(f"{file}.tif")


    #inspect_raw_dataset(dataset_path)
    config = get_default_config(config_path)

    #config changes for data
    config.experiment_name = '300 Flooding'   
    config.data_params.batch_size = 8 # control this depending on the space on your GPU!
    config.data_params.loader_type = 'local'
    config.data_params.path_to_splits = dataset_path # local folder to download the data
    config.data_params.train_test_split_file = None
    config.data_params.num_workers = 4
    config.data_params.persistent_workers = True
    config.data_params.train_transformation = AttrDict({
        "normalize": True,
        "random_flip": True,
        "random_brightness": True
    })
    config.data_params.val_transformation = AttrDict({
        "normalize": True,
        "random_flip": False,
        "random_brightness": False
    })
    config.data_params.test_transformation = AttrDict({
        "normalize": True,
        "random_flip": False,
        "random_brightness": False
    })

    # increase learning rate in inilisase model

    print(config)

    """##### STEP 1: PREPROCESSING ####"""

    preprocessor = Preprocessor(dataset_path, config_path)
    print("preprocessor initalised")

   
    dataset = get_dataset(config.data_params)
    inspect_processed_dataset(dataset)
    
    # Create DataLoaders
    train2_loader = dataset.train_dataloader()
    """for batch in train2_loader:
        unique_classes = torch.unique(batch['mask'])
        print(f"Unique mask values in dataset: {unique_classes}")
        break"""
    #check_class_distribution(train2_loader)
    val2_loader = dataset.val_dataloader()
    test2_loader = dataset.test_dataloader()
    print("data loaders initalised")

    # Inspect dataset shapes
    print("\nDataset shape inspection:")
    print("Inspecting the dataset directly:")

    model = preprocessor.initialize_model()

    print("======================================================")
    print("FINISHED PREPROCESSING")
    print("======================================================")

    """##### STEP 2: TRAINING ####"""

    # config setup
    config = preprocessor.config
    config['model_params']['model_folder'] = model_folder
    
    trainer = ModelTrainer(config, train2_loader, test2_loader, model)
    print("trainer initalized")


    print("training started")
    trainer.train()

    experiment_path = f"{config.model_params.model_folder}/{config.experiment_name}"
    fs = get_filesystem(experiment_path)
    trainer.save_model(fs, experiment_path, model)

    """##### STEP 3: INFERENCE AND EVALUATION ####"""
    print("======================================================")
    print("FINISHED TRAINING, SAVING MODEL")
    print("======================================================")

    print("Starting model testing...")
    # Test using the best checkpoint
    trainer.test()
    print("Test Results:")

    print("======================================================")
    print("FINISHED TESTING, STARTING EVALUATION")
    print("======================================================")

    #test_metrics = evaluate_model(model, test2_loader)
    #print_evaluation_results(test_metrics)

    # Save metrics if needed
    """with open(f"{config.model_params.model_folder}/{config.experiment_name}/test_metrics.txt", "w") as f:
        for metric_name, value in test_metrics.items():
            f.write(f"{metric_name}: {value:.4f}\n")


    channels = list(range(13)) 
    #inference = FloodInference(config, device="cuda", channels=channels) #consider all the channels
    
    colours = np.array([
        [0, 0, 0],  # Invalid
        [139, 64, 0],  # Land
        [0, 0, 240],  # Water
        [220, 220, 220],  # Cloud
        [60, 85, 92],  # Flood trace
    ]) / 255

    interpretation = ["Invalid", "Land", "Water", "Cloud", "Flood Trace"]"""

    #inference_image_dir = r"D:\Documents\coding\300\ml4floods\WORLDFLOODS\WORLDFLOODS\val\S2"
    #inference_mask_dir = r"D:\Documents\coding\300\ml4floods\WORLDFLOODS\WORLDFLOODS\val\gt"
    model_inference_checkpoint = r"D:\Documents\coding\300\ml4floods\models\worldfloods_demo_test\model.pt"

    inference_model = load_trained_model(model_inference_checkpoint, config.model_params)
    process_inference(val2_loader, inference_model)


    print("======================================================")
    print("FINISHED INFERENCE AND EVALUATION")
    print("======================================================")

    """
    LEGACY

    # Inspect dataset shapes
    print("\nDataset shape inspection:")
    print("Inspecting the dataset directly:")

    # Check if train_dataset, val_dataset, and test_dataset are available
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
            break  # Remove this `break` to inspect more samples if needed
    else:
        print("The dataset does not have a train_dataset attribute.")

    # Inspect val_dataset
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

    # Inspect test_dataset
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


    
    print("\nShape inspection:")
    print("Training data:")
    print("inspecting train set")
    batch = next(iter(train2_loader))

    # If batch is a tuple/list of (images, masks):
    images, masks = batch
    print("Images type:", type(images))
    print("Masks type:", type(masks))

    # Only try to print shape if it's a tensor
    for batch in train2_loader:
        if isinstance(batch, (tuple, list)):
            images, masks = batch
        elif isinstance(batch, dict):
            images = batch['image']
            masks = batch['mask']
        
        print(f"Images: {images.shape}")
        print(f"Masks: {masks.shape}")
        break

    print("\nValidation data:")
    for batch in val2_loader:
        if isinstance(batch, (tuple, list)):
            images, masks = batch
        elif isinstance(batch, dict):
            images = batch['image']
            masks = batch['mask']
        
        print(f"Images: {images.shape}")
        print(f"Masks: {masks.shape}")
        break

    print("\nTest data:")
    for batch in test2_loader:
        if isinstance(batch, (tuple, list)):
            images, masks = batch
        elif isinstance(batch, dict):
            images = batch['image']
            masks = batch['mask']
        
        print(f"Images: {images.shape}")
        print(f"Masks: {masks.shape}")
        break

    """