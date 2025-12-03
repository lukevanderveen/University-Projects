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

    model_inference_checkpoint = r"D:\Documents\coding\300\ml4floods\models\worldfloods_demo_test\model.pt"

    inference_model = load_trained_model(model_inference_checkpoint, config.model_params)
    process_inference(val2_loader, inference_model)

    print("======================================================")
    print("FINISHED INFERENCE AND EVALUATION")
    print("======================================================")