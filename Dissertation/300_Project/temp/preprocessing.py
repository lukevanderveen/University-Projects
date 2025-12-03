import os
import cv2
import rasterio
import numpy as np
import torch
from torchvision.transforms import functional as F
from ml4floods.models.config_setup import get_default_config
from ml4floods.models.model_setup import get_model
from model_trainer import floodmodel


class Preprocessor:
    def __init__(self, dataset_folder, config_path):
        self.config = get_default_config(config_path)
        self.dataset_folder = dataset_folder
        self.target_size = self.config["data_params"]["window_size"]
        self.normalize = self.config["data_params"]["train_transformation"]["normalize"]


    def validate_split(self, split):
        split_path = os.path.join(self.dataset_folder, split)
        s2_folder = os.path.join(split_path, "S2")
        gt_folder = os.path.join(split_path, "gt")

        if not os.path.exists(s2_folder):
            raise FileNotFoundError(f"Input folder not found for split '{split}': {s2_folder}")
        if not os.path.exists(gt_folder):
            raise FileNotFoundError(f"Target folder not found for split '{split}': {gt_folder}")
    
    def check_mask_values(self, mask_files):
        for mask_path in mask_files:
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
                unique_values = np.unique(mask)
                print(f"Mask {mask_path} has values: {unique_values}")

    # image loader 
    def load_image(self, path):
        with rasterio.open(path) as p:
            image = p.read()
        return image
    
    # image normaliser
    def normalise_image(self, image):
        if self.config['data_params']['train_transformation']['normalize']:
            return image / 10000.00
        return image
    
    def remap_mask(self, mask):
        mask[mask == 3] = 0 # if mask is 3 map it to land
        return mask
    
    
    def prepare_data(self, split, transform_config = None):
        self.validate_split(split)

        s2_folder = os.path.join(self.dataset_folder, split, "S2")
        mask_folder = os.path.join(self.dataset_folder, split, "gt")

        s2_files = sorted([os.path.join(s2_folder, f) for f in os.listdir(s2_folder) if f.endswith('.tif')])
        mask_files = sorted([os.path.join(mask_folder, f) for f in os.listdir(mask_folder) if f.endswith('.tif')])
        #self.check_mask_values(mask_files)

        # error if these are empty
        if len(s2_files) != len(mask_files):
            raise ValueError("Mismatch between input and target file counts")
        
        valid_files = []

        # scrape files and remove images less than 256 x 256
        for s2_file, mask_file in zip(s2_files, mask_files):
            with rasterio.open(s2_file) as f:
                height, width = f.height, f.width
                if height >= 256 and width >= 256:
                    valid_files.append((s2_file, mask_file))
                    

        if not valid_files:
            raise ValueError(f"No valid images found in {split} split with dimensions >= 256 × 256.")
        

        
        #seperate valid images and masks
        s2_files, mask_files = zip(*valid_files)
        # Example usage

        # Initialize the dataset
        dataset = FloodDataset(
            image_files=s2_files,
            mask_files=mask_files,
            normalise_function=self.normalise_image,
            remap_function=self.remap_mask,
            transform_config=transform_config,
            target_size=(256, 256),
            training=(split == "train")
        )

        return dataset
    

    def initialize_model(self):
        self.config.model_params.model_folder = "D:\Documents\coding\300\ml4floods\300_Project\models"#"models" 
        os.makedirs("models", exist_ok=True)
        self.config.model_params.hyperparameters = {
            'num_classes': 3,
            'num_channels': 13,
            'latent_dim': 64,
            'lr': 0.0003,  # Slightly higher learning rate
            'lr_decay': 0.1,  # More aggressive decay
            'lr_patience': 15,  # Wait longer before reducing LR
            'early_stopping_patience': 15,  # Give more epochs before stopping
            'max_epochs': 45,  # Train longer
            'model_type': "segnet", #"unet"
            'batch_size': 8,  # Smaller batch size for better generalization & makes it really slow 
            'metric_monitor': 'train_iou_water',
            'channel_configuration': 'all',  # Use all available channels
            'weight_per_class': [1.93445299, 36.60054169, 2.19400729],  #most recent water was 50 for unet, changed to 10 for seg Land, Water, Cloud weights,[1.93445299, 2.75, 2.19400729] origninal weghts: [1.93445299, 36.60054169, 2.19400729]
            'label_names': ['land', 'water', 'cloud'],
            'metrics_to_track': [
                'train_bce_loss',
                #'val_bce_loss',
                'train_iou_land',
                'train_iou_water',
                'train_iou_cloud',
                'train_loss'
                #'val_loss'
            ]
        } 
        self.config.model_params.test = False
        self.config.model_params.train = True
        
        print(self.config.model_params)
        base_model = get_model(self.config.model_params)
        model = floodmodel(base_model, self.config.model_params)
        return model
    

    
class FloodDataset:
    def __init__(self, image_paths, mask_paths, transform=None, target_size=(256, 256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_UNCHANGED)

        # Resize both image and mask to target size
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)  # Nearest for segmentation masks

        # Convert to torch tensor
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # CHW format
        mask = torch.tensor(mask, dtype=torch.long)  # Keep mask as integer values

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask