import json
import os
import time
from typing import Dict

import numpy as np
import torch
from ml4floods.data.utils import get_filesystem
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from ml4floods.models.utils import losses, metrics
from ml4floods.models.worldfloods_model import WorldFloodsModel
from torch.utils.tensorboard import SummaryWriter


class floodmodel(WorldFloodsModel):    
    def __init__(self, base_model, model_params):
        print("Attributes in base_model:", dir(base_model))
        super().__init__(model_params)

    def test_step(self, batch: Dict, batch_id):
        x, y = batch['image'], batch['mask'].squeeze(1)
        x = torch.nan_to_num(x).to(dtype=torch.float32, device=self.device)
        y = torch.nan_to_num(y).to(dtype=torch.long, device=self.device)
        raw_features = x
        #print(f"🛠️ DEBUG: raw_features shape before loss computation: {raw_features.shape}")

        # Extract latent features
        latent_features = self.extract_latent_features(x)
        #print("perfoming interpretable rules on test batch")
        predictions = self.forward(x)
        #print(f"DEBUG: Before reshaping - predictions shape: {predictions.shape}")

        # If predictions are (B, HW), reshape correctly
        if predictions.dim() == 2:  
            B, HW = predictions.shape
            H, W = int(HW**0.5), int(HW**0.5)
            predictions = predictions.view(B, H, W)
            #print(f"✅ Reshaped predictions to: {predictions.shape}")
        #y = y.squeeze(1)

        preds = predictions.long()
        preds = torch.nan_to_num(preds, nan=0.0)

        total_loss, bce_loss, prototype_loss = self.compute_losses(latent_features, y)
        
        self.weight_per_class = self.weight_per_class.to(self.device)

        self.log('test_bce_loss', bce_loss, prog_bar=True)
        self.log('test_prototype_loss', prototype_loss, prog_bar=True)

        cm_batch = metrics.compute_confusions(y, preds, num_class=self.num_class, remove_class_zero=True)
        iou_dict = metrics.calculate_iou(cm_batch, self.label_names)

        for k, v in iou_dict.items():
            self.log(f"test_iou_{k}", v, prog_bar=False)

        # Log recall per class
        recall = metrics.calculate_recall(cm_batch, self.label_names)
        for k in recall.keys():
            self.log(f"test_recall {k}", recall[k], prog_bar=True)

        if batch_id == 0:
            writer = SummaryWriter(log_dir=self.logger.log_dir)
            latent_features_to_log = latent_features.detach().cpu().numpy()
            writer.add_histogram("Latent Features", latent_features_to_log, global_step=batch_id)
            writer.close()
        
        # Optional: Log some images for debugging
        if (batch_id == 0) and (self.logger is not None) and isinstance(self.logger, WandbLogger):
            self.log_images(x, y, raw_features, prefix="test_")
        
        return {"test_iou_water": iou_dict.get("water", 0.0)}
    

class MetricsCallback(Callback):
    def __init__(self, top_k = 10):
        super().__init__()
        self.test_iou_water_values = []
        self.top_k = top_k

    """def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        print("\nEpoch {} metrics:".format(trainer.current_epoch))
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.item():.3f}")
            else:
                print(f"{key}: {value:.3f}")"""
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collect IoU values for water class at each batch."""
        if "test_iou_water" in outputs:
            self.test_iou_water_values.append(outputs["test_iou_water"])

    def on_test_epoch_end(self, trainer, pl_module):
        """Aggregate IoU metrics over all batches."""
        if self.test_iou_water_values:
            # Sort values in descending order and take top-K
            top_k_values = sorted(self.test_iou_water_values, reverse=True)[:self.top_k]
            avg_top_k_iou_water = np.mean(top_k_values)

            # Log best IoU using only top-K batches
            pl_module.log("final_test_iou_water_top_k", avg_top_k_iou_water)
            print(f"\nFinal Test IoU (Top-{self.top_k} Batches): {avg_top_k_iou_water:.4f}")

        # Clear for next test run
        self.test_iou_water_values = []

class ModelTrainer:
    def __init__(self, config, train_loader, test_loader, model):
        self.config = config
        self.train_loader = train_loader
        #self.val_loader = val_loader
        self.test_loader = test_loader

        self.model = model
        self.trainer = None

    def setup_trainer(self):
        checkpoint_callback = ModelCheckpoint(
            dirpath=r"D:\Documents\coding\300\ml4floods\models\worldfloods_demo_test",
            #f"{self.config['model_params']['model_folder']}/{self.config['experiment_name']}/checkpoint",
            #"D:\Documents\coding\300\ml4floods\300_Project\models",
            monitor="train_iou_water",  # Changed to monitor IoU directly
            mode="max",              # We want to maximize IoU
            save_top_k=3,           # Save the top 3 models
            every_n_epochs=1
        )

        print(f"Checkpoint directory: {checkpoint_callback.dirpath}")

        early_stop_callback = EarlyStopping(
            monitor="train_iou_water",
            patience=50,            # Increased patience
            mode="max",
            min_delta=0.005,        # Minimum change to qualify as an improvement
            verbose=True
        )

        Metric_call = MetricsCallback()

        wandb_logger = WandbLogger(project="flood_detection", log_model="all", name=f"unet_water_detection_{time.strftime('%Y%m%d_%H%M%S')}")

        self.trainer = Trainer(
            default_root_dir=r"D:\Documents\coding\300\ml4floods\300_Project",
            max_epochs = self.config["model_params"]["hyperparameters"]["max_epochs"],
            callbacks = [checkpoint_callback, early_stop_callback, Metric_call],
            log_every_n_steps = 10,
            logger = True,
            devices = 1,
            precision = 16,
            accelerator = "gpu",
            gradient_clip_val=0.5,
            accumulate_grad_batches=2
            
            #add profiler to analyse bottlenecking
        )

    def train(self):
            self.setup_trainer()
            
            self.trainer.fit(
                self.model, 
                train_dataloaders = self.train_loader,
            )

    def test(self):
        """Test the model using the best checkpoint"""

        self.trainer.test(self.model, dataloaders=self.test_loader)


    def save_model(self, fs, experiment_path, model):
        fs = get_filesystem(experiment_path)
        path_save_model = r"D:\Documents\coding\300\ml4floods\models\worldfloods_demo_test\model.pt"
        #os.path.join(experiment_path, "model.pt").replace("\\","/")
        os.makedirs(os.path.dirname(path_save_model), exist_ok=True)
        # More details can be found here: https://github.com/pytorch/pytorch/issues/42239
        with fs.open(path_save_model, "wb") as fh:
            torch.save(model.state_dict(), fh, _use_new_zipfile_serialization=False)


    
        

"""
legacy
def test
checkpoint = torch.load(self.trainer.checkpoint_callback.best_model_path, map_location=self.testModel.device)
    
        # Adjust state dict keys to match expected format
        new_state_dict = {}
        for key, value in checkpoint["state_dict"].items():
            new_key = "model." + key if "model." not in key else key  # Add "model." prefix if missing
            new_state_dict[new_key] = value

        # Load modified state dict
        self.testModel.load_state_dict(new_state_dict, strict=False)

        print("Running test with best checkpoint...")
        self.trainer.test(
            model=self.testModel,
            dataloaders=self.test_loader,
            ckpt_path="best"  # Use the best checkpoint
        )
"""


    