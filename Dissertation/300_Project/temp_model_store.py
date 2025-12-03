import torch
import numpy as np
import pytorch_lightning as pl
from typing import List, Optional, Dict, Tuple
from ml4floods.preprocess.worldfloods import normalize
from ml4floods.models.config_setup import  AttrDict

from ml4floods.models.utils import losses, metrics
from ml4floods.models.architectures.baselines import SimpleLinear, SimpleCNN
from ml4floods.models.architectures.unets import UNet, UNet_dropout
from ml4floods.models.architectures.hrnet_seg import HighResolutionNet
from ml4floods.data.worldfloods.configs import COLORS_WORLDFLOODS, CHANNELS_CONFIGURATIONS, BANDS_S2, COLORS_WORLDFLOODS_INVLANDWATER, COLORS_WORLDFLOODS_INVCLEARCLOUD
from pytorch_lightning.loggers import WandbLogger
from ml4floods.data.utils import get_filesystem
from sklearn.cluster import MiniBatchKMeans

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

class WorldFloodsModel(pl.LightningModule):
    """
    Model to do multiclass classification.
    It expects ground truths y (B, H, W) tensors to be encoded as: {0: invalid, 1: clear, 2:water, 3: cloud}
    The preds (model.forward(x)) will produce a tensor with shape (B, 3, H, W)
    """
    def __init__(self, model_params: AttrDict, normalized_data:bool=True):
        super().__init__()
        self.save_hyperparameters()
        h_params_dict = model_params.get('hyperparameters', {})
        self.num_class = h_params_dict.get('num_classes', 3)
        self.network = configure_architecture(h_params_dict)
        self.weight_per_class = torch.tensor(h_params_dict.get('weight_per_class',[1.0 for _ in range(self.num_class)]),
                                            dtype=torch.float32,
                                            ).to(self.device)
        
        """class_counts = torch.tensor([85636729, 725670728, 44626862, 97745553], dtype=torch.float32)
        class_weights = 1.0 / class_counts
        class_weights /= class_weights.sum()  # Normalize
        self.weight_per_class = class_weights.to(self.device)
        print(f"class weights: {self.weight_per_class}")"""
        self.normalized_data = normalized_data
        self.num_prototypes = 200

        #changes
        #prototype layer initalisations
        # Add prototype layer
        # prototype vector is a representation of a class so the model can learn by comparing pixel to vector
        # act as clusters
        self.latent_dim = h_params_dict.get('latent_dim', 64)
        self.prototype_vectors = torch.nn.Parameter(
                torch.nan_to_num(
                    torch.randn(
                        self.num_class, self.num_prototypes, self.latent_dim) / np.sqrt(self.latent_dim)
            )  # Initialize randomly
        )
        torch.nn.init.xavier_uniform_(self.prototype_vectors)

        self.raw_dim = 13  # Sentinel-2 images have 13 spectral bands
        self.prototype_vectors_raw = torch.nn.Parameter(
            torch.nan_to_num(torch.randn(self.num_class, self.num_prototypes, self.raw_dim))
        )
        torch.nn.init.xavier_uniform_(self.prototype_vectors_raw)

        #print(f"✅ Model Initialized | Prototypes Shape: {self.prototype_vectors.shape}")
        self.rules = {}

        """self.weight_per_class = torch.Tensor(
            h_params_dict.get('weight_per_class', [1 for _ in range(self.num_class)]), 
            device=self.device
        )"""

        # learning rate params
        self.lr = h_params_dict.get('lr', 1e-4)
        self.lr_decay = h_params_dict.get('lr_decay', 0.5)
        self.lr_patience = h_params_dict.get('lr_patience', 2)

        self.train_epoch_metrics = {
            "train_total_loss": [],
            "train_bce_loss": [],
            "train_prototype_loss": [],
            "train_recall_land": [],
            "train_recall_water": [],
            "train_recall_cloud": [],
            "train_iou_land": [],
            "train_iou_water": [],
            "train_iou_cloud": []
        }
        
        #label names setup
        self.label_names = h_params_dict.get('label_names', [i for i in range(self.num_class)])


    def training_step(self, batch: Dict, batch_idx) -> float:
        """
        Args:
            batch: includes
                x (torch.Tensor): (B, C, W, H), input image
                y (torch.Tensor): (B, 1, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        x, y = batch['image'], batch['mask']
        x = torch.nan_to_num(x).to(dtype=torch.float32)
        y = torch.nan_to_num(y).to(dtype=torch.long)
        #print(f"Unique mask values seen in training step: {torch.unique(y)}")
        latent_features = self.extract_latent_features(x)
        raw_features = x
        #print(f"🛠️ DEBUG: raw_features shape before loss computation: {raw_features.shape}")
        #print(latent_features)

        # Re-cluster prototypes at the start of each epoch
        n = 50;
        if batch_idx % n == 0: # update every n batches
            self.update_prototypes(latent_features, raw_features, y)

        total_loss, bce_loss, prototype_loss = self.compute_losses(latent_features, y)

        predictions = self.classify(latent_features, raw_features)
        #print(f"Shape of predictions from classify(): {predictions.shape}")  # Debugging print

        y = y.squeeze(1)  # Ensure shape matches predictions
        pred_categorical = predictions.long()

        #print(f"Shape of pred_categorical before unpacking: {pred_categorical.shape}")
        B, HW = pred_categorical.shape
        H, W = int(HW ** 0.5), int(HW ** 0.5)  # Assuming square images
        pred_categorical = pred_categorical.view(B, H, W)

        cm_batch = metrics.compute_confusions(y, pred_categorical, num_class=self.num_class, remove_class_zero=True)
        recall = metrics.calculate_recall(cm_batch, self.label_names)
        iou_dict = metrics.calculate_iou(cm_batch, self.label_names)

        # Accumulate metrics over the epoch
        self.train_epoch_metrics["train_total_loss"].append(total_loss.detach().cpu().item())
        self.train_epoch_metrics["train_bce_loss"].append(bce_loss.detach().cpu().item())
        self.train_epoch_metrics["train_prototype_loss"].append(prototype_loss.detach().cpu().item())

        for k in recall.keys():
            self.train_epoch_metrics[f"train_recall_{k}"].append(recall[k])
        for k in iou_dict.keys():
            self.train_epoch_metrics[f"train_iou_{k}"].append(max(iou_dict[k], 0))
        #for k, v in iou_dict.items():
            #print(f"IoU calculation for class '{k}': {v}")

        return total_loss
    
    def extract_latent_features(self, x):
        logits = self.network.get_penultimate_layer(x)
        B, C, H, W = logits.shape
        #print(f"Penultimate logits shape: {logits.shape}")
        latent_features = logits.view(B, C, -1).permute(0, 2, 1)
        #print(f"Latent features shape after reshaping: {latent_features.shape}")
        latent_features = latent_features / torch.norm(latent_features, p=2, dim=-1, keepdim=True)
        latent_features = torch.nan_to_num(latent_features, nan=0.0).to(dtype=torch.float32)
        return latent_features

    def update_prototypes(self, latent_features, raw_features, y, momentum=0.2, min_samples=10):
        """
        update prototypesd useing MiniBatchKMeans per class

        """
        # flatten y to match latent_features
        B, HW, latent_dim = latent_features.shape
        #raw_dim = raw_features.shape[-1]

        #prototypes = self.prototype_vectors
        y = y.view(B, -1) # [b, hw]
        latent_features = latent_features.contiguous().view(B * HW, latent_dim)
        latent_features = torch.nn.functional.normalize(latent_features, p=2, dim=-1)
        latent_features = torch.nan_to_num(latent_features, nan=0.0)
        raw_features = raw_features.view(B * HW, -1)
        y = y.view(-1) # [b*hw]

        updated_prototypes  = self.prototype_vectors.clone()

        """if not hasattr(self, "prototype_vectors_raw") or self.prototype_vectors_raw.shape[-1] != raw_dim:
            self.prototype_vectors_raw = torch.zeros((self.num_class, self.num_prototypes, raw_dim), device=self.device)"""

        for class_id in range(self.num_class):
            #select features for class
            class_mask = y == class_id
            latent_class_features = latent_features[class_mask]
            latent_class_features_np = latent_class_features.detach().cpu().numpy()
            raw_class_features = raw_features[class_mask]
            
            #print(f"Class {class_id}: {latent_class_features.shape[0]} latent samples")  # Debugging print
            #print(f"Class {class_id}: {raw_class_features.shape[0]} raw samples")  # Debugging print

            num_valid_samples = latent_class_features.size(0)
            if num_valid_samples < min_samples:
                print(f"Skipping update for class {class_id}: Too few samples ({num_valid_samples}).")
                continue 

            if latent_class_features.size(0) == 0:
                continue

            # Get valid class counts

            total_count = sum((y == i).sum().item() for i in range(1, self.num_class)) + 1e-6
            #land_ratio, water_ratio, cloud_ratio = [(y == i).sum().item() / total_count for i in [1, 2, 3]]

            # Allocate prototypes with **higher water & cloud priority**
            # Adjust prototype allocation dynamically
            land_ratio = (y == 1).sum().item() / total_count
            water_ratio = (y == 2).sum().item() / total_count
            cloud_ratio = (y == 3).sum().item() / total_count

            # Ensure minimum water prototypes
            land_clusters = int(self.num_prototypes * 0.25)
            water_clusters = max(int(self.num_prototypes * (0.60 + 0.20 * water_ratio)), 100)
            cloud_clusters = self.num_prototypes - (land_clusters + water_clusters)
            
            excess = (land_clusters + water_clusters + cloud_clusters) - self.num_prototypes
            if excess > 0:
                cloud_clusters -= excess
            # Assign dynamically calculated clusters
            num_clusters = {
                1: land_clusters,
                2: water_clusters,
                3: cloud_clusters
            }.get(class_id, self.num_prototypes)  # Default to num_prototypes if class is unknown

            #print(f"🔹 Class {class_id} | Assigned Clusters: {num_clusters} (Land: {land_clusters}, Water: {water_clusters}, Cloud: {cloud_clusters})")

            if num_clusters < 2:
                continue
            
            assert not torch.isnan(latent_class_features).any(), f"NaN detected in latent features for class {class_id}"

            latent_class_features = torch.nan_to_num(latent_class_features, nan=0.0)

            print(f"🔹 Class {class_id} | Num Clusters: {num_clusters}")

            k = MiniBatchKMeans(n_clusters = num_clusters, batch_size=128, max_iter=300, init="k-means++")
            try:
                #store cluster centers
                # Dynamically adjust prototype size
                k.fit(latent_class_features_np)
                cluster_centers = torch.tensor(k.cluster_centers_, device=latent_features.device)
                cluster_centers_raw = raw_class_features.mean(dim=0)

                num_actual_clusters = cluster_centers.shape[0]  # Sometimes smaller than num_clusters

                # Scaling factor to prevent small classes from being overwritten aggressively
                scaling_factor = num_valid_samples / HW  # Ratio of valid pixels to total pixels

                # Apply momentum-based updates
                updated_prototypes [class_id, :num_actual_clusters] = (
                    (momentum * self.prototype_vectors[class_id, :num_actual_clusters]) +
                    ((1 - momentum) * cluster_centers * scaling_factor)
                )

                updated_prototypes = torch.nn.functional.normalize(updated_prototypes, p=2, dim=-1)

                self.prototype_vectors_raw.data[class_id, :num_actual_clusters] = (
                    momentum * self.prototype_vectors_raw[class_id, :num_actual_clusters] +
                    (1 - momentum) * cluster_centers_raw.expand(num_actual_clusters, -1)
                )

                # Compute inter-cluster variance
                water_variance = torch.var(self.prototype_vectors[2], dim=0).mean()
                max_distance = torch.norm(self.prototype_vectors[2] - self.prototype_vectors[2].mean(dim=0), dim=-1).max()

                # Adaptive scaling based on variance
                if class_id == 2:  # Water class
                    max_distance = torch.norm(
                        self.prototype_vectors[2] - self.prototype_vectors[2].mean(dim=0), dim=-1
                    ).max()
                    if max_distance > 0.7:  
                        print(f"⚠️ Scaling water prototypes down (variance={water_variance:.3f})")
                        scale_factor = 0.90 if water_variance > 0.5 else 0.95
                        self.prototype_vectors.data[2] *= scale_factor  # Scale dynamically
                    else:
                        print(f"✅ Water prototypes are well-distributed (max_distance={max_distance:.3f}), no scaling applied.")            
                print(f"Distance between water prototypes: {torch.norm(self.prototype_vectors[2] - self.prototype_vectors[2].mean(dim=0), dim=-1).mean()}")
            except ValueError as e:
                print(f"Skipping prototype update for class {class_id} due to clustering issue: {e}")

        self.prototype_vectors.data.copy_(torch.nan_to_num(updated_prototypes, nan=0.0))
            #print(f"Updated prototypes for class {class_id}: {prototypes[class_id]}")

    """=================Interpretable Funcitons==================="""

    def classify(self, latent_features, raw_features):
        """ use protoype distances and interpretable rules to class pixels """
        #print(f"📏 DEBUG: classify | raw_features shape BEFORE accessing bands: {raw_features.shape}")
        B, HW, latent_dim = latent_features.shape
        latent_features = latent_features.contiguous().reshape(B * HW, latent_dim)

        self.prototype_vectors = torch.nn.Parameter(torch.nan_to_num(self.prototype_vectors, nan=0.0).to(dtype=torch.float32))

        distances = torch.cdist(
            torch.nan_to_num(latent_features, nan=0.0).to(dtype=torch.float32), 
            torch.nan_to_num(self.prototype_vectors.view(-1, latent_dim), nan=0.0)
        )
        distances = distances.view(B, HW, self.num_class, self.num_prototypes)

        similarity_scores = torch.exp(-distances ** 2)
        similarity_scores /= similarity_scores.sum(dim=-2, keepdim=True)

        # Find the k-nearest prototypes for each pixel
        distances_flat = distances.view(B, HW, -1)
        _, nearest_prototypes = distances_flat.topk(k=10, dim=-1, largest=False)
        #print(f"\n🔍 DEBUG: nearest_prototypes Shape: {nearest_prototypes.shape}")

        max_valid_index = self.num_class * self.num_prototypes - 1
        assert nearest_prototypes.max().item() <= max_valid_index, "⚠️ Index out of bounds in nearest_prototypes!"
        assert nearest_prototypes.min().item() >= 0, "⚠️ nearest_prototypes contains negative indices!"

        class_votes = nearest_prototypes // self.num_prototypes  # Convert prototype index to class, Shape: [B, HW, 10]
        #print(f"\n🔍 DEBUG: class_votes Shape: {class_votes.shape}")
        proto_indices = nearest_prototypes % self.num_prototypes
        
        weighted_votes = torch.zeros((B, HW, self.num_class), device=latent_features.device)

        proto_indices_expanded = proto_indices.unsqueeze(2)
        #print(f"\n🔍 DEBUG: nearest_prototypes_expanded Shape: {nearest_prototypes_expanded.shape}")  # Expect [B, HW, num_classes, 10]
        for class_id in range(self.num_class):
            class_mask = (class_votes == class_id).float()  # Convert boolean mask to float, Shape: [B, HW, 10]
            #print(f"🔍 DEBUG: Class {class_id} | class_mask Shape: {class_mask.shape}")
            # Gather correct similarity scores for top-10 nearest prototypes
            gathered_similarities = torch.gather(
                similarity_scores[:, :, class_id, :],  # Shape: [B, HW, num_prototypes]
                -1,
                proto_indices_expanded.squeeze(2)  # Ensure shape matches similarity_scores
            )
            
            #print(f"🔍 DEBUG: Class {class_id} | gathered_similarities Shape: {gathered_similarities.shape}")
            
            vote_sum = (class_mask * gathered_similarities).sum(dim=-1)  # Sum across top 10, Shape: [B, HW]
            #print(f"🔍 DEBUG: Class {class_id} | vote_sum Shape: {vote_sum.shape}")
            
            weighted_votes[:, :, class_id] += vote_sum
        #print(f"\n🔍 DEBUG: Weighted Votes Shape: {weighted_votes.shape}")  # Expect [B, HW, num_classes]

        weighted_votes /= torch.clamp(weighted_votes.sum(dim=-1, keepdim=True), min=1e-6)

        predictions = torch.argmax(weighted_votes, dim=-1).reshape(B, HW)
        """if raw_features.shape[1] < 8:  # Expecting at least 8 bands (NIR = Band 7)
            print(f"❌ ERROR: raw_features has {raw_features.shape[1]} bands, expected at least 8 for NIR access!")"""
        
        # Compute NDWI (Normalized Difference Water Index)
        green_band = raw_features[:, 2, :, :].reshape(B, -1)  # Assume Band 2 is Green
        nir_band = raw_features[:, 7, :, :].reshape(B, -1)  # Assume Band 7 is NIR
        ndwi = (green_band - nir_band) / (green_band + nir_band + 1e-6)

        # Apply NDWI-based rule for water detection
        water_threshold = 0.1  # Adjust based on dataset
        water_mask = (ndwi > water_threshold) & (predictions != 2)
        
        weighted_votes[:, :, 2][water_mask] += 0.1  # Boost water probability
        predictions = torch.argmax(weighted_votes, dim=-1)  # Recompute final predictions

        #print(f"🔹 Class Distribution: {torch.bincount(predictions.view(-1))}")
        print(f"classify predictions shape: {predictions.shape}")
        return predictions

    def interpretable_rules(self, latent_features: torch.Tensor, raw_features: torch.Tensor) ->  torch.Tensor:
        """ if then decisions based on prototype based on prototype distances and stored rules"""
        #print(f"📏 DEBUG: interpretable_rules | raw_features shape BEFORE accessing bands: {raw_features.shape}")
        predictions = self.classify(latent_features, raw_features)
        B, HW, latent_dim = latent_features.shape
        predictions = predictions.argmax(dim=-1)
        print(f"📏 DEBUG: Unique values in raw predictions: {torch.unique(predictions)}")

        predictions = torch.clamp(predictions, min=0, max=self.num_class - 1)
        valid_classes = torch.tensor([0, 1, 2, 3], device=predictions.device)
        predictions = torch.where(torch.isin(predictions, valid_classes), predictions, torch.tensor(3, device=predictions.device))

        weighted_votes = torch.zeros((B, HW, self.num_class), device=latent_features.device)
        one_hot_preds = F.one_hot(predictions, num_classes=self.num_class).float()
        weighted_votes[:] = one_hot_preds
        #print(f"Predictions shape at {b},{i}: {predictions[b, i].shape}")  # Debugging print
        #class_id = predictions[b, i].item()
        #feature_values = latent_features[b, i, :].tolist()
        
        green_band = raw_features[:, 2, :, :].reshape(B, -1)  
        nir_band = raw_features[:, 7, :, :].reshape(B, -1)  
        swir_band = raw_features[:, 11, :, :].reshape(B, -1)  # Shortwave Infrared (SWIR)

        #print(f"📏 DEBUG: NIR Band Shape: {nir_band.shape}, Green Band Shape: {green_band.shape}")

        ndwi = (green_band - nir_band) / (green_band + nir_band + 1e-6)
        ndvi = (nir_band - green_band) / (nir_band + green_band + 1e-6)
        print(f"📏 DEBUG: weighted_votes shape before boosting: {weighted_votes.shape}, num_class: {self.num_class}")
        
        if weighted_votes.shape[2] > 2:
            water_boost = (ndwi > 0.01).float() * 0.15  
            weighted_votes[:, :, 2] += water_boost  # Boost water class
        
        if weighted_votes.shape[2] > 3:
            cloud_boost = (swir_band > 0.2).float() * 0.1  
            weighted_votes[:, :, 3] += cloud_boost  # Boost cloud class

        for class_id, rules in self.rules.items():
            mask = predictions == class_id
            feature_vals = latent_features[mask]  # Extract only affected pixels

            # Apply stored rules for the given class
            if self.apply_rule(class_id, feature_vals.tolist()):
                predictions[mask] = class_id

        print(f"📏 DEBUG: classify() output shape: {predictions.shape}")
        return predictions
    
    """
    predictions = self.classify(latent_features, raw_features)
        B, HW, latent_dim = latent_features.shape
        predictions = predictions.argmax(dim=-1)

        for b in range(B):
            for i in range(HW):
                #print(f"Predictions shape at {b},{i}: {predictions[b, i].shape}")  # Debugging print
                class_id = predictions[b, i].item()
                feature_values = latent_features[b, i, :].tolist()

                green_band = raw_features[:, 2, :, :].reshape(B, -1)  
                nir_band = raw_features[:, 7, :, :].reshape(B, -1)  
                swir_band = raw_features[:, 11, :, :].reshape(B, -1)  # Shortwave Infrared (SWIR)

                ndwi = (green_band - nir_band) / (green_band + nir_band + 1e-6)
                ndvi = (nir_band - green_band) / (nir_band + green_band + 1e-6)

                # Lower NDWI threshold to **capture more water pixels**
                ndwi_threshold = 0.01 if class_id == 2 else 0.05  

                if ndwi > ndwi_threshold and class_id != 2:  
                    print(f"🌊 Overriding class at {b}, {i} to Water (NDWI: {ndwi:.2f})")
                    predictions[b, i] = 2  # Force Water classification

                # 🌥 **SWIR-based cloud detection**
                if swir_band > 0.2:
                    print(f"☁️ Overriding {b}, {i} to Cloud (SWIR: {swir_band:.2f})")
                    predictions[b, i] = 3  # Force Cloud classification

                # Apply rule-based classification
                if class_id in self.rules and self.apply_rule(class_id, feature_values):
                    predictions[b, i] = class_id

        return predictions"""

    def compute_losses(self, latent_features: torch.Tensor, labels: torch.Tensor):
        """
        Compute BCE loss and prototype loss
        """
        B, HW, latent_dim = latent_features.shape
        H, W = int(HW**0.5), int(HW**0.5)

        latent_features = torch.nan_to_num(latent_features, nan=0.0).to(dtype=torch.float32)
        self.prototype_vectors.data.copy_(torch.nan_to_num(self.prototype_vectors, nan=0.0).to(dtype=torch.float32))

        prototypes = self.prototype_vectors.view(-1, latent_dim)

        distances = torch.cdist(latent_features.reshape(-1, latent_dim), prototypes, p=2)
        distances = distances.view(B, HW, self.num_class, self.num_prototypes)
        distances = torch.clamp(distances, min=1e-6, max=100).to(dtype=torch.float32)

        labels = torch.clamp(labels, min=0, max=self.num_class - 1)

        valid_mask = (labels != 0)# Assuming invalid pixels are labeled as 0
        labels_non_invalid = torch.clamp((labels - 1) * valid_mask, min=0)
        #valid_mask = valid_mask.view(B, H, W) 
        #add some debugging
        #print(f"Latent Features Shape: {latent_features.shape}")  # Expected: (B, HW, latent_dim)
        #print(f"Distances Shape: {distances.shape}")

        total_loss, prototype_loss, bce_loss = 0.0, 0.0, 0.0                        

        for class_id in range(self.num_class):
            class_mask = (labels_non_invalid == class_id).float() * valid_mask
            #print(f"class mask Shape: {class_mask.shape}")
            
            class_distances = distances[:, :, class_id]
            if torch.isnan(class_distances).any():
                print(f"⚠️ NaN detected in `class_distances` for class {class_id} before logits computation.")
                class_distances = torch.nan_to_num(class_distances, nan=0.0)
            #print(f"class Distances Shape: {class_distances.shape}")

            class_mask_flat = class_mask.view(B, H, W)# Flatten to [B, HW]
            class_mask_flat = torch.where((class_mask_flat >= 0) & (class_mask_flat < self.num_class), 
                              class_mask_flat, 
                              torch.tensor(0, device=class_mask_flat.device)).long()

            # Clamp class distances for stability
            logits = -class_distances.mean(dim=-1).view(B, H, W)
            logits = logits.unsqueeze(1)
            logits = torch.clamp(logits, min=-50, max=50).to(dtype=torch.float32)
            logits = logits.expand(-1, self.num_class, -1, -1)

            if torch.isnan(logits).any():
                print(f"⚠️ NaN detected in logits for class {class_id} before BCE loss.")
                logits = torch.nan_to_num(logits, nan=0.0).to(dtype=torch.float32)

            #print(f"Unique values in class_mask_flat: {class_mask_flat.unique()}")
            #print(f"Expected number of classes: {self.num_class}")
            #print(f"logits shape: {logits.shape}, class_mask_flat shape: {class_mask_flat.shape}")

            class_bce_loss = F.cross_entropy(
                                logits, 
                                class_mask_flat, 
                                weight=self.weight_per_class.to(logits.device),
                                reduction='none'
                            ).unsqueeze(1)
            
            class_bce_loss *= valid_mask
            bce_loss += class_bce_loss.sum() / (valid_mask.sum() + 1e-6)

            # Compute prototype loss for the current class
            class_mask_flat = class_mask_flat.view(B, -1, 1)
            if class_mask.sum() > 0:
                prototype_loss += ((class_distances * class_mask_flat).sum() / (class_mask_flat.sum() + 1e-6))

        total_loss = bce_loss + prototype_loss
        assert torch.isfinite(total_loss), "Total loss contains NaN or Inf!"
        #print(f"Total loss: {total_loss}, BCE loss: {bce_loss}, Prototype loss: {prototype_loss}")
        return total_loss, bce_loss, prototype_loss


    def apply_rule(self, class_id: int, feature_vals: List[float]) -> bool:
        """ checks if featuer set matches stored rules for class """
        rules = self.rules.get(class_id, [])
        
        match_scores = []

        for rule in rules:
            num_conditions_met = sum(
                rule[idx]["min"] <= feature_vals[idx] <= rule[idx]["max"]
                for idx in range(len(rule))
        )
        if len(rule) > 0:  
            confidence = num_conditions_met / len(rule)  # % of conditions met
            match_scores.append(confidence)

        return max(match_scores) >= 0.8 if match_scores else False
    
    def add_rule(self,class_id: int, feature_ranges: List[Tuple[float, float]]):
        """ Automatically generates rules based on prototype variance """
        prototype_vectors = self.prototype_vectors[class_id]
        mean_proto = prototype_vectors.mean(dim=0)
        std_proto = prototype_vectors.std(dim=0)

        feature_ranges = [(mean_proto[i] - std_proto[i], mean_proto[i] + std_proto[i]) 
                          for i in range(len(mean_proto))]

        if class_id not in self.rules:
            self.rules[class_id] = []

        self.rules[class_id].append(
            [{"min": feature_range[0], "max": feature_range[1]} for feature_range in feature_ranges]
        )
            

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_channels, H, W) input tensor

        Returns:
            (B, 3, H, W) prediction of the network
        """
        #changes
        print(f"📏 fwd DEBUG: Model input shape: {x.shape}")
        latent_features = self.extract_latent_features(x)
        raw_features = x
        print(f"📏 fwd DEBUG: Shape after feature extraction (latent_features): {latent_features.shape}")  
        print(f"📏 fwd DEBUG: Shape of raw_features BEFORE classify: {raw_features.shape}")
        rule_based_predictions = self.interpretable_rules(latent_features, raw_features)
        print(f"📏 fwd DEBUG: Predictions shape: {rule_based_predictions.shape}")
        return rule_based_predictions
    

    def image_to_logger(self, x:torch.Tensor) -> Optional[np.ndarray]:
        return batch_to_unnorm_rgb(x,
                                   self.hparams["model_params"]["hyperparameters"]['channel_configuration'],
                                   unnormalize=self.normalized_data)


    def log_images(self, x, y, logits, prefix=""):
        import wandb
        mask_data = y.cpu().numpy()
        pred_data = torch.argmax(logits, dim=1).long().cpu().numpy()

        img_data = self.image_to_logger(x)

        if img_data is not None:
            self.logger.experiment.log(
                {f"{prefix}overlay": [self.wb_mask(img, pred, mask) for (img, pred, mask) in zip(img_data, pred_data, mask_data)]})
            self.logger.experiment.log({f"{prefix}image": [wandb.Image(img) for img in img_data]})

        self.logger.experiment.log({f"{prefix}y": [wandb.Image(mask_to_rgb(img)) for img in mask_data]})
        self.logger.experiment.log({f"{prefix}pred": [wandb.Image(mask_to_rgb(img + 1)) for img in pred_data]})

    def validation_step(self, batch: Dict, batch_idx):
        """
        Args:
            batch: includes
                x (torch.Tensor): (B, C, W, H), input image
                y (torch.Tensor): (B, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        x, y = batch['image'], batch['mask'].squeeze(1)
        logits = self.network(x)
        
        bce_loss = losses.cross_entropy_loss_mask_invalid(logits, y, weight=self.weight_per_class.to(self.device))
        dice_loss = losses.dice_loss_mask_invalid(logits, y)
        self.log('val_bce_loss', bce_loss)
        self.log('val_dice_loss', dice_loss)

        pred_categorical = torch.argmax(logits, dim=1).long()

        # cm_batch is (B, num_class, num_class)
        cm_batch = metrics.compute_confusions(y, pred_categorical, num_class=self.num_class,
                                              remove_class_zero=True)

        # Log accuracy per class
        recall = metrics.calculate_recall(cm_batch, self.label_names)
        for k in recall.keys():
            self.log(f"val_recall {k}", recall[k])

        # Log IoU per class
        iou_dict = metrics.calculate_iou(cm_batch, self.label_names)
        for k in iou_dict.keys():
            iou_value = iou_dict[k]
            iou_value = max(iou_value, 0) # ensure it isn't negative
            self.log(f"val_iou {k}", iou_dict[k])
            
        if (batch_idx == 0) and (self.logger is not None) and isinstance(self.logger, WandbLogger):
            self.log_images(x, y, logits,prefix="val_")

    def visualize_prototypes(self, epoch: int):
        """
        Visualize and log prototypes for interpretability.
        """
        # Ensure prototypes exist
        if not hasattr(self, "prototype_vectors") or self.prototype_vectors is None:
            print("No prototypes found to visualize.")
            return

        prototypes = self.prototype_vectors  # Shape: [num_classes * num_prototypes, latent_dim]
        num_classes = self.num_class
        num_prototypes_per_class = prototypes.shape[0] // num_classes

        fig, axs = plt.subplots(num_classes, num_prototypes_per_class, figsize=(15, 5))
        for class_id in range(num_classes):
            for proto_id in range(num_prototypes_per_class):
                prototype = prototypes[class_id * num_prototypes_per_class + proto_id].detach().cpu().numpy()

                ax = axs[class_id, proto_id] if num_prototypes_per_class > 1 else axs[class_id]
                ax.imshow(prototype, cmap="viridis")
                ax.set_title(f"Class {class_id}, Proto {proto_id}")
                ax.axis("off")

        plt.tight_layout()

        # Save the figure as an image
        writer = SummaryWriter(log_dir=self.logger.log_dir)
        writer.add_figure(f"Prototypes/Epoch_{epoch}", fig)
        writer.close()

        plt.close(fig)

        # Visualize KMeans cluster centers
        if hasattr(self, "kmeans") and self.kmeans is not None:
            cluster_centers = self.kmeans.cluster_centers_
            fig, ax = plt.subplots(figsize=(10, 6))

            for i, center in enumerate(cluster_centers):
                ax.plot(center, label=f"Cluster {i}")

            ax.set_title("KMeans Cluster Centers")
            ax.legend()
            plt.tight_layout()

            # Save KMeans visualizations
            writer.add_figure(f"KMeans/Epoch_{epoch}", fig)
            writer.close()
            plt.close(fig)

        else:
            print("No KMeans model found to visualize.")


    def visualize_clusters(self):
        num_features = self.prototype_vectors.shape[-1]

        fig, axes = plt.subplots(1, self.num_class, figsize=(15, 5))

        for class_id in range(self.num_class):
            prototypes = self.prototype_vectors[class_id].detach().cpu().numpy()
            
            # Compute PCA for visualization (reduce dimensionality)
            pca = PCA(n_components=2)
            reduced_prototypes = pca.fit_transform(prototypes)

            # Plot prototypes
            ax = axes[class_id]
            ax.scatter(reduced_prototypes[:, 0], reduced_prototypes[:, 1], alpha=0.7)
            ax.set_title(f"Class {class_id} Prototypes")
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")

        plt.suptitle("Prototype Clusters for Each Class")
        plt.show(block=False)
        plt.pause(0.001)

    def on_train_epoch(self):
        """
        Hook to visualize prototypes after each epoch.
        """
        super().on_train_epoch_end()
        super().on_train_epoch_start()
        self.visualize_prototypes(epoch=self.current_epoch)

    def on_train_epoch_start(self): 
        super().on_train_epoch_start()
        self.visualize_clusters()
        # Reset metric storage for the new epoch
        self.train_epoch_metrics = {
            "train_total_loss": [],
            "train_bce_loss": [],
            "train_prototype_loss": [],
            "train_recall_land": [],
            "train_recall_water": [],
            "train_recall_cloud": [],
            "train_iou_land": [],
            "train_iou_water": [],
            "train_iou_cloud": [],
        }

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        self.visualize_clusters()
        top_k = 20
        avg_metrics = {}
        for key, value in self.train_epoch_metrics.items():
            if len(value) > 0:
                # Sort values in descending order and take top-K
                top_k_values = sorted(value, reverse=True)[:top_k]
                avg_metrics[key] = np.mean(top_k_values)
            else:
                avg_metrics[key] = 0.0  # Default value if empty

        # Log the filtered best training metrics
        for metric_name, metric_value in avg_metrics.items():
            self.log(metric_name, metric_value, prog_bar=False)

        print("\nEpoch {} metrics:".format(self.current_epoch))
        for metric_name, metric_value in avg_metrics.items():
            print(f"{metric_name}: {metric_value:.3f}")
            

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode=METRIC_MODE[self.hparams["model_params"]["hyperparameters"]["metric_monitor"]],
                                                               factor=self.lr_decay, verbose=True,
                                                               patience=self.lr_patience)

        return {"optimizer": optimizer, "lr_scheduler": scheduler,
                "monitor": self.hparams["model_params"]["hyperparameters"]["metric_monitor"]}

    def wb_mask(self, bg_img, pred_mask, true_mask):
        import wandb
        return wandb.Image(bg_img, masks={
            "prediction" : {"mask_data" : pred_mask, "class_labels" : self.labels()},
            "ground truth" : {"mask_data" : true_mask, "class_labels" : self.labels()}})
        
    def labels(self):
        return {
            0: "invalid",
            1: "land",
            2: "water",
            3: "cloud"
        }


METRIC_MODE = {
    "val_dice_loss": "min",
    "val_iou water": "max",
    "val_bce_land_water": "min",
    "val_bce_loss": "min",
    "val_Acc_land_water": "max",
    "train_iou_water": "max"
}
