import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from tensorflow.keras.metrics import MeanIoU
import torch
from typing import List, Dict, Any, Callable, Optional
from ml4floods.models.utils import losses, metrics 

np.set_printoptions(suppress=True)

def calculate_iou(confusions, labels):
    """
    Caculate IoU for a list of confusion matrices

    Args:
        confusions: List with shape (batch_size, len(labels), len(labels))
        labels: List of class names

        returns: dictionary of class names and iou scores for that class (summed across whole matrix list)
    """
    conf_matrix = np.array(confusions)

    if len(conf_matrix.shape) != 3 or conf_matrix.shape[1] != conf_matrix.shape[2]:
        print(f"❌ ERROR: Confusion matrix shape is {conf_matrix.shape}, expected (N, num_classes, num_classes)")
        return {label: 0.0 for label in labels}  # Return invalid IoU
    
    conf_matrix = np.sum(confusions, axis=0)
    num_classes = conf_matrix.shape[0]

    true_positive = np.diag(conf_matrix) + 1e-6
    # true_positive = np.diag(conf_matrix)
    false_negative = np.sum(conf_matrix, 0) - true_positive
    false_positive = np.sum(conf_matrix, 1) - true_positive
    #iou = true_positive / (true_positive + false_positive + false_negative)

    #iou = np.maximum(iou, 0)
    denominator = true_positive + false_positive + false_negative
    iou = np.where(denominator > 0, true_positive / denominator, 0.0)  

    iou_dict = {}
    for i, l in enumerate(labels):
        if i < num_classes:
            iou_dict[labels[i]] = iou[i]
        else:
            print(f"⚠️ WARNING: Class index {i} is missing in confusion matrix. Assigning IoU = 0.0")
            iou_dict[labels[i]] = 0.0  # Handle missing classes safely
    return iou_dict


def calculate_recall(confusions, labels):
    confusions = np.array(confusions)
    conf_matrix = np.sum(confusions, axis=0)
    true_positive = np.diag(conf_matrix) + 1e-6
    false_negative = np.sum(conf_matrix, 0) - true_positive
    recall = true_positive / (true_positive + false_negative  + 1e-6)

    recall_dict = {}
    for i, l in enumerate(labels):
        recall_dict[l] = recall[i]
    return recall_dict

def compute_confusion_matrix(y_pred, y_true, num_classes):
    """
    Computes the confusion matrix for IoU calculation.
    
    Args:
        y_pred: (H, W) predicted labels
        y_true: (H, W) ground truth labels
        num_classes: Total number of classes
    
    Returns:
        (num_classes, num_classes) confusion matrix
    """
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid_mask = (y_true < num_classes) & (y_pred < num_classes)

    if np.sum(valid_mask) == 0:  # If everything is class 3, set a dummy entry
        print("⚠️ WARNING: No valid class labels found in y_pred_ or y_true_!")
        return conf_matrix  # Will be all zeros
    
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    for i in range(num_classes):
        for j in range(num_classes):
            conf_matrix[i, j] = np.sum((y_true == i) & (y_pred == j))

    return conf_matrix

def process_and_plot(images):
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    axes = axes.flatten()
    colours = ['brown',    # 0: land
             'blue',      # 1: water
             'white',     # 2: cloud
             'gray']      # 3: invalid
    cmap = plt.cm.colors.ListedColormap(colours)
    bounds = [0, 1, 2, 3, 4]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    for idx, (y_pred, image_gt) in enumerate(images):
        if y_pred is None or image_gt is None or y_pred.size == 0 or image_gt.size == 0:
            print(f"⚠️ Skipping empty image at index {idx}")
            continue
        
        # Convert y_pred to a displayable format
        img_to_display = y_pred.astype(np.uint8)

        axes[idx].imshow(img_to_display, cmap=cmap, norm=norm)
        axes[idx].set_title(f"Prediction {idx+1}")
        axes[idx].axis("off")
    #add a colour bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=axes.ravel().tolist())
    cbar.set_ticks([0.5, 1.5, 2.5, 3.5])
    cbar.set_ticklabels(['Land', 'Water', 'Cloud', 'Invalid'])
    plt.tight_layout()


def plot_confusion_matrix(conf_matrix, labels):
    fig, ax = plt.subplots(figsize=(8, 8))
    conf_matrix_norm = conf_matrix.astype(np.float64) / conf_matrix.sum(axis=1, keepdims=True) * 100  # Normalize

    sns.heatmap(conf_matrix_norm, annot=True, fmt=".1f", cmap="magma", xticklabels=labels, yticklabels=labels, cbar=True, linewidths=0.5, ax=ax)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Confusion Matrix (Normalized %)")

def visualize_predictions(model, test_loader, num_samples=5):
    """
    Visualize model predictions alongside ground truth
    Args:
        model: The trained model
        test_loader: DataLoader containing test data
        num_samples: Number of samples to visualize
    """
    for i, batch in enumerate(test_loader):
        if i >= num_samples:
            break
            
        # Get predictions
        x = batch['image'].to(model.device)
        
        # Extract features and classify
        latent_features = model.extract_latent_features(x)
        raw_features = x
        outputs = model.classify(latent_features, raw_features)
        
        # Reshape prediction to image dimensions
        B, HW = outputs.shape
        H = W = int(HW ** 0.5)  # Assuming square images
        pred = outputs.view(B, H, W)
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot input image (RGB)
        ax1.imshow(batch['image'][0, [3,2,1]].permute(1,2,0).cpu())
        ax1.set_title('Input Image')
        ax1.axis('off')
        
        # Plot prediction
        ax2.imshow(pred[0].cpu())
        ax2.set_title('Prediction')
        ax2.axis('off')
        
        # Plot ground truth
        ax3.imshow(batch['mask'][0].cpu())
        ax3.set_title('Ground Truth')
        ax3.axis('off')

def visualize_original_images(images_list):
    """
    Plot original images in true RGB colors
    Args:
        images_list: List of tuples containing (image, mask)
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    axes = axes.flatten()
    
    for idx, (image, _) in enumerate(images_list):
        if idx >= 6:  # Only show first 6 images
            break
            
        # Convert to RGB (bands 3,2,1 for Sentinel-2)
        rgb_image = image[[3,2,1]]  # Select RGB bands
        
        # Normalize for display
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        rgb_image = np.clip(rgb_image, 0, 1)
        
        # Display
        axes[idx].imshow(rgb_image.transpose(1, 2, 0))
        axes[idx].set_title(f"Original Image {idx+1}")
        axes[idx].axis("off")
    
    plt.tight_layout()



def process_inference(dataloader: torch.utils.data.DataLoader, model):
    #visualize_predictions(model, dataloader, num_samples=3)
    labels = ["land", "water", "cloud", "invalid"]

    images_plot = []
    original_images = []
    mets = []
    #batch_id = 0
    for batch in dataloader: #Loop through the folders
        #print(f"batch number: {batch_id}") # prints whole batch big no
        with torch.no_grad(): # the .model interpretation for inference
            if isinstance(batch, dict):
                x = batch['image']#.to(model.device)
                y = batch['mask']#.to(model.device)
            else:
                x, y = batch
                x = x#.to(model.device)
                y = y#.to(model.device)

            x = torch.nan_to_num(x).to(dtype=torch.float32, device=model.device)
            y = torch.nan_to_num(y).to(dtype=torch.long)

            """latent_features = model.extract_latent_features(x)
            raw_features = x
            pred = model.classify(latent_features, raw_features)  # Get class logits"""
            #print("perfoming interpretable rules on infernece batch")
            pred = model(x)

            pred = pred.long()
            B, HW = pred.shape
            H, W = int(HW ** 0.5), int(HW ** 0.5)  # Assuming square images
            pred = pred.view(B, H, W)

        #valid_classes = [0, 1, 2, 3]  # Expected classes
        #print(pred)
        #print(y)
        #print(y.shape)
        y = y.squeeze(1)
        print(y.shape)

        print(np.unique(pred.cpu().numpy()))
        print(np.unique(y.cpu().numpy()))

        pred = pred.clone().detach().to(dtype=torch.long, device=model.device)
        mask = y.clone().detach().to(dtype=torch.long, device=model.device)
        print(pred.shape)

        pred = torch.where(pred == 120, torch.tensor(0, device=model.device), pred)
        pred = torch.where(pred == 141, torch.tensor(1, device=model.device), pred)
        pred = torch.where(pred == 36, torch.tensor(2, device=model.device), pred)
        pred = torch.where(pred == 84, torch.tensor(3, device=model.device), pred)

        for i in range(min(B, 6 - len(images_plot))):  # Take what we need from this batch
            if len(images_plot) < 6:  # Only store up to 6 images
                images_plot.append((
                    pred[i].cpu().numpy(),  # Prediction
                    y[i].cpu().numpy()      # Ground truth
                ))
                original_images.append((
                        x[i].cpu().numpy(),      # Original image
                        y[i].cpu().numpy()       # Original ground truth
                ))
        
        #mask = torch.where(mask == 120, torch.tensor(0, device=model.device), mask)
        #mask = torch.where(mask == 141, torch.tensor(1, device=model.device), mask)
        #mask = torch.where(mask == 36, torch.tensor(2, device=model.device), mask)
        #mask = torch.where(mask == 84, torch.tensor(3, device=model.device), mask)
        
        print(torch.unique(mask))

        num_classes = 4
        #conf_matrix = np.array(compute_confusion_matrix(y_pred_, image_gt_, num_classes))
        conf_matrix = metrics.compute_confusions(mask, pred, num_class=num_classes, remove_class_zero=True)
        if conf_matrix.dim() == 3:  # If shape is (B, 4, 4)
            conf_matrix = conf_matrix.sum(dim=0)  # Convert to (4, 4)
        mets.append(conf_matrix)
        #print(f"Processed batch {batch_id + 1}/{len(dataloader)}")

    # Convert mets to a tensor and sum across batch dimension
    if mets:
        print("\nComputing final metrics...")
        try:
            # Stack and sum confusion matrices
            mets_tensor = torch.stack(mets)  # Should now work as all matrices are (4, 4)
            values = torch.sum(mets_tensor, dim=0)
            
            # Calculate metrics
            filtered_conf_matrix = values[:3, :3].cpu().numpy()
            
            # Print results
            print("\nResults:")
            print("Full Confusion Matrix:")
            print(values.cpu().numpy())
            print("\nFiltered Confusion Matrix (3x3):")
            print(filtered_conf_matrix)

            iou_dict = metrics.calculate_iou(filtered_conf_matrix, labels=["land", "water", "cloud", "invalid"])
            print("\nIoU per class:")
            for label, iou in iou_dict.items():
                print(f"{label}: {iou:.3f}")
            print("Mean IoU =", np.mean(list(iou_dict.values())))

            print(f"DEBUG: mets shape = {np.array(mets).shape}")
            print(f"DEBUG: First Confusion Matrix in mets =\n{mets[0] if len(mets) > 0 else 'Empty'}")
            #original_method = calculate_iou(mets,labels)
            #print(original_method)
            print("-----------")

            recall_water = calculate_recall(filtered_conf_matrix,labels)
            print("water recall: ", recall_water)
            values = np.sum(mets, axis=0)
            print("Final Confusion Matrix:\n", values)
      
        except Exception as e:
            print(f"Error processing metrics: {str(e)}")
            print(f"Shapes of confusion matrices: {[m.shape for m in mets]}")
            
            # Additional debugging information
            print("\nFirst confusion matrix:")
            print(mets[0])
            print("\nLast confusion matrix:")
            print(mets[-1])
            
    else:
        print("⚠️ ERROR: No valid confusion matrices were collected. Check preprocessing.")
        return
    
    #valid_classes = [0, 1, 2]  # Only compute IoU for land, water, cloud
    #filtered_conf_matrix = mets[:, :3, :3]
    #iou_results = calculate_iou(filtered_conf_matrix, labels)
    

    #print(values)

    #class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[1,0]+ values[2,0])
    #class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[0,1]+ values[2,1])
    #class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[0,2]+ values[1,2])
    # class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

    #rint("IoU for class 0 is: ", class1_IoU)
    #print("IoU for class 1 is: ", class2_IoU)
    #print("IoU for class 2 is: ", class3_IoU)
    plot_confusion_matrix(filtered_conf_matrix, labels[:3])
    process_and_plot(images_plot)
    visualize_original_images(original_images)
    
    #plot_confusion_matrix(filtered_conf_matrix, labels)
    plt.show()

    # Compute IoU for each class
    denominator = filtered_conf_matrix.sum(axis=1) + filtered_conf_matrix.sum(axis=0) - np.diag(filtered_conf_matrix)
    iou_values = np.diag(filtered_conf_matrix) / denominator
    iou_values = np.nan_to_num(iou_values)  # Handle NaN values

    for i, label in enumerate(labels[:3]):  # Exclude 'invalid' class
        print(f"IoU for {label}: {iou_values[i]:.3f}")

