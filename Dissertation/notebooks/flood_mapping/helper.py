import rasterio
import numpy as np
from rasterio import plot as rasterioplt
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches

from typing import Optional, Tuple, Union

import torch
from ml4floods.data.worldfloods.configs import BANDS_S2
from ml4floods.visualization.plot_utils import plot_segmentation_mask


@torch.no_grad()
def read_inference_pair(tiff_inputs:str, folder_ground_truth:str, 
                        window:Optional[Union[rasterio.windows.Window, Tuple[slice,slice]]], 
                        return_ground_truth: bool=False, channels:bool=None, 
                        folder_permanent_water=Optional[str],
                       cache_folder=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, rasterio.Affine]:
    """
    Read a pair of layers from the worldfloods bucket and return them as Tensors to pass to a model, return the transform for plotting with lat/long
    
    Args:
        tiff_inputs: filename for layer in worldfloods bucket
        folder_ground_truth: folder name to be replaced by S2 in the input
        window: window of layer to use
        return_ground_truth: flag to indicate if paired gt layer should be returned
        channels: list of channels to read from the image
        return_permanent_water: Read permanent water layer raster
    
    Returns:
        (torch_inputs, torch_targets, transform): inputs Tensor, gt Tensor, transform for plotting with lat/long
    """
    
    if cache_folder is not None and tiff_inputs.startswith("gs"):
        tiff_inputs = download_tiff(cache_folder, tiff_inputs, folder_ground_truth, folder_permanent_water) # could be here 
    
    tiff_targets = tiff_inputs.replace("/S2/", folder_ground_truth)
    print("here")
    print(f"Attempting to read TIFF cache_folder: {cache_folder}")
    print(f"Attempting to open TIFF file: {tiff_inputs}")
    print(f"Attempting to read TIFF gt: {folder_ground_truth}")
    print(f"Attempting to read TIFF folder_permanent_water: {folder_permanent_water}")
    print("here")
 

    try:
        with rasterio.open(tiff_inputs, "r") as rst: #issue here
            inputs = rst.read((np.array(channels) + 1).tolist(), window=window)
            # Shifted transform based on the given window (used for plotting)
            transform = rst.transform if window is None else rasterio.windows.transform(window, rst.transform)
            torch_inputs = torch.Tensor(inputs.astype(np.float32)).unsqueeze(0)
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    if folder_permanent_water is not None:
        tiff_permanent_water = tiff_inputs.replace("/S2/", folder_permanent_water)
        with rasterio.open(tiff_permanent_water, "r") as rst:
            permanent_water = rst.read(1, window=window)  
            torch_permanent_water = torch.tensor(permanent_water)
    else:
        torch_permanent_water = torch.zeros_like(torch_inputs)
        
    if return_ground_truth:
        with rasterio.open(tiff_targets, "r") as rst:
            targets = rst.read(1, window=window)
        
        torch_targets = torch.tensor(targets).unsqueeze(0)
    else:
        torch_targets = torch.zeros_like(torch_inputs)
    
    return torch_inputs, torch_targets, torch_permanent_water, transform

COLORS_WORLDFLOODS = np.array([[0, 0, 0], # invalid
                               [139, 64, 0], # land
                               [0, 0, 139], # water
                               [220, 220, 220]], # cloud
                              dtype=np.float32) / 255

INTERPRETATION_WORLDFLOODS = ["invalid", "land", "water", "cloud"]

COLORS_WORLDFLOODS_PERMANENT = np.array([[0, 0, 0], # 0: invalid
                                         [139, 64, 0], # 1: land
                                         [237, 0, 0], # 2: flood_water
                                         [220, 220, 220], # 3: cloud
                                         [0, 0, 139], # 4: permanent_water
                                         [60, 85, 92]], # 5: seasonal_water
                                        dtype=np.float32) / 255

INTERPRETATION_WORLDFLOODS_PERMANENT = ["invalid", "land", "flood water", "cloud", "permanent water", "seasonal water"]

def gt_with_permanent_water(gt: np.ndarray, permanent_water: np.ndarray)->np.ndarray:
    """ Permanent water taken from: https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_2_YearlyHistory"""
    gt[(gt == 2) & (permanent_water == 3)] = 4 # set as permanent_water
    gt[(gt == 2) & (permanent_water == 2)] = 5 # set as seasonal water
        
    return gt
            

def get_cmap_norm_colors(color_array, interpretation_array):
    cmap_categorical = colors.ListedColormap(color_array)
    norm_categorical = colors.Normalize(vmin=-.5,
                                        vmax=color_array.shape[0]-.5)
    patches = []
    for c, interp in zip(color_array, interpretation_array):
        patches.append(mpatches.Patch(color=c, label=interp))
    
    return cmap_categorical, norm_categorical, patches


def plot_inference_set(inputs: torch.Tensor, targets: torch.Tensor, 
                       predictions: torch.Tensor, permanent_water: torch.Tensor, transform: rasterio.Affine)->None:
    """
    Plots inputs, targets and prediction into lat/long visualisation
    
    Args:
        inputs: input Tensor
        targets: gt target Tensor
        prediction: predictions output by model (softmax, argmax already applied)
        permanent_water: permanent water raster
        transform: transform used to plot with lat/long
    """
    fig, ax = plt.subplots(2,2,figsize=(16,16))
    
    inputs_show = inputs.cpu().numpy().squeeze()
    targets_show = targets.cpu().numpy().squeeze()
    permanent_water_show = permanent_water.numpy().squeeze()
    
    targets_show = gt_with_permanent_water(targets_show, permanent_water_show)
    
    
    # Color categories {-1: invalid, 0: land, 1: water, 2: clouds}
    
    cmap_preds, norm_preds, patches_preds = get_cmap_norm_colors(COLORS_WORLDFLOODS, INTERPRETATION_WORLDFLOODS)
    cmap_gt, norm_gt, patches_gt = get_cmap_norm_colors(COLORS_WORLDFLOODS_PERMANENT, INTERPRETATION_WORLDFLOODS_PERMANENT)
    
    prediction_show = (predictions).cpu().numpy() 
    
    band_names_current_image = [BANDS_S2[iband] for iband in channels]
    bands_rgb = [band_names_current_image.index(b) for b in ["B4", "B3", "B2"]] # swir_1, nir, red composite
    bands_false_composite = [band_names_current_image.index(b) for b in ["B11", "B8", "B4"]] # swir_1, nir, red composite
    false_rgb = np.clip(inputs_show[bands_false_composite, :, :]/3000.,0,1)
    rgb = np.clip(inputs_show[bands_rgb, :, :]/3000.,0,1)
    

    rasterioplt.show(rgb,transform=transform,ax=ax[0,0])
    ax[0,0].set_title("RGB Composite")
    rasterioplt.show(false_rgb,transform=transform,ax=ax[0,1])
    ax[0,1].set_title("SWIR1,NIR,R Composite")

    plot_segmentation_mask(targets_show,transform=transform, ax = ax[1,0], 
                           color_array=COLORS_WORLDFLOODS_PERMANENT, interpretation_array=INTERPRETATION_WORLDFLOODS_PERMANENT)


    plot_segmentation_mask(prediction_show,transform=transform, ax = ax[1,1], color_array=COLORS_WORLDFLOODS, interpretation_array=INTERPRETATION_WORLDFLOODS)
    
    ax[1,0].set_title("Ground Truth")
    ax[1,0].legend(handles=patches_gt,
                 loc='upper right')
    
    ax[1,1].set_title("Prediction water")
    ax[1,1].legend(handles=patches_preds,
                   loc='upper right')
    
