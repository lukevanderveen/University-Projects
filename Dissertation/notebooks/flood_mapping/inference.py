"""
 this is a newer inferences tutorial: https://spaceml-org.github.io/ml4floods/content/ml4ops/HOWTO_Run_Inference_multioutput_binary.html#ml4floods-trained-model

"""

import os
from huggingface_hub import hf_hub_download, hf_hub_url

import torch
import numpy as np
from shapely.geometry import shape

from georeader.rasterio_reader import RasterioReader
from georeader.geotensor import GeoTensor
from georeader.readers import ee_query
from georeader.readers import ee_image
from datetime import datetime
from georeader import plot

import ee
from ml4floods.visualization import plot_utils
import matplotlib.pyplot as plt

from helper import plot_inference_set, read_inference_pair

COLORS_PRED = np.array([[0, 0, 0], # 0: invalid
                       [139, 64, 0], # 1: land
                       [0, 0, 240], # 2: water
                       [220, 220, 220], # 3: cloud
                       [60, 85, 92]], # 5: flood_trace
                    dtype=np.float32) / 255

#load trained model 
experiment_name = "WF2_unetv2_bgriswirs"
subfolder_local = f"models/{experiment_name}"
config_file = hf_hub_download(repo_id="isp-uv-es/ml4floods",subfolder=subfolder_local, filename="config.json",
                              local_dir=".", local_dir_use_symlinks=False)
model_file = hf_hub_download(repo_id="isp-uv-es/ml4floods",subfolder=subfolder_local, filename="model.pt",
                              local_dir=".", local_dir_use_symlinks=False)


from ml4floods.scripts.inference import load_inference_function
from ml4floods.models.model_setup import get_channel_configuration_bands

inference_function, config = load_inference_function(subfolder_local, device_name = 'cpu', max_tile_size=1024,
                                                     th_water=0.7, th_brightness=3500,
                                                     distinguish_flood_traces=True)

channel_configuration = config['data_params']['channel_configuration']
channels  = get_channel_configuration_bands(channel_configuration, collection_name='S2')

#make a prediction with the model with the selected channels 
def predict(input_tensor, channels = [1, 2, 3, 7, 11, 12] ):
    input_tensor = input_tensor.astype(np.float32)
    input_tensor = input_tensor[channels]
    torch_inputs = torch.tensor(np.nan_to_num(input_tensor))
    return inference_function(torch_inputs)

# Select image to use
subset = "test"
filename = "EMSR264_18MIANDRIVAZODETAIL_DEL_v2"

s2url = hf_hub_url(repo_id="isp-uv-es/WorldFloodsv2",
                   subfolder=f"{subset}/S2", filename=f"{filename}.tif",
                   repo_type="dataset")


channels  = get_channel_configuration_bands(channel_configuration, collection_name='S2')

s2rst = RasterioReader(s2url).isel({"band": channels})
s2rst = s2rst.load()

prediction_test, prediction_test_cont  = predict(s2rst.values, channels = list(range(len(channels))))
prediction_test_raster = GeoTensor(prediction_test.numpy(), transform=s2rst.transform,
                                   fill_value_default=0, crs=s2rst.crs)
prediction_test_raster

#create a figure for observations
fig, ax = plt.subplots(1,2,figsize=(14,7), sharey=True)


plot.show((s2rst.isel({"band": [4,3,2]})/3_500).clip(0,1), ax=ax[0], add_scalebar=True)
ax[0].set_title(f"{subset}/{filename}")


plot.plot_segmentation_mask(prediction_test_raster, COLORS_PRED, ax=ax[1],
                            interpretation_array=["invalids", "land", "water", "cloud", "flood_trace"])

ax[1].set_title(f"{subset}/{filename} floodmap")

print(ax[0], " ", ax[1])

# download a sentinel 2 image 
ee.Authenticate()
ee.Initialize(project="kherson-tutorial")

aoi = shape({'type': 'Polygon',
 'coordinates': [[[153.20789834941638, -28.75874177524779],
          [153.20789834941638, -28.91332819718112],
          [153.38848611797107, -28.91332819718112],
          [153.38848611797107, -28.75874177524779]]]})

print("intitalised")

s2data = ee_query.query(aoi, datetime(2022,3,1), datetime(2022,3,2), producttype="S2")
bands_s2 = get_channel_configuration_bands(channel_configuration, collection_name='S2',as_string=True)
asset_id = f"{s2data.iloc[0].collection_name}/{s2data.iloc[0].gee_id}"
geom = s2data.iloc[0].geometry.intersection(aoi)
postflood = ee_image.export_image_getpixels(asset_id, geom, proj=s2data.iloc[0].proj,bands_gee=bands_s2)
postflood
print(postflood)
print("prepared to run infrence")

#run inference
prediction_postflood, prediction_postflood_cont  = predict(postflood.values, channels = list(range(len(bands_s2))))
prediction_postflood_raster = GeoTensor(prediction_postflood.numpy(), transform=postflood.transform,
                                        fill_value_default=0, crs=postflood.crs)

print("infrences ran")




# plot results
print("plotting results")
fig, ax = plt.subplots(1,2,figsize=(14,7))

plot.show((postflood.isel({"band": [4,3,2]})/3_500).clip(0,1), ax=ax[0], add_scalebar=True)
ax[0].set_title(f"{s2data.iloc[0].satellite} {s2data.iloc[0].solarday}")


plot.plot_segmentation_mask(prediction_postflood_raster, COLORS_PRED, ax=ax[1],
                            interpretation_array=["invalids", "land", "water", "cloud", "flood_trace"])


ax[1].set_title(f"{s2data.iloc[0].solarday} floodmap")


print("results plotted")

#landsat image

#dowload
satdata = ee_query.query(aoi, datetime(2022,4,4), datetime(2022,4,5), producttype="Landsat")
bands_landsat = get_channel_configuration_bands(channel_configuration, collection_name='Landsat',as_string=True)
asset_id = f"{satdata.iloc[0].collection_name}/{satdata.iloc[0].gee_id}"
geom = satdata.iloc[0].geometry.intersection(aoi)
postfloodl8 = ee_image.export_image_getpixels(asset_id, geom, proj=satdata.iloc[0].proj,bands_gee=bands_landsat)
postfloodl8.values *= 10000
postfloodl8

#run inference
prediction_postfloodl8, prediction_postflood_contl8  = predict(postfloodl8.values, channels = list(range(len(bands_s2))))
prediction_postfloodl8_raster = GeoTensor(prediction_postfloodl8.numpy(), transform=postfloodl8.transform,
                                        fill_value_default=0, crs=postfloodl8.crs)
prediction_postfloodl8_raster

#plot results

fig, ax = plt.subplots(1,2,figsize=(14,7))

plot.show((postfloodl8.isel({"band": [4,3,2]})/3_500).clip(0,1), ax=ax[0], add_scalebar=True)
ax[0].set_title(f"{satdata.iloc[0].satellite} {satdata.iloc[0].solarday}")


plot.plot_segmentation_mask(prediction_postfloodl8_raster, COLORS_PRED, ax=ax[1],
                            interpretation_array=["invalids", "land", "water", "cloud", "flood_trace"])

ax[1].set_title(f"{satdata.iloc[0].solarday} floodmap")
#plt.show()

#reading from gcp bucket
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"D:\Documents\coding\300\ml4floods\notebooks\flood_mapping\kherson-tutorial-f8ca1e1a82d6.json"
os.environ['GS_USER_PROJECT']= 'kherson-tutorial'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'# use the first available gpu

from ml4floods.models.model_setup import get_channel_configuration_bands

download_image = False
cache_folder = None # "tiffs_for_inference"
#os.makedirs(cache_folder, exist_ok=True)

import rasterio.windows 
window = rasterio.windows.Window(col_off=1543, row_off=247, 
                                 width=2000, height=2000)
tiff_s2, channels = "gs://ml4cc_data_lake/2_PROD/1_Staging/WorldFloods/S2/EMSR501/AOI01/EMSR501_AOI01_DEL_MONIT01_r1_v1.tif", get_channel_configuration_bands(config.model_params.hyperparameters.channel_configuration)

torch_inputs, torch_targets, torch_permanent_water, transform = read_inference_pair(tiff_s2, folder_ground_truth="/GT/V_1_1/", 
                                                                                    window=window, 
                                                                                    return_ground_truth=True, channels=channels,
                                                                                    folder_permanent_water="/JRC/",

                                                                                    cache_folder=cache_folder)
outputs, cont_pred = inference_function(torch_inputs[0]) 
plot_inference_set(torch_inputs, torch_targets, outputs, torch_permanent_water, transform)