from cloudsen12_models import cloudsen12
import ee
import matplotlib.pyplot as plt
from georeader import plot
from shapely.geometry import box
from georeader.readers import ee_image

"""
cloud segmentation
"""

# ee.Authenticate()
ee.Initialize(project="kherson-tutorial")

#select image based on gee id
collection_name = "COPERNICUS/S2_HARMONIZED"
tile = "S2A_MSIL1C_20240417T064631_N0510_R020_T40RCN_20240417T091941"
img_col = ee.ImageCollection(collection_name)
image = img_col.filter(ee.Filter.eq("PRODUCT_ID", tile)).first()
info_img = image.getInfo()

#use lat/lon to download obj, result is geotensor object
aoi = box(55.325, 25.225, 55.415, 25.28)

bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
crs = info_img["bands"][1]["crs"]
transform = info_img["bands"][1]["crs_transform"]
projgee = {"crs": crs, "transform": transform}
img_local = ee_image.export_image_getpixels(asset_id=info_img['id'],
                                            proj=projgee,
                                            bands_gee=bands,
                                            geometry=aoi)
print(img_local) # geotensor object

swirnirred = (img_local.isel({"band": [bands.index(b) for b in ["B11","B8","B4"]]}) / 4_500.).clip(0,1)
#plot.show(swirnirred)

# load the model
model = cloudsen12.load_model_by_name(name="cloudsen12", weights_folder="cloudsen12_models")

# predict
cloudmask = model.predict(img_local/10_000)

#plot prediction
fig, ax = plt.subplots(1,2,figsize=(14,5),sharey=True, tight_layout=True)

plot.show(swirnirred,ax=ax[0])
cloudsen12.plot_cloudSEN12mask(cloudmask,ax=ax[1])

# fig.savefig("example_flood_dubai_2024.png")


"""
flood segmentation 
"""
# download model
from huggingface_hub import hf_hub_download
# os.makedirs("models/WF2_unetv2_bgriswirs", exist_ok=True)
experiment_name = "WF2_unetv2_bgriswirs"
subfolder_local = f"models/{experiment_name}"
config_file = hf_hub_download(repo_id="isp-uv-es/ml4floods",subfolder=subfolder_local, filename="config.json",
                              local_dir=".", local_dir_use_symlinks=False)
model_file = hf_hub_download(repo_id="isp-uv-es/ml4floods",subfolder=subfolder_local, filename="model.pt",
                              local_dir=".", local_dir_use_symlinks=False)

import numpy as np
import torch
from ml4floods.scripts.inference import load_inference_function, vectorize_outputv1
from ml4floods.models.model_setup import get_channel_configuration_bands

# load model and define wrapper func around inference
inference_function, config = load_inference_function(subfolder_local, device_name = 'cpu', max_tile_size=1024,
                                                     th_water=0.7, th_brightness=3500, 
                                                     distinguish_flood_traces=True)

channel_configuration = config['data_params']['channel_configuration']
channels  = get_channel_configuration_bands(channel_configuration, collection_name='S2')

def predict(input_tensor, channels = [1, 2, 3, 7, 11, 12] ): # fiddle with this to see if you can increase the channels used 
    input_tensor = input_tensor.astype(np.float32)
    input_tensor = input_tensor[channels]
    torch_inputs = torch.tensor(np.nan_to_num(input_tensor))
    return inference_function(torch_inputs)

# Predict using GeoTensor .values
prediction_postflood, prediction_postflood_cont  = predict(img_local.values, channels)

"""
combine cloudSense and flood segmentation outputs
"""

# from ml4floods.visualization.plot_utils import plot_segmentation_mask
from georeader.geotensor import GeoTensor

interpretation_array=["invalids", "land", "water", "cloud", "cloud shadow"]

COLORS_PRED = np.array([[0, 0, 0], # 0: invalid
                       [139, 64, 0], # 1: land
                       [0, 0, 240], # 2: water
                       [220, 220, 220], # 3: cloud
                       # [60, 85, 92], # 4: flood_trace
                       [60, 60, 60]], # 5: cloud_shadow
                    dtype=np.float32) / 255

prediction_tensor = torch.ones_like(prediction_postflood)

prediction_tensor[(prediction_postflood == 2)] = 2 # water is 2
# prediction_tensor[(prediction_postflood == 4)] = 4 # flood trace is 4
prediction_tensor[cloudmask.values == 1] = 3
prediction_tensor[cloudmask.values == 3] = 4

prediction_tensor = GeoTensor(prediction_tensor.cpu().numpy(), 
                              transform=swirnirred.transform, crs=swirnirred.crs,
                              fill_value_default=0)

fig, ax = plt.subplots(1,2,figsize=(14,5),sharey=True, tight_layout=True)

plot.show(swirnirred, ax=ax[0], add_scalebar=True)
plot.plot_segmentation_mask(prediction_tensor,
                            color_array=COLORS_PRED, 
                            interpretation_array=interpretation_array, legend = True)

plt.show()
