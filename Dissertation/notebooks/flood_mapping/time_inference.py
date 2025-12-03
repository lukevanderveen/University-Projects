from datetime import datetime, timedelta, timezone
import geopandas as gpd
import pandas as pd
import ee
import geemap.foliumap as geemap
from ml4floods.data import ee_download
from shapely.geometry import mapping, shape
import matplotlib.pyplot as plt
from georeader.readers import ee_query, ee_image
from huggingface_hub import hf_hub_download
from ml4floods.scripts.inference import load_inference_function
from ml4floods.models.model_setup import get_channel_configuration_bands
from georeader.readers import S2_SAFE_reader
from georeader.save import save_cog
from ml4floods.scripts.inference import vectorize_outputv1
from georeader import plot
from tqdm import tqdm
from georeader.rasterio_reader import RasterioReader
from georeader.geotensor import GeoTensor
import numpy as np
import matplotlib.colors
import warnings
import torch
import os

"""
time series inference
"""

#data configuration to search for S2 images
date_event = datetime.strptime("2021-02-12","%Y-%m-%d").replace(tzinfo=timezone.utc)

date_start_search = datetime.strptime("2021-01-15","%Y-%m-%d").replace(tzinfo=timezone.utc)
date_end_search = date_start_search + timedelta(days=45)

area_of_interest_geojson = {'type': 'Polygon',
 'coordinates': (((19.483318354000062, 41.84407200000004),
   (19.351701478000052, 41.84053242300007),
   (19.298659824000026, 41.871157520000054),
   (19.236388306000038, 41.89588351100008),
   (19.22956438700004, 42.086957306000045),
   (19.327827977000027, 42.09102668200006),
   (19.778082109000025, 42.10312055000003),
   (19.777652446000047, 41.97309238100007),
   (19.777572772000042, 41.94912981900006),
   (19.582705341000064, 41.94398333100003),
   (19.581417139000052, 41.94394820700006),
   (19.54282145700006, 41.90168177700008),
   (19.483318354000062, 41.84407200000004)),)}

area_of_interest = shape(area_of_interest_geojson)


ee.Initialize(project="kherson-tutorial")

# This function returns a GEE collection of Sentinel-2 and Landsat 8 data and a Geopandas Dataframe with data related to the tiles, overlap percentage and cloud cover
img_col_info_local, img_col = ee_query.query(
    area=area_of_interest, 
    date_start=date_start_search, 
    date_end=date_end_search,                                                   
    producttype="S2", 
    return_collection=True, 
    add_s2cloudless=False)

# Grab the S2 images and the Permanent water image
n_images_col = img_col_info_local.shape[0]
print(f"Found {n_images_col} S2 images between {date_event.isoformat()} and {date_end_search.isoformat()}")

plt.figure(figsize=(15,5))
plt.plot(img_col_info_local['utcdatetime'], img_col_info_local['cloudcoverpercentage'],marker="x")
plt.ylim(0,101)
plt.xticks(rotation=30)
plt.ylabel("mean cloud coverage % over AoI")
plt.grid(axis="x")
img_col_info_local.columns


import geemap.foliumap as geemap
import folium

tl = folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr='Google',
            name="Google Satellite",
            overlay=True,
            control=True,
            max_zoom=22,
        )

m = geemap.Map(location=area_of_interest.centroid.coords[0][-1::-1], 
               zoom_start=8)

tl.add_to(m)

img_col_info_local["localdatetime_str"] = img_col_info_local["localdatetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
showcolumns = ["geometry","overlappercentage","cloudcoverpercentage", "localdatetime_str","solarday","satellite"]
colors = ["#ff7777", "#fffa69", "#8fff84", "#52adf1", "#ff6ac2","#1b6d52", "#fce5cd","#705334"]
   
# Add the extent of the products
for i, ((day,satellite), images_day) in enumerate(img_col_info_local.groupby(["solarday","satellite"])):
    images_day[showcolumns].explore(
        m=m, 
        name=f"{satellite}: {day} outline", 
        color=colors[i % len(colors)], 
        show=False)

# Add the S2 data
for (day, satellite), images_day in img_col_info_local.groupby(["solarday", "satellite"]):    
    if images_day.cloudcoverpercentage.mean() >= 50:
        continue
    
    image_col_day_sat = img_col.filter(ee.Filter.inList("title", images_day.index.tolist()))    
    bands = ["B11","B8","B4"] if satellite.startswith("S2") else ["B6","B5","B4"]
    m.addLayer(image_col_day_sat, 
               {"min":0, "max":3000 if satellite.startswith("S2") else 0.3, "bands": bands},
               f"{satellite}: {day}",
               False)

aoi_gpd = gpd.GeoDataFrame({"geometry": [area_of_interest]}, crs= "EPSG:4326",geometry="geometry")
aoi_gpd.explore(style_kwds={"fillOpacity": 0}, color="black", name="AoI", m=m)
folium.LayerControl(collapsed=False).add_to(m)


experiment_name = "WF2_unetv2_bgriswirs"
subfolder_local = f"models/{experiment_name}"
config_file = hf_hub_download(repo_id="isp-uv-es/ml4floods",subfolder=subfolder_local, filename="config.json",
                              local_dir=".", local_dir_use_symlinks=False)
model_file = hf_hub_download(repo_id="isp-uv-es/ml4floods",subfolder=subfolder_local, filename="model.pt",
                              local_dir=".", local_dir_use_symlinks=False)

inference_function, config = load_inference_function(subfolder_local, device_name = 'cpu', max_tile_size=1024,
                                                     th_water=0.5, th_brightness=3500,
                                                     distinguish_flood_traces=True)

channel_configuration = config['data_params']['channel_configuration']
channels  = get_channel_configuration_bands(channel_configuration, collection_name='S2')

def predict(input_tensor, channels = [1, 2, 3, 7, 11, 12] ):
    input_tensor = input_tensor.astype(np.float32)
    input_tensor = input_tensor[channels]
    torch_inputs = torch.tensor(np.nan_to_num(input_tensor))
    return inference_function(torch_inputs)

COLORS_PRED = np.array([[0, 0, 0], # 0: invalid
                       [139, 64, 0], # 1: land
                       [0, 0, 240], # 2: water
                       [220, 220, 220], # 3: cloud
                       [60, 85, 92]], # 5: flood_trace
                    dtype=np.float32) / 255

band_names_S2  = get_channel_configuration_bands(channel_configuration, collection_name='S2', as_string=True)
path_to_export = "cache_s2"
os.makedirs(path_to_export, exist_ok=True)

floodmaps = {}
for i in tqdm(range(n_images_col), total=n_images_col):
    s2data = img_col_info_local.iloc[i]
    if s2data.cloudcoverpercentage > 50:
        continue

    date = s2data.solarday
    filename = os.path.join(path_to_export,f"albania_ts_{date}.tif")
    filename_pred = os.path.join(path_to_export,f"albania_ts_{date}_pred.tif")
    filename_jpg = os.path.join(path_to_export,f"albania_ts_{date}.jpg")
    filename_gkpg = os.path.join(path_to_export,f"albania_ts_{date}.gpkg")

    # Download S2 image
    if not os.path.exists(filename):
        asset_id = f"{s2data.collection_name}/{s2data.gee_id}"
        geom = s2data.geometry.intersection(area_of_interest)
        postflood = ee_image.export_image_getpixels(asset_id, geom, proj=s2data.proj,bands_gee=band_names_S2)
        save_cog(postflood, filename, descriptions=band_names_S2)
    else:
        postflood = RasterioReader(filename).load()    

    # Run inference
    if not os.path.exists(filename_pred):
        prediction_postflood, prediction_postflood_cont  = predict(postflood.values, channels = list(range(len(band_names_S2))))
        prediction_postflood_raster = GeoTensor(prediction_postflood.numpy(), transform=postflood.transform,
                                            fill_value_default=0, crs=postflood.crs)
        save_cog(prediction_postflood_raster, filename_pred, descriptions=["floodmap"])
    else:
        prediction_postflood_raster = RasterioReader(filename_pred).load().squeeze()

    # Plot 
    fig, ax = plt.subplots(1,2,figsize=(14,4.75), tight_layout=True)
    plot.show((postflood.isel({"band": [4,3,2]})/3_500).clip(0,1), ax=ax[0], add_scalebar=True)
    ax[0].set_title(f"{s2data.satellite} {s2data.solarday}")
    plot.plot_segmentation_mask(prediction_postflood_raster, COLORS_PRED, ax=ax[1],
                                interpretation_array=["invalids", "land", "water", "cloud", "flood-trace"])
    ax[1].set_title(f"{s2data.solarday} floodmap")
    plt.show()
    fig.savefig(filename_jpg)
    plt.close()

    # Vectorize the predictions
    postflood_shape = vectorize_outputv1(prediction_postflood_raster.values, 
                                         prediction_postflood_raster.crs, 
                                         prediction_postflood_raster.transform)
    floodmaps[s2data.solarday] = postflood_shape
    postflood_shape.to_file(filename_gkpg, driver='GPKG')