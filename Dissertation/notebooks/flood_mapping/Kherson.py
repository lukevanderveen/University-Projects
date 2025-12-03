import ee
ee.Authenticate()
ee.Initialize()
from datetime import datetime, timezone
from georeader.readers import ee_query
from shapely.geometry import shape 
import geopandas as gpd
from georeader.readers import S2_SAFE_reader
from georeader.save import save_cog
from georeader import window_utils, mosaic
from georeader.plot import show
from georeader.rasterio_reader import RasterioReader
import os
import torch
import numpy as np
from georeader.geotensor import GeoTensor
from ml4floods.scripts.inference import load_inference_function, vectorize_outputv1
from ml4floods.models.model_setup import get_channel_configuration_bands
from ml4floods.data import utils
import warnings
from rasterio.plot import show
from georeader.readers import ee_image
from ml4floods.models import postprocess
from georeader import plot
import matplotlib.pyplot as plt
from ml4floods.visualization import plot_utils

aoi = shape({'type': 'Polygon',
 'coordinates': (((33.40965055141422, 46.849975215311474),
   (33.24671826582107, 46.923511440491325),
   (32.936224664974134, 46.845770100334164),
   (32.33368262768653, 46.62876156455022),
   (32.25990197005967, 46.514641087646424),
   (32.31216326921171, 46.408759851523826),
   (32.843998842939385, 46.56961795883814),
   (33.21905051921081, 46.72367854887557),
   (33.40965055141422, 46.849975215311474)),)})

aoi_gpd = gpd.GeoDataFrame({'geometry':aoi},index = [0]).set_crs('epsg:4326')

tz = timezone.utc
start_period = datetime.strptime('2023-05-31',"%Y-%m-%d").replace(tzinfo=tz)
end_period = datetime.strptime('2023-06-12',"%Y-%m-%d").replace(tzinfo=tz)

# This function returns a GEE collection of Sentinel-2 and Landsat 8 data and a Geopandas Dataframe with data related to the tiles, overlap percentage and cloud cover
flood_images_gee, flood_collection = ee_query.query(
    area=aoi, 
    date_start=start_period, 
    date_end=end_period,                                                   
    producttype="both", 
    return_collection=True, 
    add_s2cloudless=False)

flood_images_gee.groupby(["solarday","satellite"])[["cloudcoverpercentage","overlappercentage"]].agg(["count","mean"])