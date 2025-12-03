import os
from huggingface_hub import hf_hub_download, hf_hub_url

import torch
import numpy as np
from shapely.geometry import shape
from georeader.georeader.rasterio_reader import RasterioReader
from georeader.georeader.geotensor import GeoTensor
import rasterio
import ee
from ml4floods.visualization import plot_utils
import matplotlib.pyplot as plt
from georeader.georeader import plot

# import os
# from huggingface_hub import hf_hub_download, hf_hub_url

# import torch
# import numpy as np
# from shapely.geometry import shape

# from georeader.rasterio_reader import RasterioReader
# from georeader.geotensor import GeoTensor
# from georeader.readers import ee_query
# from georeader.readers import ee_image
# from datetime import datetime
# from georeader import plot

COLORS_PRED = np.array([[0, 0, 0], # 0: invalid
                       [139, 64, 0], # 1: land
                       [0, 0, 240], # 2: water
                       [220, 220, 220], # 3: cloud
                       [60, 85, 92]], # 5: flood_trace
                    dtype=np.float32) / 255



# experiment_name = "WF2_unetv2_bgriswirs"
subfolder_local = f"/home/zhangz65/NASA_model/Worldfloods/multioutput/experiments/U_net/models/training_v2_unet_hec_whole_data"
# config_file = hf_hub_download(repo_id="isp-uv-es/ml4floods",subfolder=subfolder_local, filename="config.json",
#                               local_dir=".", local_dir_use_symlinks=False)
# model_file = hf_hub_download(repo_id="isp-uv-es/ml4floods",subfolder=subfolder_local, filename="model.pt",
#                               local_dir=".", local_dir_use_symlinks=False)


from worldfloods.inference import load_inference_function
from ml4floods.models.model_setup import get_channel_configuration_bands

inference_function, config = load_inference_function(subfolder_local, device_name = 'cuda', max_tile_size=128,
                                                     th_water=0.7, th_brightness=3500,
                                                     distinguish_flood_traces=False)

channel_configuration = config['data_params']['channel_configuration']
channels  = get_channel_configuration_bands(channel_configuration, collection_name='S2')

def predict(input_tensor, channels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
    input_tensor = input_tensor.astype(np.float32)
    input_tensor = input_tensor[channels]
    torch_inputs = torch.tensor(np.nan_to_num(input_tensor))
    return inference_function(torch_inputs)


tiff_s2 = '/home/zhangz65/NASA_model/Worldfloods/dataset_v2/test/S2' # swithc when training on hec
label = '/home/zhangz65/NASA_model/Worldfloods/dataset_v2/test/gt' # swithc when training on hec

contents = os.listdir(tiff_s2)

patch_size = 256



for index, each in enumerate(contents):

    imgdir = os.path.join(tiff_s2,each)

    s2rst = RasterioReader(imgdir).isel({"band": channels})
    s2rst = s2rst.load()
    prediction_test, prediction_test_cont  = predict(s2rst.values, channels = list(range(len(channels))))
    prediction_test_raster = GeoTensor(prediction_test.numpy(), transform=s2rst.transform,
                                   fill_value_default=0, crs=s2rst.crs)
    fig, ax = plt.subplots(1,2,figsize=(14,7), sharey=True)

    plot.show((s2rst.isel({"band": [4,3,2]})/3_500).clip(0,1), ax=ax[0], add_scalebar=True)
    # ax[0].set_title(f"{subset}/{filename}")


    plot.plot_segmentation_mask(prediction_test_raster, COLORS_PRED, ax=ax[1],
                                interpretation_array=["invalids", "land", "water", "cloud", "flood_trace"])
    
    if not os.path.exists('/home/zhangz65/NASA_model/Worldfloods/multioutput/experiments/U_net/images'):
        os.makedirs('/home/zhangz65/NASA_model/Worldfloods/multioutput/experiments/U_net/images')

    plt.savefig(f'/home/zhangz65/NASA_model/Worldfloods/multioutput/experiments/U_net/images/{index}.png')

