import torch
import wandb
import argparse
import os

from ml4floods.data.utils import get_filesystem
from ml4floods.models.config_setup import save_json
from pytorch_lightning import seed_everything
from dataset_setup import get_dataset
from ml4floods.models.utils.metrics import compute_metrics_v2
from ml4floods.models.model_setup import get_model, get_model_inference_function
from ml4floods.models import worldfloods_model
from ml4floods.models.config_setup import setup_config
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from ml4floods.models.config_setup import get_default_config
from pytorch_lightning import Trainer
import numpy as np
import pkg_resources


def get_code(x):
    """" Get CEMS code """

    bn = os.path.basename(x)
    if bn.startswith("EMSR"):
        cems_code = bn.split("_")[0]
    else:
        cems_code = os.path.splitext(bn)[0]
    return cems_code


def train(config,tmp_dir):


    # import shutil

    # src = '/mmfs1/scratch/hpc/00/zhangz65/satlas/code_worldfloods/whole_data/satlas_landsat/acclerate_learnig_rate_test/dataset_v2_no_cache.zip'

    # dst = tmp_dir

    # shutil.copy2(src, dst)



    # try:
    #     from google.colab import drive
    #     drive.mount('/content/drive')
    #     public_folder = '/content/drive/My Drive/Public WorldFloods Dataset'
    #     assert os.path.exists(public_folder), "Add a shortcut to the publice Google Drive folder: https://drive.google.com/drive/u/0/folders/1dqFYWetX614r49kuVE3CbZwVO6qHvRVH"
    #     google_colab = True
    # except ImportError as e:
    #     print(e)
    #     print("Setting google colab to false, it will need to install the gdown package!")
    #     public_folder = tmp_dir
    #     google_colab = False

    # from ml4floods.models import dataset_setup
    # import zipfile

    # # Unzip the data
    # path_to_dataset_folder = tmp_dir
    # dataset_folder = os.path.join(path_to_dataset_folder, "dataset_v2_no_cache")

    # try:
    #     dataset_setup.validate_worldfloods_data(dataset_folder)
    # except FileNotFoundError as e:
    #     print(e)
    #     zip_file_name = os.path.join(public_folder, "dataset_v2_no_cache.zip") # this file size is 12.7Gb

    #     print("We need to unzip the data")
    #     # Download the zip file
    #     if not os.path.exists(zip_file_name):
    #         print("Download the data from Google Drive")
    #         import gdown
    #         # https://drive.google.com/file/d/11O6aKZk4R6DERIx32o4mMTJ5dtzRRKgV/view?usp=sharing
    #         gdown.download(id="11O6aKZk4R6DERIx32o4mMTJ5dtzRRKgV", output=zip_file_name)

    #     print("Unzipping the data")
    #     with zipfile.ZipFile(zip_file_name, "r") as zip_ref:
    #         zip_ref.extractall(path_to_dataset_folder)
    #         zip_ref.close()


    # Set filepath to configuration files
    # config_fp = 'path/to/worldfloods_template.json'

    dataset_folder = tmp_dir  # swithc when training on hec
    config_fp = pkg_resources.resource_filename("ml4floods",config)

    config = get_default_config(config_fp)

    print(config)


    # Seed
    seed_everything(config.seed)

    config.experiment_name = 'training_v2_unet_hec_whole_data'
    # dataset_folder = '/home/zhangz65/NASA_model/Worldfloods/dataset_v2' # swithc when training on hec


    config.data_params.batch_size = 16 # control this depending on the space on your GPU!
    config.data_params.loader_type = 'local'
    config.data_params.path_to_splits = dataset_folder # local folder to download the data
    config.data_params.train_test_split_file = None

    # If files are not in config.data_params.path_to_splits this will trigger the download of the products.
    dataset = get_dataset(config.data_params)

    train_dl = dataset.train_dataloader()

    train_dl_iter = iter(train_dl)
    batch = next(train_dl_iter)

    print(batch["image"].shape, batch["mask"].shape)

    print(config.model_params)


    config.model_params.model_folder = "models" 
    os.makedirs("models", exist_ok=True)
    config.model_params.test = False
    config.model_params.train = True
    config.model_params.hyperparameters.model_type = "unet" # Currently implemented: simplecnn, unet, linear
    model = get_model(config.model_params)
    print(model)

    setup_weights_and_biases = True

    if setup_weights_and_biases:


        # UNCOMMENT ON FIRST RUN TO LOGIN TO Weights and Biases (only needs to be done once)
        # wandb.login()
        # run = wandb.init()

        # Specifies who is logging the experiment to wandb

        config['wandb_entity'] = 'falaliku'
        # Specifies which wandb project to log to, multiple runs can exist in the same project
        config['wandb_project'] = 'worldfloods'

        wandb_logger = WandbLogger(
            name=config.experiment_name,
            project=config.wandb_project, 
            entity=config.wandb_entity
        )
    else:
        wandb_logger = None



    experiment_path = f"{config.model_params.model_folder}/{config.experiment_name}"

    print(f"The trained model will be stored in {experiment_path}"
            f" and the best model will be stored in {experiment_path}/checkpoint")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{experiment_path}/checkpoint",
        save_top_k=True,
        verbose=True,
        monitor='val_bce_land_water',
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_bce_land_water',
        patience=10,
        strict=False,
        verbose=False,
        mode='min'
    )

    callbacks = [checkpoint_callback, early_stop_callback]

    print(f"The trained model will be stored in {config.model_params.model_folder}/{config.experiment_name}")


    from pytorch_lightning import Trainer

    # config.gpus = '0'  # which gpu to use
    # config.gpus = None # to not use GPU

    config.model_params.hyperparameters.max_epochs = 4 # train for maximum 4 epochs

    trainer = Trainer(
        fast_dev_run=False,
        logger=wandb_logger,
        callbacks=callbacks,
        # default_root_dir=f"{config.model_params.model_folder}/{config.experiment_name}",
        accumulate_grad_batches=1,
        gradient_clip_val=0.0,
        benchmark=False,
        devices = 3,
        accelerator="gpu",
        max_epochs=10,
        check_val_every_n_epoch=1,
        strategy="ddp",
    )

    trainer.fit(model, dataset)


    # ======================================================
    # SAVING SETUP 
    # ======================================================
    print("======================================================")
    print("FINISHED TRAINING, SAVING MODEL")
    print("======================================================")
    fs = get_filesystem(experiment_path)
    path_save_model = os.path.join(experiment_path, "model.pt").replace("\\","/")

    # More details can be found here: https://github.com/pytorch/pytorch/issues/42239
    with fs.open(path_save_model, "wb") as fh:
        torch.save(model.state_dict(), fh, _use_new_zipfile_serialization=False)

    wandb.init()
    wandb.save(path_save_model)
    wandb.finish()

    # Save cofig file in experiment_path
    config_file_path = os.path.join(experiment_path, "config.json").replace("\\","/")
    save_json(config_file_path, config)

    config["model_params"]["max_tile_size"] = 512

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    # Compute metrics in test and val datasets
    if config.model_params.get("model_version", "v1") == "v2":
        inference_function = get_model_inference_function(model, config, apply_normalization=False,
                                                            activation='sigmoid',device='cuda')
    else:
        inference_function = get_model_inference_function(model, config, apply_normalization=False,
                                                            activation='softmax',device = 'cuda')


    for dl, dl_name in [(dataset.test_dataloader(), "test"), (dataset.val_dataloader(), "val")]:
        metrics_file = os.path.join(experiment_path, f"{dl_name}.json").replace("\\","/")
        if fs.exists(metrics_file):
            print(f"File {metrics_file} exists. Continue")
            continue
        mets = compute_metrics_v2(
            dl,
            inference_function, threshold_water=0.5,
            plot=False,
            mask_clouds=True)

        if hasattr(dl.dataset, "image_files"):
            mets["cems_code"] = [get_code(f) for f in dl.dataset.image_files]
        else:
            mets["cems_code"] = [get_code(f.file_name) for f in dl.dataset.list_of_windows]

        save_json(metrics_file, mets)

if __name__ == '__main__':
    # setting working directory
    tmp_dir = os.getenv('TMPDIR')
    os.environ['TORCH_HOME'] = tmp_dir #setting the environment variable
    os.system('huggingface-cli download --cache-dir /tmp --local-dir /tmp --repo-type dataset isp-uv-es/WorldFloodsv2')
    # dataset_folder = '/home/zhangz65/NASA_model/Worldfloods/dataset_v2' # swithc when training on hec
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--config', default = 'models/configurations/worldfloods_template_v2.json')
    args = parser.parse_args()
    train(args.config, tmp_dir)

