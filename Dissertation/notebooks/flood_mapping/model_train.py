import argparse
import os
from ml4floods.data.utils import get_filesystem
from ml4floods.models.config_setup import get_default_config
from ml4floods.models.dataset_setup import get_dataset
from pytorch_lightning import seed_everything
import pkg_resources
from ml4floods.models import dataset_setup
from ml4floods.models import worldfloods_model
import matplotlib.pyplot as plt
from ml4floods.models.model_setup import get_model, get_model_inference_function
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
import torch
import torch
import numpy as np
from ml4floods.models.utils import metrics
from ml4floods.models.utils.metrics import compute_metrics_v2
import pandas as pd
from ml4floods.models.config_setup import save_json
import time
import shutil
import rasterio
from torchvision import transforms
from torchvision.transforms import functional as F
from logger import log_system_stats, log_system_stats_to_wandb

# setx WANDB_API_KEY "1fdb73390339848661b747f02b1e4cdb81b18571"

os.environ['TMPDIR'] = 'D:/temp'

class Sentinel2Dataset(torch.utils.data.Dataset):
    def __init__(self, input_folder, target_folder, config, transform=None):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.config = config
        self.transform = transform
        self.target_size = tuple(config.data_params.window_size)

        self.input_paths = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tif')])
        self.target_paths = sorted([os.path.join(target_folder, f) for f in os.listdir(target_folder) if f.endswith('.tif')])
        assert len(self.input_paths) == len(self.target_paths), "Mismatch between input and target file counts"

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        with rasterio.open(self.input_paths[idx]) as src:
            image = src.read()
            image = F.resize(torch.tensor(image).float(), self.target_size)

        with rasterio.open(self.target_paths[idx]) as src:
            gt = src.read(1)
            gt = F.resize(torch.tensor(gt).long().unsqueeze(0), self.target_size).squeeze(0)

        # Normalize and preprocess
        if self.config['data_params']['train_transformation']['normalize']:
            image = image / 10000.0

        gt = preprocess_masks(gt)

        # One-hot encode
        num_classes = self.config['model_params']['hyperparameters']['num_classes']
        gt_one_hot = torch.nn.functional.one_hot(gt, num_classes=num_classes).permute(2, 0, 1).float()

        return {"image": image, "mask": gt_one_hot}


def get_code(x):
    """" Get CEMS code """

    bn = os.path.basename(x)
    if bn.startswith("EMSR"):
        cems_code = bn.split("_")[0]
    else:
        cems_code = os.path.splitext(bn)[0]
    return cems_code




def preprocess_masks(masks):
    # Map classes 2 and 3 to class 1
    masks[masks > 1] = 0  # Set invalid/cloud to 0
    masks[masks == 1] = 1  # Water remains 1
    return masks

def validation_step(self, batch, batch_idx):
    x, y = batch['image'], batch['mask']  # No changes needed if batch keys match
    logits = self(x)
    loss = self.loss_fn(logits, y)
    self.log("val_loss", loss, prog_bar=True)
    return loss


def train(config, dir, train_test_split_file=None, bucket_name=None):

    # Set filepath to configuration files
    # config_fp = 'path/to/worldfloods_template.json'
    # split paths

    config_fp = pkg_resources.resource_filename("ml4floods",config)
    config = get_default_config(config_fp)

    # Seed
    seed_everything(config.seed)

    config.experiment_name = 'training_demo'   
    config.data_params.batch_size = 16 # control this depending on the space on your GPU!
    config.data_params.loader_type = 'local'
    config.data_params.path_to_splits = dataset_folder # local folder to download the data
    config.data_params.train_test_split_file = None

    print(config)
    print(type(config))
    
    # Load train-test split
    try:
        if train_test_split_file:
            filenames = process_filename_train_test(
                train_test_split_file=train_test_split_file,
                input_folder="S2",
                target_folder="gt",
                bucket_id=bucket_name,
                path_to_splits=dataset_folder,
                download={"train": False, "val": False, "test": False},
            )
        else:
            filenames = {"train": {}, "val": {}, "test": {}}
        print("Train-Test split processed:", filenames.keys())
    except FileNotFoundError as e:
        print(f"Error loading train-test splits: {e}")
        return

    # Validate dataset
    try:
        dataset_setup.validate_worldfloods_data(dataset_folder)
        print("valid")
    except FileNotFoundError as e:
        print(f"data invalid {e}")

    # If files are not in config.data_params.path_to_splits this will trigger the download of the products.
    dataset = get_dataset(config.data_params)


   # Paths for train, val, and test splits
    train_input_folder = os.path.join(dir, "train", config.data_params.input_folder)
    train_target_folder = os.path.join(dir, "train", config.data_params.target_folder)

    val_input_folder = os.path.join(dir, "val", config.data_params.input_folder)
    val_target_folder = os.path.join(dir, "val", config.data_params.target_folder)

    test_input_folder = os.path.join(dir, "test", config.data_params.input_folder)
    test_target_folder = os.path.join(dir, "test", config.data_params.target_folder)

    #print(dataset)

    train_dl = dataset.train_dataloader()
    #val_loader = dataset.val_dataloader()
   
    train_dl_iter = iter(train_dl)
    batch = next(train_dl_iter)

    batch["image"].shape, batch["mask"].shape


    # show some images
    n_images=6
    fig, axs = plt.subplots(3,n_images, figsize=(18,10),tight_layout=True)
    worldfloods_model.plot_batch(batch["image"][:n_images],axs=axs[0],max_clip_val=3500.)
    worldfloods_model.plot_batch(batch["image"][:n_images],bands_show=["B11","B8", "B4"],
                                axs=axs[1],max_clip_val=4500.)
    worldfloods_model.plot_batch_output_v1(batch["mask"][:n_images, 0],axs=axs[2], show_axis=True)

    

    # model setup 
    # folder to store the trained model (it will create a subfolder with the name of the experiment)
    
    config.model_params.model_folder = "models" 
    os.makedirs("models", exist_ok=True)
    config.model_params.hyperparameters.num_classes = 2  
    config.model_params.test = False
    config.model_params.train = True
    config.model_params.hyperparameters.model_type = "unet" # Currently implemented: simplecnn, unet, linear
    print(config.model_params)
    model = get_model(config.model_params)
    print(model)
    

    # setting up weigths and bias logger (optional)
    setup_weights_and_biases = False
    if setup_weights_and_biases:

        # UNCOMMENT ON FIRST RUN TO LOGIN TO Weights and Biases (only needs to be done once)
        wandb.login()
        run = wandb.init()

        # Specifies who is logging the experiment to wandb
        config['wandb_entity'] = 'ml4floods'
        # Specifies which wandb project to log to, multiple runs can exist in the same project
        config['wandb_project'] = 'worldfloods-notebook-demo-project'

        wandb_logger = WandbLogger(
            name=config.experiment_name,
            project=config.wandb_project, 
            entity=config.wandb_entity
        )
    else:
        wandb_logger = None


    # lightning callbacks to act as checkpoints for the model 
    experiment_path = f"{config.model_params.model_folder}/{config.experiment_name}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{experiment_path}/checkpoint",
        save_top_k=True,
        verbose=True,
        monitor='val_bce_loss',
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_bce_loss',
        patience=10,
        strict=False,
        verbose=False,
        mode='min'
    )

    callbacks = [checkpoint_callback, early_stop_callback]

    print(f"The trained model will be stored in {config.model_params.model_folder}/{config.experiment_name}")

    #setup lightning trainer
    config.gpus = '0'  # which gpu to use
    # config.gpus = None # to not use GPU

    config.model_params.hyperparameters.max_epochs = 4 # train for maximum 4 epochs

   # Initialize datasets
    train_dataset = Sentinel2Dataset(input_folder=os.path.join(dir, "train", config.data_params.input_folder),
                                     target_folder=os.path.join(dir, "train", config.data_params.target_folder),
                                     config=config)

    val_dataset = Sentinel2Dataset(input_folder=os.path.join(dir, "val", config.data_params.input_folder),
                                   target_folder=os.path.join(dir, "val", config.data_params.target_folder),
                                   config=config)
    
    test_dataset = Sentinel2Dataset(input_folder=os.path.join(dir, "test", config.data_params.input_folder),
                                    target_folder=os.path.join(dir, "test", config.data_params.target_folder),
                                    config=config)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.data_params.batch_size, 
        shuffle=True, # changing this = bad check metric 53
        num_workers=4,  # Increase for performance
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.data_params.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.data_params.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Validate data loading and check batch shapes
    for batch in train_loader:
        print(f"Image shape: {batch['image'].shape}, Mask shape: {batch['mask'].shape}")
        break


    trainer = Trainer(
        precision=16,
        fast_dev_run=False,
        logger=wandb_logger,
        callbacks=callbacks,
        #default_root_dir=f"{config.model_params.model_folder}/{config.experiment_name}",
        accumulate_grad_batches=1,
        gradient_clip_val=0.0,
        benchmark=False,
        devices = 1,
        accelerator="gpu",
        max_epochs=config['model_params']['hyperparameters']['max_epochs'],
        #check_val_every_n_epoch=1
        #strategy="ddp"
    )

    wandb.init()

     # Automate W&B login
    wandb_api_key = os.getenv("WANDB_API_KEY")  # Fetch API key from the environment variable
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        print("W&B API Key not found. Set WANDB_API_KEY as an environment variable.")
        exit(1)  # Exit if the API key is not provided

    # Log system stats after each epoch
    for epoch in range(config['model_params']['hyperparameters']['max_epochs']):
        print(f"Starting epoch {epoch + 1}/{config['model_params']['hyperparameters']['max_epochs']}...")
        log_system_stats()  # Log to file
        log_system_stats_to_wandb()  # Log to WandB (if required)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
# ======================================================
    # SAVING SETUP 
    # ======================================================
    print("======================================================")
    print("FINISHED TRAINING, SAVING MODEL")
    print("======================================================")

    for batch in test_loader:
        print(f"Test Image shape: {batch['image'].shape}, Mask shape: {batch['mask'].shape}")
        break

    fs = get_filesystem(experiment_path)
    path_save_model = os.path.join(experiment_path, "model.pt").replace("\\","/")

    # More details can be found here: https://github.com/pytorch/pytorch/issues/42239
    with fs.open(path_save_model, "wb") as fh:
        torch.save(model.state_dict(), fh, _use_new_zipfile_serialization=False)


    wandb_run = wandb.init()
    if wandb_run is not None:
        # Save the model in W&B's run directory
        
        destination_path = os.path.join(wandb_run.dir, "model.pt")
        shutil.copy(path_save_model, destination_path) 
        wandb.save(destination_path)
    else:
        print("Failed to initialize W&B run. Model not logged to W&B.")
    wandb.finish()

    # Save cofig file in experiment_path
    config_file_path = os.path.join(experiment_path, "config.json").replace("\\","/")
    save_json(config_file_path, config)

    #display some inferences
    logits = model(batch["image"].to(model.device))
    print(f"Shape of logits: {logits.shape}")
    probs = torch.softmax(logits, dim=1)
    print(f"Shape of probs: {probs.shape}")
    prediction = torch.argmax(probs, dim=1).long().cpu()
    print(f"Shape of prediction: {prediction.shape}")

    n_images=6
    fig, axs = plt.subplots(4, n_images, figsize=(18,14),tight_layout=True)
    worldfloods_model.plot_batch(batch["image"][:n_images],axs=axs[0],max_clip_val=3500.)
    worldfloods_model.plot_batch(batch["image"][:n_images],bands_show=["B11","B8", "B4"],
                                axs=axs[1],max_clip_val=4500.)
    worldfloods_model.plot_batch_output_v1(batch["mask"][:n_images, 0],axs=axs[2], show_axis=True)
    worldfloods_model.plot_batch_output_v1(prediction[:n_images] + 1,axs=axs[3], show_axis=True)
    # Prevent plot from closing
    plt.show(block=True)

    # Optionally save the figure to file
    fig.savefig(os.path.join(experiment_path, "inference_images.png"))

    for ax in axs.ravel():
        ax.grid(False)


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


    for dl, dl_name in [(test_loader, "test"), (val_loader, "val")]:
        metrics_file = os.path.join(config.model_params.model_folder, f"{dl_name}_metrics.json")
        inference_function = get_model_inference_function(model, config, activation='sigmoid', device='cuda')

        mets = compute_metrics_v2(dl, inference_function, threshold_water=0.5, plot=True, mask_clouds=True)
        save_json(metrics_file, mets)

        Label_Names = ["land", "water"]
        iou_plot_path = os.path.join(config.model_params.model_folder, f"{dl_name}_iou_plot.png")
        recall_plot_path = os.path.join(config.model_params.model_folder, f"{dl_name}_recall_plot.png")
        metrics.plot_metrics(mets, Label_Names)
        plt.savefig(iou_plot_path)
        plt.savefig(recall_plot_path)

        wandb.log({
            f"{dl_name}_iou": mets.get("iou"),
            f"{dl_name}_recall": mets.get("recall"),
            f"{dl_name}_IoU_vs_Threshold": wandb.Image(iou_plot_path),
            f"{dl_name}_Recall_vs_Threshold": wandb.Image(recall_plot_path)
        })



if __name__ == '__main__':

    # Unzip the data
    dataset_folder = r'D:\Documents\coding\300\ml4floods\notebooks\flood_mapping\worldfloods_v1_0_sample'
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--config', default = 'models/configurations/worldfloods_template_v2.json')
    parser.add_argument("--split", help="Train-Test split file path", default=None)
    parser.add_argument("--bucket", help="Cloud bucket ID", default=None)
    args = parser.parse_args()
    train(args.config, dataset_folder, train_test_split_file=args.split, bucket_name=args.bucket)

#"models/configurations/worldfloods_template_v2.json"

#legacy
"""
    for dl, dl_name in [(test_loader, "test"), (val_loader, "val")]:
        metrics_file = os.path.join(experiment_path, f"{dl_name}.json").replace("\\","/")

        if fs.exists(metrics_file):
            print(f"File {metrics_file} exists. Continue")
            continue
        
        #thresholds_water = [0,1e-3,1e-2]+np.arange(0.5,.96,.05).tolist() + [.99,.995,.999]
        mets = metrics.compute_metrics_v2(
            dl,
            inference_function, 
            threshold_water=0.5, #was 0.5 (binary threshold)
            plot=True,
            mask_clouds=True)
        
            # Save metrics as JSON
        save_json(metrics_file, mets)
        Label_Names = ["land", "water"]

        # Log metrics to W&B
        if "iou" in mets and "recall" in mets:
            wandb.log({
                f"{dl_name}_iou": mets["iou"],
                f"{dl_name}_recall": mets["recall"]
            })

        # Plot metrics
        iou_plot_path = os.path.join(experiment_path, f"{dl_name}_iou_plot.png")
        recall_plot_path = os.path.join(experiment_path, f"{dl_name}_recall_plot.png")
        
        metrics.plot_metrics(mets, Label_Names)
        plt.savefig(iou_plot_path)
        plt.savefig(recall_plot_path)

        # Log plots to W&B
        wandb.log({
            f"{dl_name}_IoU_vs_Threshold": wandb.Image(iou_plot_path),
            f"{dl_name}_Recall_vs_Threshold": wandb.Image(recall_plot_path)
        })

        if hasattr(dl.dataset, "image_files"):
            mets["cems_code"] = [get_code(f) for f in dl.dataset.image_files]
        elif hasattr(dl.dataset, "list_of_windows"):
            mets["cems_code"] = [get_code(f.file_name) for f in dl.dataset.list_of_windows]
        else:
            mets["cems_code"] = []

        iou_per_code = pd.DataFrame(metrics.group_confusion(
            mets["confusions"],
            mets["cems_code"], 
            metrics.calculate_iou,
            label_names=[f"IoU_{l}"for l in Label_Names]
            ))

        recall_per_code = pd.DataFrame(metrics.group_confusion(
            mets["confusions"],
            mets["cems_code"], 
            metrics.calculate_recall,
            label_names=[f"Recall_{l}"for l in Label_Names]
            ))

        join_data_per_code = pd.merge(recall_per_code,iou_per_code,on="code").set_index("code")*100
        print(f"Mean values across flood events: {join_data_per_code.mean(axis=0).to_dict()}")
        print(join_data_per_code)
"""