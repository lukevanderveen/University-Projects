import torch
from ml4floods.models.utils.metrics import compute_metrics

def evaluate_model(model, test_loader, device="cuda"):
    model.eval()
    model = model.to(device)

    # Create prediction function
    def pred_fun(x):
        return model(x.to(device))
    
    # Use compute_metrics with the dataloader and prediction function
    metrics = compute_metrics(
        dataloader=test_loader,
        pred_fun=pred_fun,
        threshold=0.5,
        plot=False,
        mask_clouds=False
    )
    
    return metrics

def print_evaluation_results(metrics):
    print("\nTest Set Evaluation Results:")
    print("=" * 50)
    print("\nIoU Scores:")
    print(f"Water: {metrics['iou_water']:.4f}")
    print(f"Land: {metrics['iou_land']:.4f}")
    print(f"Cloud: {metrics['iou_cloud']:.4f}")
    
    print("\nPrecision Scores:")
    print(f"Water: {metrics['precision_water']:.4f}")
    print(f"Land: {metrics['precision_land']:.4f}")
    print(f"Cloud: {metrics['precision_cloud']:.4f}")
    
    print("\nRecall Scores:")
    print(f"Water: {metrics['recall_water']:.4f}")
    print(f"Land: {metrics['recall_land']:.4f}")
    print(f"Cloud: {metrics['recall_cloud']:.4f}")
    
    print("\nF1 Scores:")
    print(f"Water: {metrics['f1_water']:.4f}")
    print(f"Land: {metrics['f1_land']:.4f}")
    print(f"Cloud: {metrics['f1_cloud']:.4f}")
    print("=" * 50)
