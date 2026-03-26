import torch
def r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes the R² (coefficient of determination) metric.

    Args:
        predictions (torch.Tensor): Predicted values tensor.
        targets (torch.Tensor): Actual target values tensor.

    Returns:
        float: R² score representing how well the predictions approximate the targets.
    """
    # Compute the total sum of squares (variance of the target)
    target_mean = torch.mean(targets)
    total_variance = torch.sum((targets - target_mean) ** 2)
    
    # Compute the residual sum of squares128
    residual_variance = torch.sum((targets - predictions) ** 2)
    
    # Calculate the R² score
    r2 = 1 - (residual_variance / total_variance)
    
    res = r2.detach()
    return res

def accuracy(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Computes accuracy for binary classification.
    Shape-agnostic (works for any ND tensor).
    """
    with torch.no_grad():
        # Flatten and binarize
        preds = (predictions > threshold).float().view(-1)
        labels = targets.float().view(-1)
        
        # Compare and average
        correct = (preds == labels).float().mean()
        
        return correct.detach()

def f1_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Computes the F1 score for binary classification.
    Shape-agnostic (works for any ND tensor).
    """
    with torch.no_grad():
        # Flatten to 1D to handle any input shape
        preds = (predictions > threshold).float().view(-1)
        labels = targets.float().view(-1)
        
        # Calculate components
        tp = (preds * labels).sum()
        fp = (preds * (1 - labels)).sum()
        fn = ((1 - preds) * labels).sum()
        
        # Calculate precision and recall with epsilon to avoid div by zero
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (fn + tp + 1e-7)
        
        # Harmonic mean
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        return f1.detach()
    
def iou_metric(pred, target, threshold=0.5):
    with torch.no_grad():
        pred = (pred> threshold).float()  # Convert to binary
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = intersection / (union + 1e-6)
        return iou.cpu().item()
