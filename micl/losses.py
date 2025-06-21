import torch
import torch.nn.functional as F
from torch.nn import functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Tensor of shape (num_classes,) or None
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape (batch_size, num_classes) - raw output before softmax
            targets: Tensor of shape (batch_size,) - class indices
        
        Returns:
            Loss value (scalar)
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)

        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()
        pt = (probs * targets_one_hot).sum(dim=1)  # Get p_t

        focal_factor = (1 - pt) ** self.gamma
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        focal_loss = focal_factor * ce_loss

        if self.alpha is not None:
            alpha_weights = self.alpha.to(logits.device)[targets]
            focal_loss = alpha_weights * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError(f"`reduction` should be in ['mean', 'sum', 'loss'], got {self.reduction}")


def generalized_cross_entropy(pred_logits, target_labels):
    pred_logits = pred_logits.view(-1, pred_logits.shape[-1])
    target_labels = target_labels.view(-1, target_labels.shape[-1])
    content_mask = ~torch.all(target_labels == -1, dim=-1)
    masked_logits = torch.where(
        target_labels[content_mask] == 1,
        pred_logits[content_mask],
        torch.tensor(-float('inf'),device=pred_logits.device)
    )
    logsumexp_pos = torch.logsumexp(masked_logits, dim=-1)
    logsumexp_all = torch.logsumexp(pred_logits, dim=-1)
    return -(logsumexp_pos - logsumexp_all).sum() / torch.sum(content_mask)


def masked_background_bce_loss(pred_logits, indices_list):
    """
    Computes BCE loss for each batch, excluding indices from indices_list.

    Args:
        pred_logits: (batch_size, n_queries, n_classes) - Model predictions
        indices_list: List of np.arrays, each containing indices to exclude per batch

    Returns:
        loss: Scalar tensor with the masked BCE loss
    """
    batch_size, n_queries, _ = pred_logits.shape
    device = pred_logits.device

    mask = torch.ones((batch_size, n_queries), dtype=torch.bool, device=device)
    if not torch.any(mask):
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Fill mask with False for matched queries (should not be background)
    for b_idx in range(batch_size):
        mask[b_idx, torch.tensor(indices_list[b_idx], dtype=torch.long, device=device)] = False
        
    logsumexp_all = torch.logsumexp(pred_logits, dim=-1)  # (batch_size, n_queries)
    logsumexp_all = logsumexp_all[mask]  # Shape: (-1,)
    
    # Extract last class logits (background class), only for masked queries
    last_logit = pred_logits[..., -1][mask]  # Shape: (-1,)
    loss = -(last_logit - logsumexp_all).mean()
    return loss
