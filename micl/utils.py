import os
from pathlib import Path
from typing import Union

import torch
from torchmetrics.utilities.distributed import gather_all_tensors
from transformers import PreTrainedTokenizerFast


def box_cxcywh_to_xyxy(x):
    if not x.numel():
        return torch.Tensor([])
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    if not x.numel():
        return torch.Tensor([])
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def get_matched_boxes(pred_boxes: torch.Tensor, indices: torch.Tensor):
    result = []
    
    for b_idx, idx_list in enumerate(indices):
        result.append(pred_boxes[b_idx, idx_list])  # Gather values from pred using batch index and sublist of indices
    
    return torch.cat(result)  # Concatenate all the gathered values into a single tensor


def get_flattened_pred_logits(pred_logits: torch.Tensor, indices: torch.Tensor, image_indices: torch.Tensor):
    """
    Given a tensor pred_logits of shape (B, N, M), a list of list of indices, and a list of image_indices,
    return two tensors:
    1. A flattened tensor of length M of positive matched logits.
    2. A flattened tensor of all elements from pred_logits that were not included in the first tensor (negative logits).
    
    Arguments:
    - pred_logits (torch.Tensor): Tensor of shape (B, N, M).
    - indices (list of list of ints): List of lists, where each sublist contains indices in the second dimension (N).
    - image_indices (list of ints): List of length M with batch indices in [0, B-1].
    """
    B, N, M = pred_logits.shape
    flatten_indices = [i for sublist in indices for i in sublist]
    mask = torch.zeros((B, N, M), dtype=torch.bool, device=pred_logits.device)
    
    # Set the positions with matched indices
    for m in range(M):
        batch_idx = image_indices[m]
        mask[batch_idx, flatten_indices[m], m] = 1
    positive_logits = pred_logits[mask].view(M)  # The selected elements are flattened to size M
    negative_logits = pred_logits[~mask].view(-1)
    return positive_logits, negative_logits


def get_dummy_tokens(class_strings: list[str], tokenizer=None, device='cuda'):
    if tokenizer is None:
        model_dirpath = Path('/home/jovyan/models/mimiccxr-clip-vit-large-patch14-336')
        tokenizer = PreTrainedTokenizerFast(
            bos_token='[BOS]',
            eos_token='[EOS]',
            unk_token='[UNK]',
            sep_token='[SEP]',
            pad_token='[PAD]',
            cls_token='[CLS]',
            mask_token='[MASK]',
            tokenizer_file=str(model_dirpath / 'tokenizer.json')
        )
    tokenized_data = tokenizer(
        class_strings,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    return tokenized_data['input_ids'].to(device), tokenized_data['attention_mask'].to(device)


def get_positive_and_negative_indices(shape, image_indices, flattened_matched_indices):
    image_indices = image_indices.to(torch.int)

    all_batch_indices = torch.arange(shape[0]).repeat_interleave(shape[1])
    all_query_indices = torch.arange(shape[1]).repeat(shape[0])
    all_indices = torch.stack([all_batch_indices, all_query_indices], dim=1).to(image_indices.device)

    # Selected indices
    selected_indices = torch.stack([image_indices, flattened_matched_indices], dim=1)
    mask = (all_indices[:, None] == selected_indices).all(dim=2).any(dim=1)

    positive_indices = all_indices[mask]
    negative_indices = all_indices[~mask]
    return positive_indices, negative_indices


def remove_duplicates(string_labels, labels):
    unique_strs = []
    indices = []
    corresponding_labels = []
    for string, label in zip(string_labels, labels):
        if string not in unique_strs:
            unique_strs.append(string)
            corresponding_labels.append(label)
        indices.append(unique_strs.index(string))
    return unique_strs, corresponding_labels, indices


def last_checkpoint(root: Path | str) -> Union[Path | str]:
    """
    Load the most fresh ckpt file based on time.
    Parameters
    ----------
    root: Union[Path, str]
        Path to folder, where ckpt or its symbolic link supposed to be.
    Returns
    -------
    checkpoint_path: Union[Path, str]
        If ckpt exists - returns Path to it. Otherwise, returns 'last'.
    """
    print("LOADING last_checkpoint", root)
    checkpoints = []
    for p in Path(root).rglob('*'):
        if p.is_symlink():
            p = p.resolve(strict=False)
            if p.suffix == '.ckpt':
                checkpoints.append(p)
        elif p.suffix == '.ckpt':
            checkpoints.append(p)

    if not checkpoints:
        return None
    checkpoint = max(checkpoints, key=lambda t: os.stat(t).st_mtime)
    print("Checkpoint", checkpoint, end='\n\n')
    return checkpoint


def load_pretrained_ckpt(model, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {key[6:]: value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)


def custom_dist_sync_fn(result, group=None):
    # Ensure the tensor is on GPU (NCCL only supports CUDA tensors)
    if not result.is_cuda:
        result = result.to(torch.device('cuda'))
    return gather_all_tensors(result, group=group)
