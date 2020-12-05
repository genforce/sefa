# python3.7
"""Contains the synchronizing operator."""

import torch
import torch.distributed as dist

__all__ = ['all_gather']


def all_gather(tensor):
    """Gathers tensor from all devices and does averaging."""
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    tensor_list = [torch.ones_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, async_op=False)
    return torch.mean(torch.stack(tensor_list, dim=0), dim=0)
