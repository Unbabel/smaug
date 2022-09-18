import logging
import torch


def use_gpu(no_gpu: bool) -> bool:
    use_gpu = not no_gpu
    if use_gpu and not torch.cuda.is_available():
        logging.warn("GPU requested but not available. Disabling GPU.")
        use_gpu = False
    return use_gpu
