import torch

def clip(x, min, max):
    return (x >= max) * max + (x <= min) * min + (x <= max * x >= min) * x
