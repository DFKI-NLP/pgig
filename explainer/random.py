from .backprop import VanillaGradExplainer
import torch

class RandomExplainer(VanillaGradExplainer):
    def __init__(self, model):
        super().__init__(model)

    def explain(self, inp, ind=None, pattern_method=None):
        return torch.rand_like(inp)
