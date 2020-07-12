import torch
import numpy as np
import types
from .backprop import VanillaGradExplainer


def pattern_augment(exp: VanillaGradExplainer, method='pattern_net'):
    patterns = load_patterns()['A']

    # According to PA paper
    exp.backprop_probability = True

    # Replace base explainer's methods
    exp.patterns = patterns
    # Create fill_in_patterns method in in explainer based on pattern method
    if method == 'pattern_net':
        exp.fill_in_patterns = types.MethodType(fill_in_pattern_net, exp)
    elif method == 'pattern_attribution':
        exp.fill_in_patterns = types.MethodType(fill_in_pattern_attribution, exp)

    # Patch backprop and forward_pass in explainer to use patterns
    exp.backprop = types.MethodType(pattern_backprop(exp.backprop), exp)
    exp.forward_pass = types.MethodType(pattern_forward_pass(exp.forward_pass), exp)
    return exp


def load_patterns(filename='./weights/imagenet_224_vgg_16.patterns.A_only.npz'):
    f = np.load(filename)
    ret = {}
    for prefix in ["A", "r", "mu"]:
        l = sum([x.startswith(prefix) for x in f.keys()])
        ret.update({prefix: [f["%s_%i" % (prefix, i)] for i in range(l)]})
    return ret


def fill_in_pattern_net(self):
    for i in range(0, 26, 2):
        self.weights[i].data.copy_(
            torch.from_numpy(self.patterns[int(i / 2)]).cuda()
        )
    for i in range(26, 32, 2):
        self.weights[i].data.copy_(
            torch.from_numpy(self.patterns[int(i / 2)].transpose()).cuda()
        )


def fill_in_pattern_attribution(self, pos=True):
    if pos:
        for i in range(0, 26, 2):
            self.weights[i].data.copy_(
                self.weights[i].data *
                torch.from_numpy(self.patterns[int(i / 2)]).cuda()
            )
        for i in range(26, 32, 2):
            self.weights[i].data.copy_(
                self.weights[i].data *
                torch.from_numpy(self.patterns[int(i / 2)].transpose()).cuda()
            )
    else:
        for i in range(0, 26, 2):
            self.weights[i].data.copy_(
                self.weights[i].data *
                torch.from_numpy(self.patterns[int(i / 2)]).cuda()
            )
        for i in range(26, 32, 2):
            self.weights[i].data.copy_(
                self.weights[i].data *
                torch.from_numpy(self.patterns[int(i / 2)].transpose()).cuda()
            )


def pattern_backprop(exp_backprop):
    """Returns a function that fills in patterns and then calls explainer's original backprop function"""
    def f(self, inp, output, grad_out):
        self.fill_in_patterns()
        return exp_backprop(inp, output, grad_out)
    return f


def pattern_forward_pass(exp_forward):
    """Returns a function that fills in weights and then calls explainer's original forward function"""
    def p(self, inp, ind):
        self.fill_in_pattern_weights()
        return exp_forward(inp, ind)
    return p
