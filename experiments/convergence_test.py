import sys
sys.path.append("..")

from create_explainer import get_explainer
import utils
import torch
import matplotlib.pyplot as plt
from explainer.lrp import lrp_utils
import numpy as np


# Need to do this in order to run convergence experiments

class ConvergenceTester:

    def __init__(self, explainer):
        self.explainer = explainer
        self.layers = []

    def get_names(self):
        return reversed([type(l) for l in self.layers])

    def get_convergence(self, random_layer_num, batch_size=32, num_batches=1000, pattern=False):
        self.explainer.reset_params()
        self._forward(batch_size, random_layer_num)
        if pattern:
            self.explainer.fill_in_patterns()
        return self._backward(random_layer_num, num_batches)

    def _forward(self, batch_size, layer_num):

        f = list(self.explainer.model._modules['features']) + \
            [self.explainer.model._modules['avgpool']] + \
            [torch.nn.Flatten()] + \
            list(self.explainer.model._modules['classifier'])
        print(len(f))
        self.layers = f[:layer_num]

        input_shape = (3,240,240)
        rand_input = torch.normal(torch.zeros(input_shape), torch.ones(input_shape)).cuda()
        rand_input = rand_input.unsqueeze(0).repeat((batch_size,1,1,1))
        X = torch.autograd.Variable(rand_input, requires_grad=True)

        self.activations = [X]
        for i in range(len(self.layers)):
            out = self.layers[i].forward(self.activations[i])
            out.retain_grad()
            self.activations.append(out)

    def _backward(self, layer_num, num_batches, batch_size=32):
        cos_sims = np.zeros((len(self.activations),))
        for _ in range(num_batches):
            grad = torch.normal(torch.zeros_like(self.activations[layer_num]),
                                torch.ones_like(self.activations[layer_num])).cuda()

            self.activations[layer_num].backward(grad, retain_graph=True)

            for idx, a in enumerate(reversed(self.activations)):
                g = a.grad
                if len(g.shape) < 3:
                    g = g.unsqueeze(1)
                r1, r2 = torch.split(g, int(batch_size/2), dim=0)
                cos_sims[idx] += torch.nn.functional.cosine_similarity(r1, r2, dim=1)\
                    .mean().cpu().data.numpy()
        return cos_sims / num_batches

    def get_baseline(self, layer_num, batch_size=32):
        exp.reset_params()
        self._forward(batch_size, layer_num)
        cos_sims = np.zeros((len(self.activations),))
        grad = torch.normal(torch.zeros_like(self.activations[layer_num]),
                            torch.ones_like(self.activations[layer_num])).cuda()

        self.activations[layer_num].backward(grad, retain_graph=True)

        for idx, a in enumerate(reversed(self.activations)):
            g = a.grad
            if len(g.shape) < 3:
                g = g.unsqueeze(1)
            rand_grad = torch.normal(torch.zeros_like(g),
                                torch.ones_like(g)).cuda()
            r1, r2 = torch.split(rand_grad, int(batch_size/2), dim=0)
            cos_sims[idx] += torch.nn.functional.cosine_similarity(r1, r2, dim=1) \
                .mean().cpu().data.numpy()
        return cos_sims


if __name__ == '__main__':

    methods = ['saliency', 'vanilla_grad', 'deconv', 'guided_backprop'] # , 'deconv', 'vanilla_grad']
    pattern_methods = ['pattern_' + m for m in methods]
    methods.extend(pattern_methods)

    results = []

    layer_num = 34

    x = range(layer_num+1)
    for m in methods:
        print(m)
        torch.cuda.empty_cache()
        model = utils.load_model('vgg16').cuda()
        exp = get_explainer(model, m)
        c = ConvergenceTester(exp)
        conv = c.get_convergence(layer_num, pattern='pattern_' in m)
        plt.plot(x, list(conv), label=m)

    torch.cuda.empty_cache()
    model = utils.load_model('vgg16').cuda()
    exp = get_explainer(model, 'vanilla_grad')
    c = ConvergenceTester(exp)
    conv = c.get_baseline(layer_num)
    plt.plot(x, conv, label='baseline')
    plt.xticks(x, c.get_names(), rotation='vertical')
    plt.legend()
    plt.show()
    plt.savefig('convergence_output.png')

