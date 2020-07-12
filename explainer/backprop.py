import numpy as np
from torch.autograd import Variable, Function
import torch
import types


def load_params(filename='./weights/imagenet_224_vgg_16.npz'):
    f = np.load(filename)
    weights = []

    for i in range(32):
        if i in [26, 28, 30]:
            weights.append(f['arr_%d' % i].T)
        else:
            weights.append(f['arr_%d' % i])

    return weights


class VanillaGradExplainer(object):

    def __init__(self, model, pattern_weights=True):
        self.model = model
        self.weights = list(self.model.parameters())
        self.backprop_probability = False

        # For compatibility with pre-computed patterns, these weights must be loaded
        if pattern_weights:
            self.set_pattern_weights()

    def set_pattern_weights(self):
        """Loads and sets model weights used to calculate patterns.
        This must be called if patterns are being used."""
        self.pattern_weights = load_params()
        self.fill_in_pattern_weights()

    def fill_in_pattern_weights(self):
        for i in range(32):
            self.weights[i].data.copy_(torch.from_numpy(
                self.pattern_weights[i]))

    def forward_pass(self, inp, ind):
        output = torch.softmax(self.model(inp), dim=-1)
        if ind is None:
            ind = output.data.max(1)[1]

        probvalue = output.data.gather(1, ind.unsqueeze(0).t()) if self.backprop_probability else 1.0

        grad_out = output.data.clone()
        grad_out.fill_(0.0)

        #grad_out = grad_out * -1.0
        grad_out.scatter_(1, ind.unsqueeze(0).t(), probvalue)
        return output, grad_out

    def backprop(self, inp, output, grad_out):
        self.model.zero_grad()
        output.backward(grad_out)
        return inp.grad.data

    def explain(self, inp, ind=None):
        output, grad_out = self.forward_pass(inp, ind)
        return self.backprop(inp, output, grad_out)


class IntegrateGradExplainer(VanillaGradExplainer):

    def __init__(self, model, steps=25):
        super(IntegrateGradExplainer, self).__init__(model)
        self.steps = steps

    def explain(self, inp, ind=None):
        grad = 0
        inp_data = inp.data.clone()

        for alpha in np.arange(1 / self.steps, 1.0, 1 / self.steps):
            new_inp = Variable(inp_data * alpha, requires_grad=True)
            g = super().explain(new_inp, ind)
            grad += g

        res = grad * inp_data / self.steps
        return res


class GradxInputExplainer(VanillaGradExplainer):

    def __init__(self, model):
        super(GradxInputExplainer, self).__init__(model)

    def explain(self, inp, ind=None):
        grad = super().explain(inp, ind)
        return inp.data * grad


class GuidedBackpropExplainer(VanillaGradExplainer):

    def __init__(self, model):
        super(GuidedBackpropExplainer, self).__init__(model)
        # According to PA paper
        self.backprop_probability = True
        self._override_backward()

    def _override_backward(self):
        """Replace model ReLU with modified GuidedBP function"""
        class _ReLU(Function):
            @staticmethod
            def forward(ctx, input):
                output = torch.clamp(input, min=0)
                ctx.save_for_backward(output)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                output, = ctx.saved_tensors
                mask1 = (output > 0).float().detach()
                mask2 = (grad_output.data > 0).float().detach()
                grad_inp = mask1 * mask2 * grad_output.data
                grad_output.data.copy_(grad_inp)
                return grad_output

        def new_forward(self, x):
            return _ReLU.apply(x)

        def replace(m):
            if m.__class__.__name__ == 'ReLU':
                m.forward = types.MethodType(new_forward, m)

        self.model.apply(replace)


# modified from https://github.com/PAIR-code/saliency/blob/master/saliency/base.py#L80
class SmoothGradExplainer(VanillaGradExplainer):

    def __init__(self, model, stdev_spread=0.15,
                nsamples=25):
        super().__init__(model)
        self.stdev_spread = stdev_spread
        self.nsamples = nsamples

    def explain(self, inp, ind=None):
        stdev = self.stdev_spread * (inp.data.max() - inp.data.min())

        total_gradients = 0

        for i in range(self.nsamples):
            noise = torch.randn(inp.size()).cuda() * stdev
            inp.data.copy_(noise + inp.data.clone())
            grad = super().explain(inp, ind)

            total_gradients += grad

        return total_gradients / self.nsamples


class SmoothGrad2Explainer(VanillaGradExplainer):

    def __init__(self, model, stdev_spread=0.15,
                nsamples=25):
        super().__init__(model)
        self.stdev_spread = stdev_spread
        self.nsamples = nsamples

    def explain(self, inp, ind=None):
        stdev = self.stdev_spread * (inp.data.max() - inp.data.min())

        total_gradients = 0

        for i in range(self.nsamples):
            noise = torch.randn(inp.size()).cuda() * stdev
            inp.data.copy_(noise + inp.data.clone())
            grad = super().explain(inp, ind)

            total_gradients += grad ** 2

        return total_gradients / self.nsamples


# from https://arxiv.org/pdf/1810.03307.pdf
class VarGradExplainer(VanillaGradExplainer):

    def __init__(self, model, stdev_spread=0.15, nsamples=25):
        super().__init__(model)
        self.stdev_spread = stdev_spread
        self.nsamples = nsamples

    def explain(self, inp, ind=None):
        stdev = self.stdev_spread * (inp.data.max() - inp.data.min())

        grad_sum = 0
        grad_sq_sum = 0

        for i in range(self.nsamples):
            noise = torch.randn(inp.size()).cuda() * stdev
            inp.data.copy_(noise + inp.data.clone())

            grads = super().explain(inp, ind)
            grad_sum += grads
            grad_sq_sum += grads ** 2

        # Var = E[X^2] - E[X]^2
        return grad_sq_sum / self.nsamples - (grad_sum / self.nsamples) ** 2


class ExpectedGradExplainer(VanillaGradExplainer):

    def __init__(self, model, n_samples=49):
        super().__init__(model)
        self.n_samples = n_samples

    def explain(self, inp, ind=None):
        if inp.shape[0] <= self.n_samples:
            raise ValueError("Input size must be > n_samples.")
        grad_sum = 0.0
        s = inp.data
        for _ in range(self.n_samples):
            s = torch.cat((s[-1:], s[:-1]), dim=0)
            alpha = torch.rand(1).cuda()
            interpolated_input = Variable(s + alpha * (inp - s), requires_grad=True)

            grad = super().explain(interpolated_input, ind)
            grad_sum += (inp - s) * grad

        return grad_sum / self.n_samples


class SmoothGradIntegratedGradientsExplainer(SmoothGradExplainer):
    def __init__(self, model, steps=25):
        super().__init__(model)
        self.steps = steps

    def explain(self, inp, ind=None):

        grad = 0
        inp_data = inp.data.clone()

        for alpha in np.arange(1 / self.steps, 1.0, 1 / self.steps):
            new_inp = Variable(inp_data * alpha, requires_grad=True)
            g = super().explain(new_inp, ind)
            grad += g

        res = grad * inp_data / self.steps
        return res