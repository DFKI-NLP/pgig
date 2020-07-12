import sys
sys.path.append("..")

from create_explainer import get_explainer
from preprocess import get_preprocess
import utils
import torch
import viz
import numpy as np
from scipy.stats import spearmanr

def normalize_range(arr, min, max):
    arr += -arr.min()
    arr /= arr.max() / (max - min)
    arr += min
    return arr

def cascading_parameter_randomization(method_name, pattern_augmented, input, target):
    model = utils.load_model('vgg16')
    init_out = None
    state_dict = model.state_dict()
    print(method_name)
    for idx, k in enumerate(reversed(state_dict.keys())):
        if 'weight' in k:
            explainer = get_explainer(model, method_name)
            explainer.set_weights_and_patterns()
            if pattern_augmented:
                explainer.set_weights_and_patterns()
            saliency = explainer.explain(input, target)
            if method_name=='pattern_net' or method_name=='pattern_attribution':
                saliency = explainer.explain(input, target, idx)
            out = saliency.cpu().flatten()
            out = normalize_range(out, -1.0, 1.0)
            print(out)

            if init_out is None:
                init_out = out
                continue

            corr = spearmanr(init_out, out)
            print(corr)
            corr = spearmanr(np.abs(init_out), np.abs(out))
            print(corr)

            state_dict[k] = torch.rand_like(state_dict[k])
            # shuffle randomization method
            # idx = torch.randperm(layer.nelement())
            # layer = layer.view(-1)[idx].view(layer.size())

            # reset randomization method
            model.load_state_dict(state_dict)


model_methods = [
    #['googlenet', 'vanilla_grad',       'camshow'],
    #['vgg16', 'grad_x_input',       'camshow'],
    #['vgg16', 'saliency',           'camshow'],
    #['vgg16', 'integrate_grad',     'camshow'],
    #['vgg16', 'deconv',             'camshow'],
    #['vgg16', 'guided_backprop',    'camshow'],
    ['vgg16',     'pattern_net',        'camshow'],
    ['vgg16',     'pattern_lrp',        'camshow'],
    ['vgg16',     'smooth_grad',        'camshow'],
    ['vgg16', 'deeplift_rescale', 'camshow']]

if __name__ == '__main__':
    for m in model_methods:
        method = m[1]
        image_path = '../images/bee.jpg'
        image_class = 309
        image_class_name = 'bee'
        raw_img = viz.pil_loader(image_path)
        img_input = get_preprocess('vgg16', method)(raw_img)
        img_input = utils.cuda_var(img_input.unsqueeze(0), requires_grad=True)
        target = torch.LongTensor([image_class])

        cascading_parameter_randomization(method, False, img_input, target)