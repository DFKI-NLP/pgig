import sys
sys.path.append("/home/pxai")

from create_explainer import get_explainer
from preprocess import get_preprocess
import utils
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import time
import datetime
from experiments.imagenet import get_imagenet_dataloader
import argparse

def get_saliency(explainer, input, target):
    input = utils.cuda_var(input.clone(), requires_grad=True)
    # sum over 3 input channels
    #saliency = explainer.explain(input, target)
    #saliency[input < 0] = 0
    saliency = explainer.explain(input, target).sum(axis=1).unsqueeze(1).detach()    # average pool - should sum according to experiment in Kindermans et al,
    # but avg_pool2d is functionally equivalent since each patch is the same size.
    saliency = torch.nn.functional.avg_pool2d(saliency, (9,9),(9,9))
    # shape = (batch_size, num_patches)
    saliency = saliency.reshape(saliency.shape[0],-1)
    saliency = torch.argsort(saliency, dim=1, descending=True)
    return saliency

def get_degradation_mask(mask, pooled_inputs, sorted_saliency_indices, patch_num):
    if patch_num <= 0:
        return mask

    orig_shape = mask.shape
    # merge height and width dimensions so that they can be indexed

    pooled_inputs = pooled_inputs.reshape(pooled_inputs.shape[0], 3, -1)
    mask = mask.reshape(pooled_inputs.shape)

    indices = sorted_saliency_indices[:,patch_num]
    indices = indices.unsqueeze(1).unsqueeze(1).repeat((1,3,1))

    patch_vals = torch.gather(pooled_inputs,2,indices)
    mask.scatter_(2,indices,patch_vals)

    mask = mask.reshape(orig_shape)

    return mask


def get_confidence(model, model_input, class_num=None):
    y = torch.softmax(model(model_input), -1).detach()
    if class_num is not None:
        vals = torch.stack([y[idx, i] for idx, i in enumerate(class_num)])
        return torch.sum(vals), None
    return torch.sum(torch.max(y, dim=-1)[0]), torch.argmax(y, dim=-1)


def degradation_test(explainer, inputs, n_patches):
    inp = inputs.clone()

    # get predicted classes from non-occluded input
    # replace this with true classes if possible
    _, classes = get_confidence(explainer.model, inp)

    # get saliency
    sorted_saliency_indices = get_saliency(explainer, inp, classes)
    explainer.fill_in_pattern_weights()

    # mean of each patch for each channel using avg_pool2d
    pooled = torch.nn.functional.avg_pool2d(inp, (9,9),(9,9))
    mask = torch.zeros_like(pooled)

    confidences = torch.Tensor([]).cuda()

    for p in range(n_patches):
        mask = get_degradation_mask(mask, pooled, sorted_saliency_indices, p)

        # upscale mean pixels to patch size, using nearest neighbor interpolation
        img_mask = torch.nn.functional.interpolate(mask, scale_factor=(9, 9), mode='nearest')

        deg_inp = inputs.clone()
        mask_idcs = img_mask > 0
        deg_inp[mask_idcs] = img_mask[mask_idcs]

        c, _ = get_confidence(explainer.model, deg_inp, class_num=classes)
        confidences = torch.cat([confidences,torch.unsqueeze(c,0)])

    return confidences.cpu().data.numpy().flatten()


def plot_results(method_dict, n_patches):
    x = range(n_patches)
    cmap = plt.get_cmap("tab10")

    # plot random first and remove since it doesn't have a pattern pair
    if 'random' in method_dict.keys():
        rand = method_dict.pop('random')
        plt.plot(x, rand, '-', label='random', color='k')

    pattern_methods = sorted([m for m in method_dict.keys() if 'pattern_' in m])
    other_methods = sorted([m for m in method_dict.keys() if 'pattern_' not in m])

    # plot method and its pattern augmented version with same color
    # pattern augmented line is dashed
    for i, name in enumerate(other_methods):
        plt.plot(x, method_dict[name], '-', label=name, color=cmap(i))
    for i, name in enumerate(pattern_methods):
        plt.plot(x, method_dict[name], '--', label=name, color=cmap(i))

    plt.legend()
    plt.show()
    plt.savefig('images/degradation_output.png')


def load_results(results_files='*_degradation.npy'):
    methods_dict = {}
    results_paths = sorted(glob.glob(results_files))

    for p in results_paths:
        c = np.load(p)[:100]
        m = str(os.path.basename(p)[:-16])
        methods_dict[m] = c
    return methods_dict


def gen_csv(method_dict, num_patches):
    with open('results.csv', 'w') as f:
        f.write('patches')
        for m in method_dict.keys():
            f.write(',{}'.format(m))
        f.write('\n')
        for i in range(num_patches):
            f.write(str(i))
            for method in method_dict.keys():
                f.write(',{}'.format(method_dict[method][i]))
            f.write('\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+',
                        help='List of methods to evaluate')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--data_dir', default='/mnt/hdd/datasets/imagenet/',
                        help='Imagenet directory')
    parser.add_argument('--val_size', type=int, default=50000,
                        help='Number of validation samples to run')
    parser.add_argument('--n_patches', type=int, default=100,
                        help='Number of patches to degrade')
    parser.add_argument('--imagenet_download_key', default=None,
                        help='Optional URL string for imagenet download')
    args = parser.parse_args()

    methods = args.methods
    batch_size = args.batch_size
    data_dir = args.data_dir
    val_size = args.val_size
    n_patches = args.n_patches
    imagenet_download_key = args.imagenet_download_key

    # Load data
    input_transform = get_preprocess('vgg16', 'pattern_vanilla_grad')
    imagenet_data = get_imagenet_dataloader(data_dir, input_transform, val_size, batch_size, imagenet_download_key)

    num_batches = len(imagenet_data)
    print("Evaluating on {} images.".format(val_size))

    model = utils.load_model('vgg16').cuda()

    results = {}

    for batch_idx, batch in enumerate(imagenet_data):
        t = time.time()
        print("Batch {} of {}".format(batch_idx+1, num_batches))

        for m in methods:
            print("Method: {}".format(m))
            explainer = get_explainer(model, m)

            with torch.cuda.device(0):

                torch.cuda.empty_cache()
                confidences = degradation_test(explainer, batch[0].cuda(), n_patches)
                if m not in results:
                    results[m] = confidences
                else:
                    results[m] += confidences
        batch_time = time.time() - t
        print("Batch time: {} seconds".format(batch_time))
        remaining_time = batch_time * (num_batches - batch_idx - 1)
        remaining = datetime.timedelta(seconds=remaining_time)
        print("Approximate time remaining: {}".format(remaining))

    for k in results.keys():
        np.save('{}_degradation.npy'.format(k), results[k] / val_size)

    # sort methods list into pattern and non-pattern methods
    method_dict = load_results()

    gen_csv(method_dict, n_patches)
    plot_results(method_dict, n_patches)

