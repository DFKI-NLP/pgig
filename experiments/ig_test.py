import sys
sys.path.append("/home/pxai/")

import torch
import glob
import random
import utils
from preprocess import get_preprocess
from create_explainer import get_explainer
import viz
import numpy as np

if __name__ == '__main__':

    dataset_path = '/mnt/hdd/datasets/imagenet/ILSVRC/Data/CLS-LOC/val/'
    test_set_size = 100
    batch_size = 50

    methods = ['pattern_integrate_grad']#, 'integrate_grad']
    #pattern_methods = ['pattern_' + m for m in methods]
    #methods.extend(pattern_methods)
    image_paths = glob.glob('{}*.JPEG'.format(dataset_path))

    random.shuffle(image_paths)
    #image_paths = image_paths
    print("Evaluating on {} images.".format(len(image_paths)))
    num_batches = int(len(image_paths) / batch_size)

    model = utils.load_model('vgg16').cuda()

    results = {}

    for batch_no in range(num_batches):
        print("Batch {} of {}".format(batch_no+1, num_batches))
        image_batch = image_paths[batch_no * batch_size:(batch_no + 1) * batch_size]
        raw_imgs = [viz.pil_loader(image_path) for image_path in image_batch]
        # make sure preprocessing is correct
        inputs = [get_preprocess('vgg16', 'pattern_vanilla_grad')(raw_img) for raw_img in raw_imgs]
        inputs = torch.stack(inputs).cuda()
        inputs = utils.cuda_var(inputs, requires_grad=True)

        diff_sum = 0

        with torch.cuda.device(0):
            torch.cuda.empty_cache()
            model = utils.load_model('vgg16').cuda()
            explainer = get_explainer(model, 'vanilla_grad')

            out = torch.softmax(model(inputs.clone()), axis=-1)
            classes = torch.max(out, dim=1)[1]
            out = out.detach().cpu().numpy()

            # get baseline val
            baseline_inp = torch.zeros_like(inputs)
            baseline_out = torch.softmax(model(baseline_inp), axis=-1).detach().cpu().numpy()

            score_diff = np.array([s[c] for s, c in zip((out - baseline_out), classes)])

            model = utils.load_model('vgg16').cuda()

            explainer = get_explainer(model, 'integrate_grad')
            ig_sum = explainer.explain(inputs, classes).sum(dim=[1, 2, 3]).cpu().numpy()

            diff = np.mean(ig_sum - score_diff)
            print("Batch={}, diff={}".format(batch_no, diff))
            diff_sum += diff
    print("Final mean diff: {}".format(diff_sum / num_batches))
