import sys
from datetime import datetime

from torchvision.transforms import transforms

from preprocess import PatternPreprocess

sys.path.append('./')

from explainer.lrp import lrp_utils
from explainer.pattern import pattern_augment

from create_explainer import get_explainer
import utils
import viz
import torch
import os
import pylab
import numpy as np
import matplotlib.pyplot as plt

params = {
    'font.family': 'sans-serif',
    'axes.titlesize': 25,
    'axes.titlepad': 10,
}
pylab.rcParams.update(params)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_methods = [
    ['vgg16', 'deconv', 'camshow'],
    #['vgg16', 'exp_grad', 'camshow'],
    ['vgg16', 'grad_x_input', 'camshow'],
    ['vgg16', 'gradcam', 'camshow'],
    ['vgg16', 'guided_backprop', 'camshow'],
    ['vgg16', 'integrate_grad', 'camshow'],
    ['vgg16', 'relu_integrate_grad', 'camshow'],
    #['vgg16', 'sg_ig', 'camshow'],
    #['vgg16', 'smooth_grad2', 'camshow'],
    #['vgg16', 'smooth_grad', 'camshow'],
    ['vgg16', 'vanilla_grad', 'camshow']
    #['vgg16', 'var_grad', 'camshow']
]

image_path = 'images/elephant.png'
image_class = 101 #309 #483
image_class_name = 'tusker'
raw_img = viz.pil_loader(image_path)

saliencies = [[],[]]

transf = transforms.Compose([
    PatternPreprocess((224, 224))
])  # this transform is the right one for pattern-based models and methods

for model_name, method_name, _ in model_methods:
    # transf = transforms.Compose([
    #     transforms.Resize((225, 225)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])
    for pattern_attribution in [False, True]:
        torch.cuda.empty_cache()

        img_input = transf(raw_img).cuda()
        img_input = utils.cuda_var(img_input.unsqueeze(0), requires_grad=True)

        model = utils.load_model(model_name).cuda()

        explainer = get_explainer(model, method_name)
        if pattern_attribution:
            explainer = pattern_augment(explainer, method='pattern_attribution')
        pred = model(img_input)

        ind = pred.data.max(1)[1]
        ind = torch.tensor(np.expand_dims(np.array(image_class), 0)).cuda()
        print(f'Processing {method_name}, predicted class is {lrp_utils.imgclasses[ind.item()]}')

        target = torch.LongTensor([image_class]).cuda()

        saliency = explainer.explain(img_input, ind)
        saliencies[int(pattern_attribution)].append(saliency.cpu().numpy())

#saliencies[1].append(saliencies[1][1] - saliencies[1][0])
#saliencies[0].append(saliencies[0][1] - saliencies[0][0])

#print(np.max(np.abs(saliencies[1][-1])))
#model_methods.append(['', 'diff', ''])

all_saliency_maps = list(zip(saliencies[0], saliencies[1]))

plt.figure(figsize=(15, 15))
rows = len(model_methods) + 1
plt.subplot(rows, 3, 1)
plt.imshow(raw_img)
plt.axis('off')

fntdct = {'size': 12}

plt.subplot(rows, 3, 2)
plt.text(0.4, 0.5, 'original', fontdict=fntdct)
plt.axis('off')

plt.subplot(rows, 3, 3)
plt.text(0.4, 0.5, 'ours', fontdict=fntdct)
plt.axis('off')

for row, ((saliency_wo_patterns, saliency_w_patterns), (model_name, method_name, show_style)) in enumerate(
        zip(all_saliency_maps, model_methods)):
    plt.subplot(rows, 3, ((row + 1) * 3) + 1)
    plt.text(0.5, 0.5, f'{method_name}', fontdict=fntdct)
    plt.axis('off')

    plt.subplot(rows, 3, ((row + 1) * 3) + 2)
    sal_img = saliency_wo_patterns.sum(axis=1).squeeze()  # todo what does max do here?
    viz.plot_cam(sal_img)
    plt.axis('off')

    plt.subplot(rows, 3, ((row + 1) * 3) + 3)
    sal_img = saliency_w_patterns.sum(axis=1).squeeze()  # todo what does max do here?
    viz.plot_cam(sal_img)  # if model_name == 'googlenet' or method_name == 'pattern_net':
    plt.axis('off')

#plt.tight_layout()
d = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
plt.savefig('images/{}.png'.format(d))
plt.show()

print()


plt.clf()
raw_img = raw_img.resize(sal_img.shape)
fig = plt.imshow(raw_img)
plt.axis('off')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.savefig('images/input.png', bbox_inches='tight', pad_inches = 0)
for i, p in enumerate(['', 'pattern_']):
    for j, m in enumerate(model_methods):
        plt.clf()
        sal_img = all_saliency_maps[j][i].sum(axis=1).squeeze()  # todo what does max do here?
        fig = viz.plot_cam(sal_img)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.axis('off')
        plt.savefig('images/{}{}'.format(p, m[1]), bbox_inches='tight', pad_inches = 0)