import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import numpy as np


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def plot_cam(attr):
    zero_val = -attr.min()
    attr -= attr.min()
    zero_val /= attr.max()
    attr /= (attr.max() + 1e-20)

    cmap=plt.get_cmap('RdBu_r')

    neg_bounds = np.linspace(0, zero_val, 100)
    pos_bounds = np.linspace(zero_val, 1, 100)
    bounds = np.concatenate((neg_bounds, pos_bounds))
    norm = BoundaryNorm(bounds, cmap.N)

    # plt.imshow(xi)
    p = plt.imshow(attr, interpolation='none', norm=norm, cmap=cmap)
    #plt.colorbar()
    return p

def plot_bbox(bboxes, xi, linewidth=1):
    ax = plt.gca()
    ax.imshow(xi)

    if not isinstance(bboxes[0], list):
        bboxes = [bboxes]

    for bbox in bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                 linewidth=linewidth, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.axis('off')

