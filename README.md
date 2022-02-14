# Pattern-Guided Integrated Gradients
(This repo is based on the [visual-attribution](https://github.com/yulongwang12/visual-attribution) (07c79...) frame work by Yulong Wang (2018).)

Code accompanying the paper [Pattern-Guided Integrated Gradients](https://arxiv.org/abs/2007.10685), presented at the [ICML 2020 Workshop on Human Interpretability in MachineLearning (WHI 2020)](http://whi2020.online/papers.html) by Robert Schwarzenberg and Steffen Castle (equal contribution):

```
@inproceedings{pgig2020,
  title={Pattern-Guided Integrated Gradients},
  author={Schwarzenberg, Robert and Steffen Castle},
  booktitle={Proc. of the ICML 2020 Workshop on Human Interpretability in Machine Learning (WHI)},
  year={2020}
}
```

For a quick overview, see the [synthetic experiments](https://github.com/dfki-nlp/pgig/blob/master/synthetic.ipynb). 

Addendum: We now also present [arguments](https://github.com/DFKI-NLP/pgig/blob/master/pattern-attribution-implementation-invariance-camera-ready.ipynb) against the implementation invariance of PA and PGIG.

Input|Integrated Gradients | PatternAttribution | Pattern-Guided Integrated Gradients
---|---|---|---
<img src="https://github.com/dfki-nlp/pgig/blob/master/images/elephant.png?raw=true" width=200 height=200 />|<img src="https://github.com/dfki-nlp/pgig/blob/master/images/integrate_grad.png?raw=true" width=200 height=200 /> | <img src="https://github.com/dfki-nlp/pgig/blob/master/images/pattern_vanilla_grad.png?raw=true" width=200 height=200 /> | <img src="https://github.com/dfki-nlp/pgig/blob/master/images/pattern_integrate_grad.png?raw=true" width=200 height=200 />


# Requirements

We ran the experiments on Ubuntu 16.04.6 LTS, with Python 3.6.9 installed, on GeForce GTX 1080 Tis, using cuda10.0+cudnn7.6.2

The minimal requirements are 

torch==1.2.0          
torchvision==0.4.0a0+6b959ee

scikit-learn==0.22.1   

All other dependencies are listed in dependencies.txt

# Data
## Weights and Patterns

Weights and patterns can be downloaded using the `download_patterns.sh` script in the `weights` directory. In case the URLs become invalid, there are backups: 
* [Patterns](https://cloud.dfki.de/owncloud/index.php/s/Q6pYgezGip6MNMK)
* [Weights](https://cloud.dfki.de/owncloud/index.php/s/MoHdiFsbdx2wGCm)


# Degradation Experiment

The degradation experiment can be executed by running `experiments/degradation_test.py`.

## ImageNet 

To run experiments, the validation set of ImageNet 2012 must already be downloaded and extracted locally, and the ImageNet directory must be specified using the `--data_dir` argument. The `--imagenet_download_key` argument and functionality to automatically download the ImageNet dataset is deprecated. 

## Arguments

Argument |Description
--- | ---
--methods| List of methods to evaluate
--batch_size| Batch size, default is 50
--data_dir| Directory where imagenet validation set is downloaded, or where to download it
--val_size| Number of samples to evaluate, default is the whole ImageNet validation set
--n_patches| Number of patches to degrade, default is 100
--imagenet_download_key| Optional URL string for ImageNet download (deprecated)

### Available methods

Method|Method argument string
---|---
Vanilla Gradient|vanilla_grad
PatternAttribution|pattern_vanilla_grad
Gradient times Input|grad_x_input
Integrated Gradients|integrate_grad
Guided Backprop|guided_backprop
SmoothGrad|smooth_grad
SmoothGrad<sup>2</sup>|smooth_grad2
SmoothGrad-Integrated Gradients|sg_ig
VarGrad|var_grad
Expected Gradients|exp_grad
Pattern-Guided Integrated Gradients|pattern_integrate_grad

### Example

```python3 experiments/degradation_test.py --methods vanilla_grad pattern_vanilla_grad integrate_grad pattern_integrate_grad --batch_size 100 --val_size 10000 --n_patches 100 --data_dir /mnt/hdd/datasets/imagenet/```

# References 

[1] Kindermans, Pieter-Jan, et al. "Learning how to explain neural networks: Patternnet and patternattribution." arXiv preprint arXiv:1705.05598 (2017). 

[2] Sundararajan, Mukund, Ankur Taly, and Qiqi Yan. "Axiomatic attribution for deep networks." arXiv preprint arXiv:1703.01365 (2017).
