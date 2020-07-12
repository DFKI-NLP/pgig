from explainer import backprop as bp
import explainer.random as rand
from explainer.pattern import pattern_augment


def get_explainer(model, name):
    methods = {
        'vanilla_grad': bp.VanillaGradExplainer,
        'grad_x_input': bp.GradxInputExplainer,
        'integrate_grad': bp.IntegrateGradExplainer,
        'guided_backprop': bp.GuidedBackpropExplainer,
        'smooth_grad': bp.SmoothGradExplainer,
        'smooth_grad2': bp.SmoothGrad2Explainer,
        'sg_ig': bp.SmoothGradIntegratedGradientsExplainer,
        'var_grad': bp.VarGradExplainer,
        'exp_grad': bp.ExpectedGradExplainer,
        'random': rand.RandomExplainer
    }

    if 'pattern_' in name:
        name = str(name[8:])
        base_explainer = methods[name](model)
        explainer = pattern_augment(base_explainer, method='pattern_attribution')

    else:
        explainer = methods[name](model)

    return explainer