from torchvision import models
from torch.autograd import Variable

        
def load_model(arch):
    '''
    Args:
        arch: (string) valid torchvision model name,
            recommendations 'vgg16' | 'googlenet' | 'resnet50'
    '''
    model = models.__dict__[arch](pretrained=True)
    model.eval()
    return model


def cuda_var(tensor, requires_grad=False):
    return Variable(tensor, requires_grad=requires_grad)

