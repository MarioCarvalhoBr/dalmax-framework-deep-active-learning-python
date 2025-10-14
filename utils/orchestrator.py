from utils.dataset import DANINHAS_Hander, CIFAR10_Handler
from utils.data import get_DANINHAS, get_CIFAR10, get_CIFAR10_Download

from core.deep_learning import DeepLearning
from core.daninhas_model import DaninhasModelResNet50
from core.cifar10_model import CIFAR10Model

from core.query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
                             LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                             KMeansSampling, KCenterGreedy, BALDDropout, \
                          AdversarialBIM, AdversarialDeepFool, SSRAEKmeansSampling, VCTexKmeansSampling, SSRAEKmeansHCSampling, VCTexKmeansHCSampling

# CONFIGURE YOUR HANDLER HERE AND ADD NEW HANDLERS TO THE get_handler FUNCTION
def get_handler(name):
    if name == 'DANINHAS':
        return DANINHAS_Hander
    elif name == 'CIFAR10':
        return CIFAR10_Handler
    elif name == 'CIFAR10Download':
        return CIFAR10_Handler
    else:
        raise NotImplementedError

# CONFIGURE YOUR DATASET HERE AND ADD NEW DATASETS TO THE get_dataset FUNCTION
def get_dataset(name, params):
    data_dir = params[name]['data_dir']
    if name == 'DANINHAS':
        return get_DANINHAS(handler=get_handler(name), data_dir=data_dir)
    elif name == 'CIFAR10':
        return get_CIFAR10(handler=get_handler(name), data_dir=data_dir)
    elif name == 'CIFAR10Download':
        return get_CIFAR10_Download(handler=get_handler(name))
    else:
        raise NotImplementedError

# CONFIGURE YOUR DEEP LEARNING MODEL HERE AND ADD NEW MODELS TO THE get_network_deep_learning FUNCTION
def get_network_deep_learning(name, device, params):
    if name == 'DANINHAS':
        return DeepLearning(DaninhasModelResNet50, params[name], device)
    elif name == 'CIFAR10':
            return DeepLearning(CIFAR10Model, params[name], device)
    else:
        raise NotImplementedError

def get_strategy(name):
    if name == "RandomSampling":
        return RandomSampling
    elif name == "LeastConfidence":
        return LeastConfidence
    elif name == "MarginSampling":
        return MarginSampling
    elif name == "EntropySampling":
        return EntropySampling
    elif name == "LeastConfidenceDropout":
        return LeastConfidenceDropout
    elif name == "MarginSamplingDropout":
        return MarginSamplingDropout
    elif name == "EntropySamplingDropout":
        return EntropySamplingDropout
    elif name == "KMeansSampling":
        return KMeansSampling
    elif name == "KCenterGreedy":
        return KCenterGreedy
    elif name == "BALDDropout":
        return BALDDropout
    elif name == "AdversarialBIM":
        return AdversarialBIM
    elif name == "AdversarialDeepFool":
        return AdversarialDeepFool
    elif name == "SSRAEKmeansSampling":
        return SSRAEKmeansSampling
    elif name == "VCTexKmeansSampling":
        return VCTexKmeansSampling
    elif name == "SSRAEKmeansHCSampling":
        return SSRAEKmeansHCSampling
    elif name == "VCTexKmeansHCSampling":
        return VCTexKmeansHCSampling
    
    else:
        raise NotImplementedError