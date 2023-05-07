import os
import torch, torchvision
import cifar10.models.vgg as vgg
import cifar10.models.resnet as resnet
import cifar10.models.densenet as densenet

# map between model name and function
models = {
    'vgg9'                  : vgg.VGG9,
    'vgg16'					: vgg.VGG16,
    'densenet121'           : densenet.DenseNet121,
     'resnet18'              : resnet.ResNet18,
    'resnet34'              : resnet.ResNet34,
    'resnet50'              : resnet.ResNet50,
    'resnet18_noshort'      : resnet.ResNet18_noshort,
    'resnet34_noshort'      : resnet.ResNet34_noshort,
    'resnet50+noshort'      : resnet.ResNet50_noshort,
    'resnet101'             : resnet.ResNet101,
    'resnet152'             : resnet.ResNet152,
}

def load(model_name, model_file=None, data_parallel=False):
    net = models[model_name]()
    if data_parallel: # the model is saved in data paralle mode
        net = torch.nn.DataParallel(net)

    if model_file:

        assert os.path.exists(model_file), model_file + " does not exist."
        stored = torch.load(model_file, map_location=lambda storage, loc: storage)
        if 'state_dict' in stored.keys():
            net.load_state_dict(stored['state_dict'])
        else:
            net.load_state_dict(stored)

    if data_parallel: # convert the model back to the single GPU version
        net = net.module

    net.eval()
    return net
