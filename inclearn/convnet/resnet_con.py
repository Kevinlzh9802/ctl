import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import copy

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, remove_last_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.remove_last_relu = remove_last_relu

    def forward(self, x_list):
        ext, x = x_list
        try:
            cx = torch.cat(ext, x)
        except:
            cx = x

        out = self.conv1(cx)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out += identity
        if not self.remove_last_relu:
            out = self.relu(out)
        return out


class ModuleGroup(nn.Module):
    def __init__(self, module_type):
        super(ModuleGroup, self).__init__()
        self.module_type = module_type

    def expand(self, **kwargs):
        new_module = get_module(self.module_type, kwargs)
        self.add_module(self.module_type, new_module)

    def forward(self, x_list):
        assert len(self.modules) == len(x_list)
        out = []
        for k in range(len(self.modules)):
            module = self.modules[k]
            x_needed = feature_cat(x_list, self.task_info)
            out.append(module(x_needed))
        return out


class MultiModuleGroup(nn.Module):
    '''This should correspond to _make_layer() function'''
    def __init__(self, module_type, length):
        super(MultiModuleGroup, self).__init__()
        self.module_type = module_type
        self.length = length

        # use this since nn.Sequential.append() is not found
        groups = []      
        for _ in range(self.length):
            groups.append(ModuleGroup(self.module_type))
        self.groups = nn.Sequential(*groups)

    def expand(self, *args):
        for k in range(self.length):
            self.groups[k].expand(args[k])

    def forward(self, x):
        return self.groups(x)


class ResConnect(nn.Module):
    def __init__(self, block, layer_num, at_info=None, dataset='cifar', remove_last_relu=False):
        super(ResConnect, self).__init__()
        self.block = block
        self.layer_num = layer_num
        self.at_info = at_info
        self.dataset = dataset
        self.remove_last_relu = remove_last_relu
        self.net_groups = nn.Sequential(
            ModuleGroup('Conv'), 
            MultiModuleGroup(block, layer_num[0]), 
            MultiModuleGroup(block, layer_num[1]), 
            MultiModuleGroup(block, layer_num[2]), 
            MultiModuleGroup(block, layer_num[3]), 
            ModuleGroup('AvgPool'), 
        )

    def update_task_info(self, at_info):
        self.at_info = at_info
    
    def expand(self, base_nf):
        layer_args = [
            {'nf': base_nf, 'dataset': self.dataset},   # conv params
            self.multigroup_expand_args(base_nf, 0, self.layer_num[0]), 
            [{}, {}], 
            [{}, {}], 
            [{}, {}], 
            {}  # four BasicBlocks and last avgpool layer, no params
        ]
        assert len(layer_args) == len(self.net_groups)
        for k in range(len(layer_args)):
            groups = self.net_groups[k]
            args = layer_args[k]
            groups.expand(args)


    def get_input_dims(self):
        c = 9

    def group_expand_args(self):
        pass

    def multigroup_expand_args(self, base_nf, stage, blocks, t_num=-1, block_name='BasicBlock', rm_last_relu=False):
        assert stage in [0, 1, 2, 3]
        planes = pow(2, stage) * base_nf
        inplanes = self.get_input_dims()
        stride = 1 if stage == 0 else 2
        if block_name == 'BasicBlock':
            expansion = 1
        elif block_name == 'BottleNeck':
            expansion = 4

        args = []
        downsample = None
        if stride != 1 or self.inplanes != base_nf * expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, base_nf * expansion, stride),
                nn.BatchNorm2d(base_nf * expansion),
            )
        args.append({
            "inplanes": inplanes, 
            "planes": planes, 
            "stride": stride, 
            "downsample": downsample
        })

        if rm_last_relu:
            for i in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, remove_last_relu=True))
        else:
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))
        

    def forward(self, x):
        return self.net_groups(x)
            

def feature_cat(x, ancestors):
    x_list = []
    for k in range(len(x)):
        if k in ancestors:
            x_list.append(x[k])
    return x_list

def get_module(m_type, **kwargs):
    if m_type == 'BasicBlock':
        return BasicBlock()
    elif m_type == 'Conv':
        nf = kwargs["nf"]
        dataset = kwargs["datasets"]
        if dataset == 'cifar':
            return nn.Sequential(nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=False),
                                           nn.BatchNorm2d(nf), nn.ReLU(inplace=True)) 
        else:
            raise ValueError('dataset imagenet100 not implemented!')
    elif m_type == 'AvgPool':
        return nn.AdaptiveAvgPool2d((1, 1))


def resconnect18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    """
    model = ResConnect('BasicBlock', [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
