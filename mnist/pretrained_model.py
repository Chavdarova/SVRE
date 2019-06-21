# -*- coding: utf-8 -*-
"""
Supplementary code for paper under review by the International Conference on Machine Learning (ICML).
Do not distribute.
"""
import os
import torch
import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Please provide int for input_dims'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i+1)] = nn.ReLU()
            layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)

        self.model = nn.Sequential(layers)
        print(self.model)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        return self.model.forward(input)


def mnist(input_dims=784, n_hiddens=[256, 256], n_class=10, pretrained=None):
    model = MLP(input_dims, n_hiddens, n_class)
    if pretrained is not None:
        if os.path.exists(pretrained):
            print('Loading trained model from %s' % pretrained)
            state_dict = torch.load(pretrained,
                    map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
            if 'parallel' in pretrained:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]
                    new_state_dict[name] = v
                state_dict = new_state_dict
        else:
            import torch.utils.model_zoo as model_zoo
            model_urls = {
                'mnist': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist-b07bb66b.pth'}
            m = model_zoo.load_url(model_urls['mnist'],
                    map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
            state_dict = m.state_dict() if isinstance(m, nn.Module) else m
            assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model = model.cuda()
    return model
