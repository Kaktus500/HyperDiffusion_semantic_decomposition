import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as dist
from torch import nn

from embedder import Embedder


class MLP(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        hidden_neurons,
        use_tanh=True,
        over_param=False,
        use_bias=True,
    ):
        super().__init__()
        multires = 1
        self.over_param = over_param
        if not over_param:
            self.embedder = Embedder(
                include_input=True,
                input_dims=2,
                max_freq_log2=multires - 1,
                num_freqs=multires,
                log_sampling=True,
                periodic_fns=[torch.sin, torch.cos],
            )
        self.layers = nn.ModuleList([])

        self.layers.append(nn.Linear(in_size, hidden_neurons[0], bias=use_bias))
        for i, _ in enumerate(hidden_neurons[:-1]):
            self.layers.append(
                nn.Linear(hidden_neurons[i], hidden_neurons[i + 1], bias=use_bias)
            )
        self.layers.append(nn.Linear(hidden_neurons[-1], out_size, bias=use_bias))
        self.use_tanh = use_tanh

    def forward(self, x):
        if not self.over_param:
            x = self.embedder.embed(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)
        if self.use_tanh:
            x = torch.tanh(x)
        return x, None


class MLP3D(nn.Module):
    def __init__(
        self,
        out_size,
        hidden_neurons,
        use_leaky_relu=False,
        use_bias=True,
        multires=10,
        output_type=None,
        move=False,
        mlp_components=1,
        **kwargs,
    ):
        super().__init__()
        self.embedder = Embedder(
            include_input=True,
            input_dims=3 if not move else 4,
            max_freq_log2=multires - 1,
            num_freqs=multires,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )

        # one component is e.g. the first 32 neurons of a 128-neuron layer
        # the mask sets all weights except for the i-th component to zero
        hid_neur = hidden_neurons[0]
        assert all(n == hid_neur for n in hidden_neurons)
        assert hid_neur % mlp_components == 0
        component_size = hid_neur // mlp_components

        self.mask = []
        for i in range(mlp_components): 
            self.mask.append(torch.zeros((1,1,hid_neur)).to(device='cuda'))
            self.mask[i][:,:,i*component_size:(i+1)*component_size] = 1
        self.mask.append(torch.ones((1,1,128)).to(device='cuda'))
        self.layers = nn.ModuleList([])
        self.output_type = output_type
        self.use_leaky_relu = use_leaky_relu
        in_size = self.embedder.out_dim
        self.layers.append(nn.Linear(in_size, hidden_neurons[0], bias=use_bias))
        for i, _ in enumerate(hidden_neurons[:-1]):
            self.layers.append(
                nn.Linear(hidden_neurons[i], hidden_neurons[i + 1], bias=use_bias)
            )
        self.layers.append(nn.Linear(hidden_neurons[-1], out_size, bias=use_bias))

    def forward(self, model_input, freeze: bool = False, p: int = 0):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        x = coords_org
        x = self.embedder.embed(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.leaky_relu(x) if self.use_leaky_relu else F.relu(x)

            if freeze:
                x = x * self.mask[p]

        x = self.layers[-1](x)

        if self.output_type == "occ":
            # x = torch.sigmoid(x)
            pass
        elif self.output_type == "sdf":
            x = torch.tanh(x)
        elif self.output_type == "logits":
            x = x
        else:
            pass
            # raise f"This self.output_type ({self.output_type}) not implemented"
        x = dist.Bernoulli(logits=x).logits

        return {"model_in": coords_org, "model_out": x}
