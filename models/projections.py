import torch
import torch.nn.functional as functional

from constants import DEBUG
from util.toolkits import grouped, convert_kwargs, Pruner


class ProjectionBase(torch.nn.Module):
    def __init__(self, config):
        super(ProjectionBase, self).__init__()
        self._config = config
        self._device = torch.device("cuda" if torch.cuda.is_available() and not DEBUG else "cpu")

    def forward(self, net):
        pass

    def initialize(self, method="xavier"):
        self.projection.apply(getattr(self, "_{}".format(method)))

    @staticmethod
    def _xavier(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


class PrunedLinear(torch.nn.Linear):
    def __init__(self, pruner=None, *args, **kwargs):
        super(PrunedLinear, self).__init__(*args, **kwargs)
        self.pruner = None if pruner is None else pruner

    def forward(self, net):
        if self.pruner is None:
            weight = self.weight
        else:
            weight = self.pruner(self.weight)
        return functional.linear(net, weight, self.bias)


class Linear(ProjectionBase):
    def __init__(self, config, input_dim, output_dim, use_bias=True):
        super(Linear, self).__init__(config)
        pruner_config = self._config.setdefault("pruner_config", None)
        pruner = None if pruner_config is None else Pruner(pruner_config)
        self.projection = PrunedLinear(pruner, input_dim, output_dim, use_bias).to(self._device)
        method = self._config.setdefault("initialize_method", "xavier")
        if method is not None:
            self.initialize(method)

    def forward(self, net):
        return self.projection(net)


class BN(torch.nn.BatchNorm1d):
    def forward(self, net):
        if len(net.shape) == 3:
            net = net.transpose(1, 2)
        net = super(BN, self).forward(net)
        if len(net.shape) == 3:
            net = net.transpose(1, 2)
        return net




class MLP(ProjectionBase):
    def __init__(self, config, input_dim, output_dim, use_bias=True):
        super(MLP, self).__init__(config)
        self._input_dim = input_dim
        layers, in_dim = [], input_dim
        use_bn = self._config.setdefault("use_bn", False)
        dropout_prob = self._config.setdefault("dropout_prob", 0.)
        pruner_config = self._config.setdefault("pruner_config", {})
        pruner = None if pruner_config is None else Pruner(pruner_config)
        layers_config = self._config.setdefault("layers", [256, "GLU", 256, "GLU"])
        self.last_dimension = layers_config[-1]
        for n_unit, act in grouped(layers_config, 2):
            out_dim = n_unit * 2 if act == "GLU" else n_unit
            l=PrunedLinear(pruner, in_dim, out_dim, use_bias)
            layers.append(l)
            del l
            if use_bn:
                l=BN(num_features=n_unit)
                layers.append(l)
                del l
            if act is not None:
                l=getattr(torch.nn, act)()
                layers.append(l)
                del l
            if dropout_prob > 0:
                l=torch.nn.Dropout(dropout_prob)
                layers.append(l)
                del l
            in_dim = n_unit
        if output_dim is not None:
            self.last_dimension = output_dim
            l=PrunedLinear(None, in_dim, output_dim, use_bias)
            layers.append(l)
            del l
        self.projection = torch.nn.Sequential(*layers).to(self._device)
        method = self._config.setdefault("initialize_method", "xavier")
        if method is not None:
            self.initialize(method)

    def forward(self, net):
        return self.projection(net)


def gen_proj(config, config_name, input_dim, output_dim, use_bias, dtype="linear"):
    proj_config = convert_kwargs(config.setdefault(config_name, {}))
    proj_type = proj_config.setdefault("type", dtype)
    if proj_type is None:
        return
    return proj_dict[proj_type](proj_config, input_dim, output_dim, use_bias)


proj_dict = {
    "linear": Linear, "mlp": MLP,
}


__all__ = ["proj_dict", "gen_proj"]
