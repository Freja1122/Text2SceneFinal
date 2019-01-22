import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init
from util import *
from util.toolkits import flat_bts, unflat_bts

class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)
        if torch.cuda.is_available():
            self.reset_gate = self.reset_gate.cuda()
            self.update_gate = self.update_gate.cuda()
            self.out_gate = self.out_gate.cuda()

    def forward(self, input_, prev_state):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = torch.zeros(state_size).cuda()
            else:
                prev_state = torch.zeros(state_size)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class ConvGRU(nn.Module):

    def __init__(self, config):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.
        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        '''

        super(ConvGRU, self).__init__()

        self._config = config
        self.input_size = self._config.setdefault("input_size", 256)
        self.hidden_sizes = self._config.setdefault("hidden_sizes", 512)
        self.n_layers = self._config.setdefault("num_layer", 1)
        self.kernel_sizes = self._config.setdefault("kernel_sizes", 3)

        if type(self.hidden_sizes) != list:
            self.hidden_sizes = [self.hidden_sizes] * self.n_layers
        else:
            assert len(self.hidden_sizes) == self.n_layers, '`hidden_sizes` must have the same length as n_layers'
        if type(self.kernel_sizes) != list:
            self.kernel_sizes = [self.kernel_sizes] * self.n_layers
        else:
            assert len(self.kernel_sizes) == self.n_layers, '`kernel_sizes` must have the same length as n_layers'

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i - 1]

            cell = ConvGRUCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

    def forward(self, x, hidden=None):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).
        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''
        if not hidden:
            hidden = [None] * self.n_layers
        if len(x.shape) == 5:
            if hidden[0] is not None:
                for i,h in enumerate(hidden):
                    hidden[i] = hidden[i].view(-1, self.hidden_sizes[i], x.shape[-2], x.shape[-1])
        batch_size_x, x = flat_bts(x)
        input_ = x
        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        # return the last layer
        output = unflat_bts(batch_size_x, upd_hidden[-1])
        # if batch_size_x is not None:
        #     output_shape = upd_hidden[-1].shape
        #     upd_hidden[-1] = upd_hidden[-1].view(batch_size_x,-1,output_shape[-3],output_shape[-2],output_shape[-1])
        return output


if __name__ == "__main__":
    model = ConvGRU()

    x1 = torch.zeros((5, 8, 64, 64))
    output1 = model(x1)
    print(output1.shape)  # torch.Size([1, 16, 64, 64])

    x2 = torch.zeros((6, 5, 8, 64, 64))
    output2 = model(x2)

    print(output2.shape)  # torch.Size([1, 16, 64, 64])
