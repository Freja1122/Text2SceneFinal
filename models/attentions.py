import os
import sys
import torch.nn.functional as F

root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)
import torch
import torch.nn.functional as functional

from constants import NEG_INF
from models.projections import gen_proj
from models.conv2dsame import Conv2dSame
from util.toolkits import flat_bts, unflat_bts
class SpatialAttention(torch.nn.Module):
    def __init__(self, config, input_dim, output_dim=None):
        super(SpatialAttention, self).__init__()
        self._config = config
        self._num_layer = self._config.setdefault("num_layer", 2)
        self._input_dim = input_dim
        self._hidden_sizes = []
        self._kernel_sizes = []
        cells = []
        for i in range(self._num_layer):
            name = 'layer' + str(i + 1)
            if i == 0:
                input_dim = self._input_dim
            else:
                input_dim = self._hidden_sizes[i - 1]
            layer_config = self._config.get(name)
            self._hidden_sizes.append(layer_config.get('hidden_size'))
            self._kernel_sizes.append(layer_config.get('kernel_size'))
            cell = Conv2dSame(input_dim, self._hidden_sizes[i], self._kernel_sizes[i])
            if torch.cuda.is_available():
                cell = cell.cuda()
            setattr(self, name, cell)
            cells.append(getattr(self, name))
        self.cells = cells

        if output_dim is None:
            self.output_dim = input_dim
        else:
            self.output_dim = output_dim

    def forward(self, x, v):
        batch_size_x, x = flat_bts(x)

        input_ = x
        upd_hidden = []

        for layer_idx in range(self._num_layer):
            cell = self.cells[layer_idx]
            if layer_idx != self._num_layer - 1:#not the last layer
                upd_cell_hidden = F.relu(cell(input_))
            else:
                upd_cell_hidden = cell(input_)
            upd_hidden.append(upd_cell_hidden)
            input_ = upd_cell_hidden
        final_hidden = upd_hidden[-1]
        final_hidden = unflat_bts(batch_size_x, final_hidden)
        # if batch_size_x is not None:
        #     output_shape = final_hidden.shape
        #     final_hidden = final_hidden.view(batch_size_x, -1, output_shape[-3], output_shape[-2], output_shape[-1])
        return final_hidden * v  # 直接乘不知道对不对


class Attention(torch.nn.Module):
    def __init__(self, config, q_input_dim, k_input_dim, v_input_dim, output_dim):
        super(Attention, self).__init__()
        self._config = config
        self._hidden_size = self._config.setdefault("hidden_size", 256)
        self._num_heads = self._config.setdefault("num_heads", 4)
        if self._hidden_size % self._num_heads != 0:
            raise ValueError("Hidden size must be evenly divisible by the number of heads.")
        self._attention_dropout = self._config.setdefault("attention_dropout", 0.1)
        self._normalize_qk = self._config.setdefault("normalize_qk", False)
        proj_config_name = "projection_config"
        # self.q_proj = gen_proj(self._config, proj_config_name,
        #                        input_dim, self._hidden_size, False, "mlp")
        # self.k_proj = gen_proj(self._config, proj_config_name,
        #                        input_dim, self._hidden_size, False, "mlp")
        # self.v_proj = gen_proj(self._config, proj_config_name,
        #                        input_dim, self._hidden_size, False, "mlp")
        self.q_proj = gen_proj(self._config, proj_config_name,
                               q_input_dim, self._hidden_size, False)
        self.k_proj = gen_proj(self._config, proj_config_name,
                               k_input_dim, self._hidden_size, False)
        self.v_proj = gen_proj(self._config, proj_config_name,
                               v_input_dim, self._hidden_size, False)
        transform_config_name = "transform_config"
        self.output_transform = gen_proj(
            self._config, transform_config_name,
            self._hidden_size, output_dim, False, "mlp"
        )

    @property
    def depth(self):
        return self._hidden_size // self._num_heads

    def forward(self, q, k, v, pad_mask, return_weights=False):
        k_equal_q = k is None
        if self.q_proj is not None:
            q = self.q_proj(q)
        if k_equal_q:
            k = q
        elif self.k_proj is not None:
            k = self.k_proj(k)
        if self.v_proj is not None:
            v = self.v_proj(v)
        if self._num_heads > 1:
            q = self._split_heads(q)
            if not k_equal_q:
                k = self._split_heads(k)
            v = self._split_heads(v)
        if self._normalize_qk:
            q = q / torch.norm(q, dim=-1).unsqueeze(-1)
            if not k_equal_q:
                k = k / torch.norm(k, dim=-1).unsqueeze(-1)
        if k_equal_q:
            k = q
        depth = (self._hidden_size // self._num_heads)
        q = q * depth ** -0.5

        # pad_mask : [B, T]
        # q, k, v  : [num_heads x B, T, depth]

        logits = torch.bmm(q, k.transpose(1, 2)) + self._get_attention_bias(pad_mask)
        weights = functional.softmax(logits, dim=-1)
        functional.dropout(weights, self._attention_dropout, training=self.training)
        attention_output = torch.bmm(weights, v)
        attention_output = self._combine_heads(attention_output)
        logits = torch.mean(logits.view(self._num_heads, -1, *logits.shape[1:]), dim=0)
        if self.output_transform is not None:
            attention_output = self.output_transform(attention_output)
        if not return_weights:
            return attention_output
        return attention_output, weights

    def _get_attention_bias(self, pad_mask):
        return (pad_mask.to(torch.float32) * NEG_INF).repeat([self._num_heads, 1]).unsqueeze(1)

    def _split_heads(self, x):
        time_step = x.shape[1]
        return (
            x.view(-1, time_step, self._num_heads, self.depth)
                .transpose(1, 2).contiguous()
                .view(-1, time_step, self.depth)
        )

    def _combine_heads(self, x):
        time_step = x.shape[1]
        return (
            x.view(-1, self._num_heads, time_step, self.depth)
                .transpose(1, 2).contiguous()
                .view(-1, time_step, self._hidden_size)
        )


attention_dict = {
    "text_attention": Attention,
    "spatial_attention": SpatialAttention
}
