import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)
import torch
import numpy as np

from constants import DEBUG
from util.toolkits import convert_kwargs
from models.attentions import Attention
from models.projections import gen_proj


class EncoderBase(torch.nn.Module):
    def __init__(self, config):
        super(EncoderBase, self).__init__()
        self._config = config
        self.self_attention = self._attention_weights = self._self_attention_config = None
        self._device = torch.device("cuda" if torch.cuda.is_available() and not DEBUG else "cpu")

    def forward(self, x_embedding, mask):
        pass

    @property
    def attention_weights(self):
        if self._attention_weights is None:
            return []
        return list(self._attention_weights)

    @property
    def attention_names(self):
        # Encoder attention name should contain "self"
        if self._attention_weights is None:
            return []
        return ["self_attn_{}".format(i) for i in range(len(self.rnns))]


class RNNEncoder(EncoderBase):
    def __init__(self, config, input_dim):
        super(RNNEncoder, self).__init__(config)
        rnn_name = config.setdefault("rnn", "GRU")
        rnn_config = convert_kwargs(config.setdefault("rnn_config", {}))
        rnn_config.setdefault("batch_first", True)
        num_layers = rnn_config.setdefault("num_layers", 2)
        self.hidden_size = rnn_config.setdefault("hidden_size", 256)
        bidirectional = rnn_config.setdefault("bidirectional", True)
        if not bidirectional:
            self.bidirectional_projection = None
        else:
            self.bidirectional_projection = None
            '''
            gen_proj(rnn_config, "bi_proj_config", self.hidden_size * 2, self.hidden_size, False)
            '''
        actual_rnn_config = rnn_config.copy()
        for redundant in ("num_layers", "bi_proj_config"):
            actual_rnn_config.pop(redundant)
        self.rnns = torch.nn.ModuleList([
            getattr(torch.nn, rnn_name)(input_dim, **actual_rnn_config).to(self._device)
            for _ in range(num_layers)
        ])
        self._n_output_history = config.setdefault("n_output_history", 1)
        self._pooling_method = config.setdefault("pooling_method", None)
        self_attention_config = convert_kwargs(config.setdefault("self_attention_config", {}))
        if self_attention_config is not None:
            self.self_attention = Attention(self_attention_config, self.hidden_size)
        if self._pooling_method is None and self._n_output_history != 1:
            self._n_features = self._n_output_history * self.hidden_size
            self._builder = torch.nn.Linear(self._n_features, self.hidden_size).to(self._device)

        if not bidirectional:
            self.output_size = rnn_config.setdefault("output_size", self.hidden_size)
        else:
            self.output_size = rnn_config.setdefault("output_size", self.hidden_size * 2)

    def forward(self, x_embedding, pad_mask, init_state=None, return_all=False, n_repeat=None):
        final_state = init_state
        if n_repeat is not None:
            if isinstance(n_repeat, int):
                x_embedding = x_embedding.unsqueeze(1).repeat(1, n_repeat, 1)
            else:
                assert len(x_embedding) == len(n_repeat)
                max_repeat = n_repeat.max()
                new_embedding = x_embedding.new(len(n_repeat), max_repeat, x_embedding.shape[1]).fill_(0.)
                position_encoding = self.get_sinusoid_encoding_table(max_repeat.item())
                for i, n in enumerate(n_repeat):
                    new_embedding[i][max_repeat-n:] = x_embedding[i] + position_encoding[-n:]
                x_embedding = new_embedding
        net = x_embedding
        self._attention_weights = None
        for rnn in self.rnns:
            net, final_state = rnn(net, init_state)
            if self.bidirectional_projection is not None:
                net = self.bidirectional_projection(net)
            if self.self_attention is not None:
                net = self._apply_self_attention(net, x_embedding, pad_mask)
        if return_all:
            return net, final_state
        return self.build_feature(net)

    def get_sinusoid_encoding_table(self, n_position):
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / self.hidden_size)

        def get_pos_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(self.hidden_size)]

        sinusoid_table = np.array([get_pos_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).to(self._device)

    def build_feature(self, outputs):
        if self._n_output_history == 1:
            return outputs[..., -1, :]
        net = outputs[..., -self._n_output_history:, :]
        return self._builder(net.contiguous().view(-1, self._n_features))

    @property
    def num_units(self):
        return self.hidden_size


encoder_dict = {
    "rnn": RNNEncoder,
"convgru": RNNEncoder
}

__all__ = ["encoder_dict"]
