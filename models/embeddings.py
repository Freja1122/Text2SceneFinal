import torch

from util.toolkits import convert_kwargs
from constants import BOS, INIT_WORD_DICT, DEBUG


class EmbeddingBase(torch.nn.Module):
    def __init__(self, config, n, glove_weight=None):
        super(EmbeddingBase, self).__init__()
        self._config, self._n = config, n
        self._device = torch.device("cuda" if torch.cuda.is_available() and not DEBUG else "cpu")
        embedding_config = convert_kwargs(self._config.setdefault("embedding_config", {}))
        if glove_weight is not None:
            self.embedding_size = glove_weight.shape[-1]
            self.pick = torch.nn.Embedding(n, self.embedding_size).to(self._device)
            # self.pick.load_state_dict({'weight': glove_weight})
            self.pick.weight.data.copy_(torch.tensor(glove_weight))
            self.pick.weight.requires_grad = True
        else:
            self.embedding_size = embedding_config.setdefault("embedding_size", 256)
            self.pick = torch.nn.Embedding(n, self.embedding_size).to(self._device)
        # self.pick_type = torch.nn.Embedding(n, n).to(self._device)


    def forward(self, x):
        return self.pick(x)

    @property
    def n_vocab(self):
        return self._n

    def bos(self, n):
        return self.pick(torch.LongTensor([INIT_WORD_DICT[BOS]] * n).to(self._device)).unsqueeze(1)

    # def bos_type(self, n):
    #     return self.pick_type(torch.LongTensor([INIT_WORD_DICT[BOS]] * n).to(self._device)).unsqueeze(1)


class NormEmbedding(EmbeddingBase):
    def forward(self, x):
        x = super(NormEmbedding, self).forward(x)
        return x / x.norm(dim=-1).unsqueeze(-1)


embedding_dict = {
    "basic": EmbeddingBase, "norm": NormEmbedding
}

__all__ = ["embedding_dict"]
