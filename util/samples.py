import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)
import torch


class SampleBase:
    def __init__(self, config):
        self._config = config
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, batch_logits):
        if len(batch_logits.shape) == 2:
            return self._sample_single(batch_logits)
        return self._sample_multiple(batch_logits)

    def _sample_single(self, batch_logits):
        pass

    def _sample_multiple(self, batch_logits):
        samples = []
        batch_logits = batch_logits.transpose(0, 1)
        for t, logits in enumerate(batch_logits):
            s=self._sample_single(logits)
            samples.append(s)
            del s
        return torch.cat(samples, dim=1).to(self._device)


class Multinomial(SampleBase):
    def _sample_single(self, batch_logits):
        return torch.multinomial(batch_logits, 1)


sample_dict = {
    "multinomial": Multinomial
}

__all__ = ["sample_dict"]
