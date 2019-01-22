import os
import json
import torch
os.environ["PATH"] += os.pathsep + '/home/bixiao/anaconda3/bin'


from constants import *
from models.pipeline import Pipeline
from util.toolkits import convert_kwargs
from constants import MODEL_DIR, GLOBAL_STEP

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

torch.random.manual_seed(142857)


class Run:
    def __init__(self, data_name, config_name):
        self._data_name, self._config_name = data_name, config_name
        with open(os.path.join("configs", data_name, "{}.json".format(config_name))) as f:
            config = json.load(f)
        self._pipe, self._config = None, config

    def train(self):
        self._config["test"] = self._config["predict"] = False
        self._pipe = Pipeline(self._config)()

    def test(self, *dtypes):
        for dtype in dtypes:
            data_config = convert_kwargs(self._config.setdefault("data_config", {}))
            data_config["training"] = dtype == "train" or dtype == "valid"
            # data_config["training"] = False
            if self._pipe is None:
                self._pipe = Pipeline(self._config)
            print('-' * 30 + 'load' + '-' * 30)
            self._pipe.load(validating=data_config["training"])
            print('-' * 30 + 'predict' + '-' * 30)
            self._pipe.predict(state=dtype)

    def visualize_attention(self, dtype, max_len, n_plots, figsize):
        # pass
        # self._config["predict"] = True
        # self._config["test"] = dtype == "test"
        if self._pipe is None:
            self._pipe = Pipeline(self._config)
        # valid = dtype == "valid"
        self._pipe.load(validating=True).visualize_attention(prefix='test')


if __name__ == '__main__':
    run = Run(DATA_NAME, CONFIG_NAME)
    if VISUALIZE:
        for d in DTYPES:
            run.visualize_attention(d, max_len=MAX_LEN, n_plots=N_PLOTS, figsize=(MAX_LEN // 2, MAX_LEN // 2))
    else:
        if "train" in DTYPES:
            run.train()
        run.test(*DTYPES)
