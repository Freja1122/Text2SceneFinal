import time
import math
import torch
import os
from constants import *

def grouped(iterable, n):
    return list(zip(*[iter(iterable)] * n))


convert_dict = {"none": None, "true": True, "false": False}

# kwargs = {dict}{'clear_cache':False, 'training':True}
# config里面的string全部换成dict，包括value中有dict的部分
def convert_kwargs(kwargs):
    # 把string换成dict
    if not isinstance(kwargs, dict):
        return convert_dict.get(kwargs, kwargs)
    for k, v in kwargs.items():
        if isinstance(v, dict):
            convert_kwargs(v)
        else:
            kwargs[k] = convert_dict.get(str(v), v)
    return kwargs

def flat_bts(x):
    batch_size_x = None
    if len(x.shape) == 5:
        batch_size_x = x.shape[0]
        x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
    return batch_size_x, x

def unflat_bts(batch_size_x, output):
    if batch_size_x is not None:
        output_shape = output.shape
        output = output.view(batch_size_x, -1, output_shape[-3], output_shape[-2], output_shape[-1])
    return output

class Pruner(torch.nn.Module):
    def __init__(self, config):
        super(Pruner, self).__init__()
        method = config.setdefault("method", "hard_prune")
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.alpha = dtype([config.setdefault("alpha", 1e-2 if method == "soft_prune" else 1e-4)])
        self.beta = dtype([config.setdefault("beta", 1.)])
        self.gamma = dtype([config.setdefault("beta", 1.)])
        self.max_ratio = dtype([config.setdefault("max_ratio", 1.)])
        self.eps = dtype([config.setdefault("eps", 1e-12)])

    def forward(self, w):
        w_abs = torch.abs(w)
        w_abs_mean = torch.mean(w_abs)
        log_w = torch.log(torch.max(self.eps, w_abs / (w_abs_mean * self.gamma)))
        if self.max_ratio > 0:
            log_w = torch.min(self.max_ratio, self.beta * log_w)
        mask = torch.max(self.alpha / self.beta * log_w, log_w)
        w = w * mask; del w_abs, w_abs_mean, log_w, mask
        return w


class ProgressBar:
    def __init__(self, min_value=0, max_value=None, min_refresh_period=0.5, width=30, name="", start=True):
        self._min, self._max = min_value, max_value
        self._task_length = int(max_value - min_value) if (
                min_value is not None and max_value is not None
        ) else None
        self._counter = min_value
        self._min_period = min_refresh_period
        self._bar_width = int(width)
        self._bar_name = " " if not name else " # {:^12s} # ".format(name)
        self._terminated = False
        self._started = False
        self._ended = False
        self._current = 0
        self._clock = 0
        self._cost = 0
        if start:
            self.start()

    def _flush(self):
        if self._ended:
            return False
        if not self._started:
            print("Progress bar not started yet.")
            return False
        if self._terminated:
            if self._counter == self._min:
                self._counter = self._min + 1
            self._cost = time.time() - self._clock
            tmp_hour = int(self._cost / 3600)
            tmp_min = int((self._cost - tmp_hour * 3600) / 60)
            tmp_sec = self._cost % 60
            tmp_avg = self._cost / (self._counter - self._min)
            tmp_avg_hour = int(tmp_avg / 3600)
            tmp_avg_min = int((tmp_avg - tmp_avg_hour * 3600) / 60)
            tmp_avg_sec = tmp_avg % 60
            print(
                "\r" +
                "##{}({:d} : {:d} -> {:d}) Task Finished. "
                "Time Cost: {:3d} h {:3d} min {:6.4} s; Average: {:3d} h {:3d} min {:6.4} s ".format(
                    self._bar_name, self._task_length, self._min, self._counter - self._min,
                    tmp_hour, tmp_min, tmp_sec, tmp_avg_hour, tmp_avg_min, tmp_avg_sec
                ) + " ##\n", end=""
            )
            self._ended = True
            return True
        if self._counter >= self._max:
            self._terminated = True
            return self._flush()
        if self._counter != self._min and time.time() - self._current <= self._min_period:
            return False
        self._current = time.time()
        self._cost = time.time() - self._clock
        if self._counter > self._min:
            tmp_hour = int(self._cost / 3600)
            tmp_min = int((self._cost - tmp_hour * 3600) / 60)
            tmp_sec = self._cost % 60
            tmp_avg = self._cost / (self._counter - self._min)
            tmp_avg_hour = int(tmp_avg / 3600)
            tmp_avg_min = int((tmp_avg - tmp_avg_hour * 3600) / 60)
            tmp_avg_sec = tmp_avg % 60
        else:
            print()
            tmp_hour = 0
            tmp_min = 0
            tmp_sec = 0
            tmp_avg_hour = 0
            tmp_avg_min = 0
            tmp_avg_sec = 0
        passed = int(self._counter * self._bar_width / self._max)
        print(
            "\r" + "##{}[".format(
                self._bar_name
            ) + "-" * passed + " " * (self._bar_width - passed) + "] : {} / {}".format(
                self._counter, self._max
            ) + " ##  Time Cost: {:3d} h {:3d} min {:6.4} s; Average: {:3d} h {:3d} min {:6.4} s ".format(
                tmp_hour, tmp_min, tmp_sec, tmp_avg_hour, tmp_avg_min, tmp_avg_sec
            ) if self._counter != self._min else "##{}Progress bar initialized  ##".format(
                self._bar_name
            ), end=""
        )
        return True

    def set_min(self, min_val):
        if self._max is not None:
            if self._max <= min_val:
                print("Target min_val: {} is larger than current max_val: {}".format(min_val, self._max))
                return
            self._task_length = self._max - min_val
        self._counter = self._min = min_val

    def set_max(self, max_val):
        if self._min is not None:
            if self._min >= max_val:
                print("Target max_val: {} is smaller than current min_val: {}".format(max_val, self._min))
                return
            self._task_length = max_val - self._min
        self._max = max_val

    def update(self, new_value=None):
        if new_value is None:
            new_value = self._counter + 1
        if new_value != self._min:
            self._counter = self._max if new_value >= self._max else int(new_value)
            return self._flush()

    def start(self):
        if self._task_length is None:
            print("Error: Progress bar not initialized properly.")
            return
        self._current = self._clock = time.time()
        self._started = True
        self._flush()

    def terminate(self):
        self._terminated = True
        self._flush()


# noinspection PyTypeChecker
class TrainMonitor:
    def __init__(self, sign=1, snapshot_ratio=3, history_ratio=3, tolerance_ratio=2,
                 extension=5, std_floor=0.001, std_ceiling=0.01):
        self.sign = sign
        self.snapshot_ratio = snapshot_ratio
        self.n_history = int(snapshot_ratio * history_ratio)
        self.n_tolerance = int(snapshot_ratio * tolerance_ratio)
        self.extension = extension
        self.std_floor, self.std_ceiling = std_floor, std_ceiling
        self._scores = []
        self.flat_flag = False
        self._is_best = self._running_best = None
        self._running_sum = self._running_square_sum = None
        self._descend_increment = self.n_history * extension / 30

        self._over_fit_performance = math.inf
        self._best_checkpoint_performance = -math.inf
        self._descend_counter = self._flat_counter = self.over_fitting_flag = 0
        self.info = {"terminate": False, "save_checkpoint": True, "save_best": True, "info": None}

    def punish_extension(self):
        self._descend_counter += self._descend_increment

    def _update_running_info(self, last_score, n_history):
        if n_history < self.n_history or n_history == len(self._scores):
            if self._running_sum is None or self._running_square_sum is None:
                self._running_sum = self._scores[0] + self._scores[1]
                self._running_square_sum = self._scores[0] ** 2 + self._scores[1] ** 2
            else:
                self._running_sum += last_score
                self._running_square_sum += last_score ** 2
        else:
            previous = self._scores[-n_history - 1]
            self._running_sum += last_score - previous
            self._running_square_sum += last_score ** 2 - previous ** 2
        if self._running_best is None:
            if self._scores[0] > self._scores[1]:
                improvement = 0
                self._running_best, self._is_best = self._scores[0], False
            else:
                improvement = self._scores[1] - self._scores[0]
                self._running_best, self._is_best = self._scores[1], True
        elif self._running_best > last_score:
            improvement = 0
            self._is_best = False
        else:
            improvement = last_score - self._running_best
            self._running_best = last_score
            self._is_best = True
        return improvement

    def _handle_overfitting(self, last_score, res, std):
        if self._descend_counter == 0:
            self.info["save_best"] = True
            self._over_fit_performance = last_score
        self._descend_counter += min(self.n_tolerance / 3, -res / std)
        self.over_fitting_flag = 1

    def _handle_recovering(self, improvement, last_score, res, std):
        if res > 3 * std and self._is_best and improvement > std:
            self.info["save_best"] = True
        new_counter = self._descend_counter - res / std
        if self._descend_counter > 0 >= new_counter:
            self._over_fit_performance = math.inf
            if last_score > self._best_checkpoint_performance:
                self._best_checkpoint_performance = last_score
                if last_score > self._running_best - std:
                    self.info["save_checkpoint"] = True
                    self.info["info"] = (
                        "Current snapshot ({}) seems to be working well, "
                        "saving checkpoint in case we need to restore".format(len(self._scores))
                    )
            self.over_fitting_flag = 0
        self._descend_counter = max(new_counter, 0)

    def _handle_is_best(self):
        if self._is_best:
            self.info["terminate"] = False
            if self.info["save_best"]:
                self.info["save_checkpoint"] = True
                self.info["save_best"] = True
                self.info["info"] = (
                    "Current snapshot ({}) leads to best result we've ever had, "
                    "saving checkpoint since ".format(len(self._scores))
                )
                if self.over_fitting_flag:
                    self.info["info"] += "we've suffered from over-fitting"
                else:
                    self.info["info"] += "performance has improved significantly"

    def _handle_period(self, last_score):
        if len(self._scores) % self.snapshot_ratio == 0 and last_score > self._best_checkpoint_performance:
            self._best_checkpoint_performance = last_score
            self.info["terminate"] = False
            self.info["save_checkpoint"] = True
            self.info["info"] = (
                "Current snapshot ({}) leads to best checkpoint we've ever had, "
                "saving checkpoint in case we need to restore".format(len(self._scores))
            )

    def check(self, new_metric):
        last_score = new_metric * self.sign
        self.info["score"] = last_score
        self._scores.append(last_score)
        n_history = min(self.n_history, len(self._scores))
        if n_history == 1:
            return self.info
        improvement = self._update_running_info(last_score, n_history)
        self.info["save_checkpoint"] = True
        mean = self._running_sum / n_history
        std = math.sqrt(max(self._running_square_sum / n_history - mean ** 2, 1e-12))
        std = min(std, self.std_ceiling)
        if std < self.std_floor:
            if self.flat_flag:
                self._flat_counter += 1
        else:
            self._flat_counter = max(self._flat_counter - 1, 0)
            res = last_score - mean
            if res < -std and last_score < self._over_fit_performance - std:
                self._handle_overfitting(last_score, res, std)
            elif res > std:
                self._handle_recovering(improvement, last_score, res, std)
        if self._flat_counter >= self.n_tolerance * self.n_history:
            self.info["info"] = "Performance not improving"
            self.info["terminate"] = True
            return self.info
        if self._descend_counter >= self.n_tolerance:
            self.info["info"] = "Over-fitting"
            self.info["terminate"] = True
            return self.info
        self._handle_is_best()
        self._handle_period(last_score)
        del last_score
        return self.info


__all__ = [
    "grouped", "convert_kwargs",
    "Pruner", "ProgressBar", "TrainMonitor",
]
