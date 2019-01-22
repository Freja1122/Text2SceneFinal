import os
import sys
import torch
import pprint
import datetime
import psutil
import math
from pattern.en import lemma
import objgraph
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
import pickle
import bcolz
import torchvision.utils as vutils
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter

from constants import *
from util.data import *
from util.losses import loss_dict
from util.optimizers import optimizer_dict
from util.toolkits import convert_kwargs, ProgressBar, TrainMonitor
from models.encoders import encoder_dict
from models.decoders import decoder_dict
from models.embeddings import embedding_dict
from models.projections import gen_proj
from models.convGRU import ConvGRU
from util.data import *

root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

font_path = "sources/AS/YaHei Mono.ttf"
prop = mfm.FontProperties(fname=font_path, size=FONT_SIZE)
mpl.rcParams["axes.unicode_minus"] = False

writer = SummaryWriter()

torch.random.manual_seed(142857)


class Wrapper(torch.nn.Module):
    def __init__(self, config, dataset, summarize=False):
        super(Wrapper, self).__init__()
        self.summarize = summarize
        self._config = config
        self._device = dataset.device
        self.type_num = dataset.n_types
        self._dataset = dataset
        train_config = convert_kwargs(self._config.setdefault("train_config", {}))
        self._text_enc_type = train_config.setdefault("text_encoder", "rnn")
        self._scene_enc_type = train_config.setdefault("scene_encoder", "convgru")
        self._dec_type = train_config.setdefault("decoder", "rnn")
        self._embedding_type = train_config.setdefault("embedding_type", "glove")
        self.hot_type = train_config.setdefault("hot_type", "nhot")
        self.y_scale, self.x_scale = dataset.scene_shape[:2]
        print("wrapper: Building embedding")
        if self._embedding_type == 'glove':
            embedding_weight = self._extract_glove_weights()
        else:
            embedding_weight = None
        self.text_embedding = embedding_dict[train_config.setdefault(
            "text_embedding", "basic")](self._config, dataset.n_vocab, embedding_weight)
        type_embedding = embedding_dict[train_config.setdefault(
            "type_embedding", "basic")](self._config, dataset.n_types)
        print("wrapper: Building encoder")
        text_encoder_config = convert_kwargs(self._config.setdefault("text_encoder_config", {}))
        self.text_encoder = encoder_dict[self._text_enc_type](text_encoder_config, self.text_embedding.embedding_size)
        scene_encoder_config = convert_kwargs(self._config.setdefault("scene_encoder_config", {}))
        self.scene_encoder = ConvGRU(scene_encoder_config)

        print("wrapper: Building decoders")
        '''[scene rep size, type_num, text_rep]'''
        self.decoder = decoder_dict[self._dec_type](
            self._config,
            self.scene_encoder.hidden_sizes[-1],
            self.type_num,
            self.text_encoder.output_size,
            self.text_embedding.embedding_size,
            type_embedding,
            dataset,
            self.scene_encoder
        )
        print("wrapper: Building pos predictors")
        print()

    def forward(self, batch):
        if not self.summarize:
            text = batch["text"]
            objects = batch.get("objects", None)
        else:
            self.train()
            text = batch[0]
            types = batch[1]
            step_scene_tensor = batch[2]
        '''get target info'''
        if not self.summarize:
            if objects is None:
                pos = types = flip_target = valid_lengths = pose_target = expression_target = None
            else:
                pos, types = objects[object_info_dict["pos"]], objects[object_info_dict["type"]]
                flip_target, valid_lengths = objects[object_info_dict["flip"]], torch.IntTensor(objects[-1])
                pose_target, expression_target = objects[object_info_dict["pose"]], objects[
                    object_info_dict["expression"]]
                step_scene_tensor = objects[object_info_dict['step_scene_tensor']]

        '''text embedding'''
        text_embedding = self.text_embedding(text)
        text_pad_mask = text.eq(INIT_WORD_DICT[PAD])
        text_valid_mask = text.ne(INIT_WORD_DICT[PAD])
        '''text encoder'''
        text_enc_output, text_final_state = self.text_encoder(text_embedding, text_pad_mask, return_all=True)
        '''scene encoder'''
        '''[batch_size, self.hidden_size] + list(spatial_size)'''
        sc_hidden_init = torch.cat([text_final_state[0][0], text_final_state[0][1]], dim=-1)  # 把h concat起来
        sc_hidden_init = self.decoder._extract_enc_feature(step_scene_tensor.shape, sc_hidden_init)
        scene_enc_output = self.scene_encoder(step_scene_tensor, hidden=[sc_hidden_init])
        '''type and attribute decoder'''
        result = self.decoder(
            scene_enc_output,
            text_embedding,
            text_enc_output,
            types,
            text_pad_mask)
        if not self.summarize:
            if pos is None:
                x_target = y_target = z_target = grid_target = grid_x_target = grid_y_target = None
            else:
                if GRID_TYPE == 'gridx':
                    x_target, y_target, z_target, grid_target = pos.split(1, dim=-1)
                elif GRID_TYPE == 'gridxy':
                    x_target, y_target, z_target, grid_x_target, grid_y_target = pos.split(1, dim=-1)
                x_target = x_target.to(torch.float32) / self.x_scale
                y_target = y_target.to(torch.float32) / self.y_scale
            # mask
            if valid_lengths is None:
                valid_mask = None
            else:
                max_valid_length = valid_lengths.max()
                arange = torch.arange(max_valid_length, dtype=torch.int32)
                valid_mask = (valid_lengths.unsqueeze(1) > arange).to(self._device)
            result["flip_target"] = flip_target
            result["mask"] = valid_mask
            result["text_mask"] = text_valid_mask
            result["x_target"] = x_target
            result["y_target"] = y_target
            result["z_target"] = z_target
            result["pose_target"] = pose_target
            result["expression_target"] = expression_target
            result["type_targets"] = types
            if GRID_TYPE == 'gridx':
                result["grid_target"] = grid_target
            elif GRID_TYPE == 'gridxy':
                result["grid_x_target"] = grid_x_target
                result["grid_y_target"] = grid_y_target
            if CLOSE_OBJ:
                result["type_logits"] = types
                result["type_targets"] = types
                result["type_samples"] = types
            # return result
            return result['grid']
    def _extract_glove_weights(self):
        emb_dim = 300
        glove_path = '/data1/bixiao/Code/glove'
        vectors_name = '6B.' + str(emb_dim) + '.dat'
        vectors = bcolz.open(os.path.join(glove_path, vectors_name))[:]
        words = pickle.load(open(glove_path + '/6B.' + str(emb_dim) + '_words.pkl', 'rb'))
        word2idx = pickle.load(open(glove_path + '/6B.' + str(emb_dim) + '_idx.pkl', 'rb'))

        glove = {w: vectors[word2idx[w]] for w in words}
        print('shape of glove embedding vector is :')
        print(glove['the'].shape)

        target_vocab = self._dataset._word2idx.keys()
        matrix_len = len(target_vocab)
        weights_matrix = np.zeros((matrix_len, emb_dim))
        words_found = 0
        hard_weights_pt = glove_path + '/hard_weights.pt'
        if os.path.exists(hard_weights_pt):
            hard_weights = torch.load(hard_weights_pt)
        else:
            hard_weights = {}

        for i, word in enumerate(target_vocab):
            try:
                weights_matrix[i] = glove[word]
                words_found += 1
            except KeyError:
                if hard_weights.get(word, None) is None:
                    hard_weights[word] = np.random.normal(scale=0.6, size=(emb_dim,))
                weights_matrix[i] = hard_weights[word]
        torch.save(hard_weights, hard_weights_pt)
        print(weights_matrix.shape)
        return weights_matrix


class Pipeline(torch.nn.Module):
    def __init__(self, config):
        super(Pipeline, self).__init__()
        self._config = config
        data_config = convert_kwargs(self._config.setdefault("data_config", {}))
        print()

        print('pipeline: making dataset')
        make_dataset = data_config.pop("make_dataset", False)
        self._dataset = Text2CVDataset(DATA_NAME, data_config).make_dataset(not make_dataset)
        self._data_loader = Text2CVDataLoader(
            self._dataset,
            shuffle=data_config.setdefault("shuffle", False),
            batch_size=data_config.setdefault("batch_size", 8)
        )
        self._name = "{}_{}".format(DATA_NAME, CONFIG_NAME)

        print('pipeline: making wrapper')
        self.wrapper = Wrapper(self._config, self._dataset)

        print('pipeline: making dir')
        train_config = convert_kwargs(self._config.setdefault("train_config", {}))
        self._begin_time = self._get_time_str()

        '''model saving dir'''
        self._model_saving_name = train_config.setdefault("model_saving_name", "model_{}".format(self._name))
        self._model_saving_folder_org = train_config.setdefault(
            "model_saving_folder", os.path.join("_trained_models", self._name))
        os.makedirs(self._model_saving_folder_org, exist_ok=True)
        self._model_saving_folder = os.path.join(self._model_saving_folder_org, "models" + self._begin_time)

        '''log'''
        self._logging_folder = train_config.setdefault("logging_folder", os.path.join("_logs", self._name))
        os.makedirs(self._logging_folder, exist_ok=True)
        self._logging_name = train_config.setdefault("logging_name", "{}.log".format(self._begin_time))

        '''export'''
        self._export_folder_org = train_config.setdefault("export_folder", os.path.join("_exports", self._name))
        os.makedirs(self._export_folder_org, exist_ok=True)
        self._export_folder = os.path.join(self._export_folder_org, "exports" + self._begin_time)

        print("pipeline: Building loss")
        loss = train_config.setdefault("loss", "basic")
        loss_config = convert_kwargs(train_config.setdefault("loss_config", {}))
        type_num_heads = self.wrapper.decoder.type_text_attention._num_heads if self.wrapper.decoder.type_text_attention is not None else None
        att_num_heads = self.wrapper.decoder.attribute_text_attention._num_heads if self.wrapper.decoder.attribute_text_attention is not None else None
        self.loss_fn = loss_dict[loss](loss_config,
                                       type_num_heads, att_num_heads)
        print("pipeline: Building optimizer")
        self.optimizer_name = train_config.setdefault("optimizer", "nag")
        optimizer_config = convert_kwargs(train_config.setdefault("optimizer_config", {}))
        if self.optimizer_name == "nag":
            optimizer_config.setdefault("lr", 0.1)
            optimizer_config.setdefault("momentum", 0.99)
            optimizer_config.setdefault("weight_decay", 1e-7)
            self.optimizer = optimizer_dict[self.optimizer_name](self.wrapper.parameters(), **optimizer_config)
        self._losses = self._best_global_step = None

        self._n_epoch = train_config.setdefault("n_epoch", 20)
        self._max_epoch = train_config.setdefault("max_epoch", 100)
        self._snapshot_ratio = train_config.setdefault("snapshot_ratio", 10)
        self._max_snapshot_step = int(train_config.setdefault("max_snapshot_step", MAX_SNAPSHOT_STEP))
        self._metric_name = train_config.setdefault("metric", "loss")
        if self._metric_name is None:
            self._metric_name = "loss"
        self._monitor = TrainMonitor(METRIC_SIGN[self._metric_name])
        self._clip_norm = train_config.setdefault("clip_norm", 0.1)
        self.real_metric = []
        self.real_metric_all_info = []

    def forward(self):
        os.makedirs(self._model_saving_folder, exist_ok=True)
        os.makedirs(self._export_folder, exist_ok=True)
        print('pipeline: model save dir:', self._model_saving_folder)
        print('pipeline: model log file:', self._logging_name)
        print('pipeline: model export dir:', self._export_folder)
        self._print_title()
        self._losses = {"train": [], "valid": []}
        self.wrapper.train()
        n_epoch = self._n_epoch
        global_step = i_epoch = 0
        best_score = self._best_global_step = None
        terminate = over_fitting_flag = False
        bar = ProgressBar(max_value=self._n_epoch, name="fit")
        n_iter = len(self._data_loader)
        snapshot_step = min(self._max_snapshot_step, int(n_iter * self._snapshot_ratio))
        while i_epoch < n_epoch:
            i_epoch += 1
            sub_bar = ProgressBar(max_value=n_iter, name="iter")
            for batch in self._data_loader:
                global_step += 1
                loss, outputs = self._step(batch, is_training=True)
                self._losses["train"].append(loss)
                del loss
                if global_step % snapshot_step == 0:
                    self.export(5, batch, outputs, name_prefix="train_sample", global_step=global_step)
                    with self._dataset.valid:
                        valid_batch = self._data_loader.next_batch
                    valid_loss, valid_metric, valid_outputs = self._step(valid_batch, is_training=False)
                    _, real_metric = self.export(5, valid_batch, valid_outputs, name_prefix="valid_sample",
                                                 global_step=global_step)
                    real_metric_mean = np.mean(np.array(real_metric), 0)
                    self._losses["valid"].append(valid_loss)
                    os.makedirs('data/metric/', exist_ok=True)
                    writer.add_scalars('train_valid_loss', {'train_loss': self._losses["train"][-1],
                                                            'valid_loss': self._losses["valid"][-1]},
                                       global_step=global_step)
                    writer.add_scalars('recall', {'recall': real_metric_mean[0],
                                                  'real_recall': real_metric_mean[1]},
                                       global_step=global_step)
                    writer.add_scalars('precision', {'precision': real_metric_mean[2],
                                                     'real_precision': real_metric_mean[3]},
                                       global_step=global_step)
                    del valid_loss
                    check_rs = self._monitor.check(valid_metric)
                    self._log("valid {} : {:6.4f} (step = {})".format(self._metric_name, valid_metric, global_step))
                    if check_rs["save_checkpoint"]:
                        self._log(check_rs["info"])
                        score = check_rs["score"]
                        if best_score is None or best_score < score:  # changed by bi
                            best_score, self._best_global_step = score, global_step
                            self._log('lead to best score: ' + str(score))
                            self.save()  # 没有参数默认存的是best global step
                        self.save(global_step)
                    del valid_batch, valid_outputs
                del batch, outputs
                sub_bar.update()
            if global_step >= snapshot_step:
                tr_loss, cv_loss = self._losses["train"][-1], self._losses["valid"][-1]
                self._log("epoch - {:4d} ; train loss - {:6.4f} ; valid loss - {:6.4f}".format(
                    i_epoch, tr_loss, cv_loss))
            sub_bar.terminate()
            if i_epoch == n_epoch and i_epoch < self._max_epoch and not self._monitor.info["terminate"]:
                self._monitor.flat_flag = True
                self._monitor.punish_extension()
                n_epoch = min(n_epoch + self._monitor.extension, self._max_epoch)
                self._log("Extending n_epoch to {}".format(n_epoch))
                bar.set_max(n_epoch)
            if i_epoch == self._max_epoch:
                terminate = True
                if not self._monitor.info["terminate"]:
                    if over_fitting_flag:
                        self._log("max_epoch reached")
                    else:
                        self._log(
                            "Model seems to be under-fitting but max_epoch reached. "
                            "Increasing max_epoch may improve performance"
                        )
            if terminate:
                bar.terminate()
                break
            bar.update()
        self._dump_best_step()
        bar.terminate()
        return self

    def _step(self, batch, is_training=True):
        if is_training:
            self.wrapper.train()
            self.optimizer.zero_grad()
            outputs = self.wrapper(batch)
        else:
            with torch.no_grad():
                # self.wrapper.eval()# training=false
                outputs = self.wrapper(batch)
        loss = self.loss_fn(batch, outputs, self.wrapper.decoder.obj_text_attention_weights,
                            self.wrapper.decoder.att_text_attention_weights)
        if is_training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.wrapper.parameters(), self._clip_norm)
            self.optimizer.step()
            return loss.item(), outputs
        else:
            loss = loss.item()
            metric = None
            if self._metric_name == "loss":
                metric = -loss
            return loss, metric, outputs

    def load(self, validating=False):
        if not os.path.exists(self._model_saving_folder):  # test
            assert MODEL_DIR is not None, "model_dir should be given"
            self._model_saving_folder = os.path.join(self._model_saving_folder_org, MODEL_DIR)
        if USEBESTSTEP is True:
            global_step = None
        elif GLOBAL_STEP is not None:
            global_step = GLOBAL_STEP
        else:
            global_step = self.last_global_step()
        path = self._model_saving_path(global_step)
        self._plot_dir = path.replace('_trained_models/AS_basic/', '')
        print("Loading path : {}".format(path))
        self.wrapper.load_state_dict(torch.load(path))
        if not validating:
            self.wrapper.eval()
        return self

    def export(self, n, batch, outputs=None, export_folder=None, name_prefix=None,
               global_step=0, test=False):  # train的时候会传入global step，test的时候不会
        '''test的时候才需要在这里计算output, valid和train都是在forward中用step计算output 的'''
        if outputs is None:
            with torch.no_grad():
                # self.wrapper.eval()
                outputs = self.wrapper(batch)
        if export_folder is None:
            export_folder = os.path.join(self._export_folder, 'global_step' + str(global_step))
        os.makedirs(export_folder, exist_ok=True)
        self.wrapper.decoder.draw_gen_batch(outputs, batch, export_folder, name_prefix, n)
        '''calculate metric and dump in correct rate file when test else(valid) just write log'''
        '''get the gen sample result'''
        '''make sure target and gen have the same size'''
        '''calculate recall real_recall precision real_precision mis_rate, real_mis_rate, gt_real_rate'''
        real_metric = []
        types_batch = self._dataset._tensor2list(outputs["type_samples"])
        type_target = batch['objects'][object_info_dict['type']]
        for gen_types, gt_types, gt_text in zip(
                *[self._dataset._tensor2list(types_batch), self._dataset._tensor2list(type_target),
                  self._dataset._tensor2list(batch['text'])]):
            real_metric_all = self._cul_real_metric(gen_types, gt_types, gt_text)
            if not self.wrapper.training:
                self.real_metric_all_info.append(real_metric_all)
            real_metric.append(real_metric_all[0])
        if not self.wrapper.training or n == -1:
            self.real_metric += real_metric
            torch.save(self.real_metric, os.path.join(export_folder, "real_metric.pt"))
            torch.save(self.real_metric_all_info, os.path.join(export_folder, "real_metric_all_info.pt"))
            print('save real_metric.pt in : ' + export_folder)
            print('save real_metric_all_info.pt in : ' + export_folder)
        else:
            real_metric_mean = np.mean(np.array(real_metric), 0)
            self._log(
                'real matric: [recall, real_recall, precision, real_precision, real_mis_rate, gt_real_rate]')
            self._log(str(real_metric_mean))

        '''calculate attribute accuracy'''

        self.draw_org_pic(batch, export_folder, n, name_prefix, outputs)

        return outputs, real_metric

    def predict(self, state='test'):
        assert MODEL_DIR is not None, "MODEL_DIR should be given"
        if USEBESTSTEP is True:
            global_step = "best"
        elif GLOBAL_STEP == None:
            global_step = self.last_global_step()
        else:
            global_step = GLOBAL_STEP
        export_folder = os.path.join(os.path.join("_predictions", self._name),
                                     'md_' + MODEL_DIR + '_cp_' + str(global_step) + '_' + state)
        print('prediction dir: ' + export_folder)
        with self._dataset.test:
            test_num = self._dataset.n_test
            max_value = math.ceil(test_num / self._data_loader.batch_size)
            bar = ProgressBar(max_value=max_value, name="pred")
            for i, batch in enumerate(self._data_loader):
                if i >= max_value:
                    break
                self.export(-1, batch, export_folder=export_folder, name_prefix="batch_{}".format(i))
                bar.update()
        if self.real_metric is not None:
            print('real matric: [recall, real_recall, precision, real_precision, real_mis_rate, gt_real_rate]')
            print(np.mean(np.array(self.real_metric), 0))

    def draw_org_pic(self, batch, export_folder, n, name_prefix, outputs):
        '''draw org batch when valid and test'''
        org_scene_batch = batch.get("scene", None)
        pose_gt_batch = outputs["pose_target"][..., 0]
        expression_gt_batch = outputs["expression_target"][..., 0]
        fs_gt_batch = outputs["flip_target"][..., 0]
        xs_gt_batch = (outputs["x_target"][..., 0] * self.wrapper.x_scale).to(torch.int32)
        ys_gt_batch = (outputs["y_target"][..., 0] * self.wrapper.y_scale).to(torch.int32)
        zs_gt_batch, type_gt_batch = outputs["z_target"], self._dataset._tensor2list(outputs["type_targets"])
        if GRID_TYPE == 'gridxy':
            grid_x_gt_batch = outputs["grid_x_target"][..., 0]
            grid_y_gt_batch = outputs["grid_y_target"][..., 0]
            # grid_gt_batch = torch.cat() 待写
        if GRID_TYPE == 'gridx':
            grid_gt_batch = outputs["grid_target"][..., 0]
        gt_batches = (
            grid_gt_batch, xs_gt_batch, ys_gt_batch, zs_gt_batch, fs_gt_batch, type_gt_batch, pose_gt_batch,
            expression_gt_batch,
            org_scene_batch)
        for i, (grids, xs, ys, zs, fs, ts, ps, es, org_scene) in enumerate(zip(*gt_batches)):
            local_export_folder = os.path.join(export_folder, "{}_{}".format(name_prefix, i))
            os.makedirs(local_export_folder, exist_ok=True)
            canvas_file_xy = os.path.join(local_export_folder, "scene_gt_xy.png")
            canvas_file_grid = os.path.join(local_export_folder, "scene_gt_grid.png")
            self._dataset.draw(xs, ys, grids, zs, fs, ts, ps, es, save_file=canvas_file_xy)
            self._dataset.draw_from_grid(grids, zs, fs, ts, ps, es, save_file=canvas_file_grid)
            if 0 < n <= i + 1:
                break

    def _get_time_str(self):
        now = datetime.datetime.now()
        otherStyleTime = now.strftime("%Y%m%d.%H%M%S")
        return otherStyleTime

    def _cul_real_metric(self, gen_types, gt_types, gt_text_idx, use_lemma=True):
        '''calculate real gt type list'''
        '''calculate real gen type list'''
        # if 1 in gt_types:
        #     gt_types.remove(1)
        if 1 in gt_types:
            idx = gt_types.index(1)
            gt_types = gt_types[:idx]
        if 1 in gen_types:
            idx = gen_types.index(1)
            gen_types = gen_types[:idx]
        if use_lemma:
            gt_text_idx = self._get_basic_word_from_idxs(gt_text_idx)
            gt_text_idx = self._get_idx_of_word_list(gt_text_idx)
        gt_idx_list, gt_real_mask, gt_real_idx, gt_real_types = self._dataset._get_real_type_list(gt_types,
                                                                                                  gt_text_idx)
        gen_idx_list, gen_real_mask, gen_real_idx, gen_real_types = self._dataset._get_real_type_list(gen_types,
                                                                                                      gt_text_idx)
        gt_word_list = [self._dataset._idx2word.get(w) for w in gt_idx_list]
        gen_word_list = [self._dataset._idx2word.get(w) for w in gen_idx_list]
        '''calculate recall and precision'''
        recall = self._cal_recall(gen_types, gt_types)  # 和ground truth相比有多少被预测出来了
        real_recall = self._cal_recall(gen_types, gt_real_types)  # 和ground truth里面text出现的obj相比有多少被预测出来了
        precision = self._cal_precision(gen_types, gt_types)  # 和ground truth相比，有多少预测是准确的
        real_precision = self._cal_precision(gen_types, gen_real_types)  # 和text比较而不是ground truth
        gt_real_rate = gt_real_mask.count(1) / len(gt_real_mask)  # ground truth中有多少比例是在text中出现了的
        real_mis_list = self._dataset._list_difference(gt_real_types, gen_types)  # ground truth和text中都出现，但是没有预测到的
        mis_list = self._dataset._list_difference(gt_types, gen_types)  # ground truth出现，但是没有预测到的
        mis_rate = len(mis_list) / len(gt_types) if len(gt_types) != 0 else 0  # 错误的占错误的比例
        real_mis_rate = len(real_mis_list) / len(gt_types) if len(gt_types) != 0 else 0  # 真正错误的占错误的比例
        recall_precision = [recall, real_recall, precision, real_precision, mis_rate, real_mis_rate, gt_real_rate]
        two_mis_list = [mis_list, real_mis_list]
        word_masks = [gt_word_list, gt_real_mask, gen_word_list, gen_real_mask]
        return recall_precision, two_mis_list, word_masks

    def _get_basic_word_from_idxs(self, idx_list):
        raw_words = self._dataset.recover_texts(idx_list)
        basic_words = [lemma(w) for w in raw_words]
        return basic_words

    def _get_idx_of_word_list(self, word_list):
        return [self._dataset._word2idx.get(w, None) for w in word_list]

    def _cal_recall(self, genlist, gtlist, return_all=False):
        interlist = self._dataset._list_intersection(gtlist, genlist)
        recall = len(interlist) / len(gtlist) if len(gtlist) != 0 else 0
        if not return_all:
            return recall
        return recall, interlist

    def _cal_precision(self, genlist, gtlist, return_all=False):
        interlist = self._dataset._list_intersection(gtlist, genlist)
        precision = len(interlist) / len(genlist) if len(genlist) != 0 else 0
        if not return_all:
            return precision
        return precision, interlist

    def _log(self, msg):
        cpu_info = psutil.virtual_memory()
        with open(self.logging_path, "a") as f:
            if self.optimizer_name == "adam":
                f.write(
                    "{} - {} - lr: {} - memory usage: {}\n".format(
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        msg, self.optimizer.param_groups[-1]['lr'], cpu_info.percent))
            elif self.optimizer_name == "nag":
                f.write(
                    "{} - {} - lr: {} - memory usage: {}\n".format(
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        msg, self.optimizer.lr, cpu_info.percent))

    def _dump_best_step(self):
        os.mknod(os.path.join(self._model_saving_folder, "best_global_step-{}".format(self._best_global_step)))

    def _model_saving_path(self, step_name=None):
        if step_name == None:
            chechpoint_name = "best_global_step.pt"
        else:
            chechpoint_name = "{}-{}.pt".format(self._model_saving_name, step_name)
        return os.path.join(self._model_saving_folder, chechpoint_name)

    @property
    def logging_path(self):
        file = os.path.join(self._logging_folder, self._logging_name)
        if not os.path.isfile(file):
            os.mknod(file)
        return file

    def last_global_step(self):
        if self._best_global_step is not None:
            return self._best_global_step
        max_step = 0
        for file in os.listdir(self._model_saving_folder):
            if file.startswith(self._model_saving_name):
                max_step = max(max_step, int(file[len(self._model_saving_name) + 1:-3]))
        return max_step

    def _print_title(self):
        str = "-" * 60 + "\n"
        str += "-" * 10 + "Training for: " + TRAIN_INFO + "-" * 10 + "\n"
        str += "-" * 18 + "GPU : " + GPU_ID + "-" * 18 + "\n"
        str += "-" * 60 + "\n"
        self._log(str)
        print(str)

    def save(self, global_step=None):
        if global_step == None:
            global_step = self._best_global_step
            state = 'best'
        else:
            state = 'normal'
        path = self._model_saving_path(global_step)
        torch.save(self.wrapper.state_dict(), path)
        self._log("Save {} step: {} in file: {}".format(state, global_step, path))
        return self

    def _visualize_attention(self, src, gen, gt_fetch, name, weights, figsize, prefix, file_name):
        n_src, n_gen = len(src), len(gen)
        w_n = len(weights)
        row = w_n
        col = 1
        plt.figure(figsize=(figsize[0] * col, figsize[1] * row))
        for h, weight_org in enumerate(weights):
            sum_gen = np.sum(weight_org, axis=1)
            sum_src = np.sum(weight_org, axis=0)
            weight = weight_org
            # weight = weight_org / weight_org.max()
            # fig, ax = plt.subplots(figsize=figsize)
            temp_str = str(row) + str(col) + str(h + 1)
            ax = plt.subplot(int(temp_str))
            ax.imshow(weight)
            ax.set_xticks(np.arange(n_src))
            ax.set_yticks(np.arange(n_gen))
            ax.set_xticklabels(self._transform(src), fontproperties=prop)
            ax.set_yticklabels(self._transform(gen), fontproperties=prop)
            for i in range(n_gen):
                ax.text(n_src, i, "{:4.4f}".format(sum_gen[i]),
                        ha="center", va="center", color="b", fontproperties=prop)
                if i < len(gt_fetch):
                    ax.text(-2, i, "{}".format(gt_fetch[i]),
                            ha="left", va="bottom", color="b", fontproperties=prop)
                for j in range(n_src):
                    if not i < weight.shape[0]:
                        print(weight.shape, n_gen, n_src)
                        break
                    ax.text(j, i, "{:4.4f}".format(weight[i, j]),
                            ha="center", va="center", color="w", fontproperties=prop)
            for j in range(n_src):
                ax.text(j, -1, "{:4.4f}".format(sum_src[j]),
                        ha="center", va="center", color="b", fontproperties=prop)
            ax.set_title(name + '_head{}'.format(h))
            # plt.tight_layout()
        folder = self._config.setdefault("plots_folder", "plots")
        folder = os.path.join(folder, self._plot_dir + "_{}".format(prefix))
        print('save plots in: ' + folder)
        os.makedirs(folder, exist_ok=True)
        print("S : {}".format("".join(src)))
        print("P : {}".format("".join(",".join([str(g) for g in gen]))))
        print("Saving figure '{}' to '{}'".format(file_name, folder))
        plt.savefig(os.path.join(folder, file_name), dpi=DPI)

    def visualize_attention(self, prefix='train', max_len=20, n_plots=5, figsize=(24, 18)):
        end, count = False, 0
        prefix = prefix
        with torch.no_grad():
            visualize_batch_num = 0
            while not end:
                if prefix == 'valid':
                    with self._dataset.valid:
                        batch = self._data_loader.next_batch
                elif prefix == 'test':
                    self.wrapper.eval()
                    with self._dataset.test:
                        batch = self._data_loader.next_batch
                else:
                    batch = self._data_loader.next_batch
                outputs = self.wrapper(batch)
                visualize_batch_num += 1
                fetches, names = [], []
                '''
                ['obj_text_attention_weights', 'att_text_attention_weights']
                '''
                weight_name = ['obj_text_attention_weights', 'att_text_attention_weights']
                title_name = ['obj_text_weights', 'att_text_weights']
                for i, name in enumerate(title_name):
                    w = getattr(self.wrapper.decoder, weight_name[i])
                    if w is not None:
                        f = w[0].data.cpu().numpy()
                        fetches.append(f)
                        del f
                        names.append(name)
                    if not fetches:
                        print("No attentions could be visualized")
                        return self
                sum_y = np.sum(fetches[0][0], 1)
                x_batch = batch['text'].cpu()
                pred_batch = outputs['type_samples'].data.cpu()
                gt_batch = outputs['type_targets'].data.cpu()
                x_batch_starts = self.find_after_last_pad(x_batch, INIT_WORD_DICT[PAD])
                pred_batch_ends = self.find_first_pad(pred_batch, INIT_WORD_DICT[EOS])
                gt_batch_ends = self.find_first_pad(gt_batch, INIT_WORD_DICT[EOS])
                x_batch_list = x_batch.data.cpu().numpy().tolist()
                pred_batch_list = pred_batch.data.cpu().numpy().tolist()
                gt_batch_list = gt_batch.data.cpu().numpy().tolist()
                source = [self._dataset.recover_texts(x) for x in x_batch_list]  # 恢复文字
                generated = [self._dataset.recover_types(pred_batch_list[i], source[i]) for i in
                             range(0, len(pred_batch_list))]  # 恢复素材
                gt_types = [self._dataset.recover_types(gt_batch_list[i], source[i]) for i in
                            range(0, len(gt_batch_list))]  # 恢复素材
                head_num = self.wrapper.decoder.attribute_text_attention._num_heads
                sample_num = len(source)
                for i, (src, gen) in enumerate(zip(source, generated)):  # visualize
                    count += 1
                    src_start, gen_len = x_batch_starts[i], pred_batch_ends[i]
                    for j, name in enumerate(names):
                        local_fetch_heads = []
                        gt_fetch = gt_types[i]
                        for h in range(0, head_num):
                            local_fetch = fetches[j][i + sample_num * h]  # 第i个sample 每head-num个sample共享一对输入输出数据
                            local_fetch_head = local_fetch[:gen_len, src_start:]
                            local_fetch_heads.append(local_fetch_head.data)
                            if h == 0:
                                local_fetch_heads_avg = local_fetch_head.copy()
                            else:
                                local_fetch_heads_avg += local_fetch_head
                            del local_fetch_head
                        file_name = '{}_{}_{}.png'.format(visualize_batch_num, i, name)
                        self._visualize_attention(src, gen, gt_fetch, "{}_{}".format(prefix, name), local_fetch_heads,
                                                  figsize, prefix, file_name)
                        if head_num != 1:
                            file_avg_name = '{}_{}_{}_avg.png'.format(visualize_batch_num, name)
                            self._visualize_attention(src, gen, gt_fetch, "{}_{}_avg".format(prefix, name),
                                                      [local_fetch_heads_avg / head_num],
                                                      figsize, prefix, file_avg_name)
                            del local_fetch_heads_avg
                    if count >= n_plots:
                        break
                if count >= n_plots:
                    break
        return self

    def find_after_last_pad(self, arr, pad_value=0):  # 第一次出现0就算停止了
        pad_mask_plus = np.zeros((arr.shape[0], arr.shape[1] + 1), dtype=np.int32)
        pad_mask_plus.fill(pad_value)
        pad_mask_plus[:arr.shape[0], :arr.shape[1]] = arr
        pad_idx = np.where(pad_mask_plus != pad_value)
        row, idx = np.unique(pad_idx[0], return_index=True)
        return pad_idx[1][idx]

    def find_first_pad(self, arr, pad_value=0):  # 第一次出现0就算停止了
        pad_mask_plus = np.zeros((arr.shape[0], arr.shape[1] + 1), dtype=np.int32)
        pad_mask_plus.fill(pad_value)
        pad_mask_plus[:arr.shape[0], :arr.shape[1]] = arr
        pad_idx = np.where(pad_mask_plus == pad_value)
        row, idx = np.unique(pad_idx[0], return_index=True)
        return pad_idx[1][idx]

    @staticmethod
    def _transform(l):
        s = set("，。、？！“”‘’：；")
        return ["" if w in s else w for w in l]


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot


import torch
from torch.autograd import Variable
import torch.nn as nn
from graphviz import Digraph
import json

if __name__ == '__main__':
    data_name = DATA_NAME
    config_name = CONFIG_NAME
    with open(os.path.join("../configs", data_name, "{}.json".format(config_name))) as f:
        config = json.load(f)
    datasets = torch.load('pipdataset.pt')
    batch = torch.load('batch.pt')
    net = Wrapper(config, datasets, summarize=False)
    y = net(batch)
    g = make_dot(y)
    g.view()

    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("该层参数和：" + str(l))
        k = k + l
    print("总参数数量和：" + str(k))



# if __name__ == '__main__':
#     config = ''
#     data_name = DATA_NAME
#     config_name = CONFIG_NAME
#     import json
#
#     with open(os.path.join("../configs", data_name, "{}.json".format(config_name))) as f:
#         config = json.load(f)
#     datasets = torch.load('pipdataset.pt')
#     batch = torch.load('batch.pt')
#     config["test"] = config["predict"] = False
#     config["train"] = True
#     wrapper = Wrapper(config, datasets, summarize=True)
#     text = batch["text"]
#     objects = batch["objects"]
#     types = objects[object_info_dict['type']]
#     step_scene_tensor = objects[object_info_dict['step_scene_tensor']]
#     with SummaryWriter() as w:
#         w.add_graph(wrapper, ([text, types, step_scene_tensor]))

    # pip = Pipeline(config)
    #
    # with SummaryWriter() as w:
    #     for batch in pip._data_loader:
    #         torch.save(pip._dataset, 'pipdataset.pt')
    #         torch.save(batch, 'batch.pt')
    #         # dummy_input = batch.copy()
    #         # dummy_input[object_info_dict['source']] = []
    #         # dummy_input[object_info_dict['step_scene']] = []
    #         # # print('text', dummy_input["text"])
    #         # # print('objects', dummy_input.get("objects", None))
    #         # w.add_graph(pip.wrapper, [dummy_input["text"], dummy_input["objects"]], )
    #         break
