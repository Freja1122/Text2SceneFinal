import os
import sys
from collections import defaultdict
import torchvision.transforms as transforms
import torch.nn.functional as F

root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)
import torch
import torch.nn.functional as functional

from constants import *
from util.samples import sample_dict
from util.toolkits import convert_kwargs
from models.projections import gen_proj
from models.attentions import attention_dict
from models.conv2dsame import Conv2dSame


class DecoderBase(torch.nn.Module):
    def __init__(self, config, embedding):
        super(DecoderBase, self).__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() and not DEBUG else "cpu")
        self.attention = self._attention_weights = None
        self._config = config
        self.embedding = embedding

    def forward(self,
                scene_rep,
                text_embedding,
                text_enc_output,
                target,
                pad_mask):
        pass

    @property
    def attention_weights(self):
        if self._attention_weights is None:
            return []
        return list(self._attention_weights)

    @property
    def attention_names(self):
        if self._attention_weights is None:
            return []
        attention_names = ["attn_{}".format(i) for i in range(len(self.rnns))]
        return attention_names

    def run_train_next(self, net, x_embedding, pad_mask, enc_feature, enc_outputs):
        pass

    def run_predict_next(self, net, x_embedding, pad_mask, enc_feature, enc_outputs, clear_state):
        pass


class GeneralDecoder(torch.nn.Module):
    def __init__(self, config, input_dim, output_dim=None):
        super(GeneralDecoder, self).__init__()
        self._config = config
        self._num_layer = self._config.setdefault("num_layer", 4)
        self._cell_type = self._config.setdefault("cell_type", "cnn")
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

            if self._cell_type == 'cnn':
                self._kernel_sizes.append(layer_config.get('kernel_size'))
                cell = Conv2dSame(input_dim, self._hidden_sizes[i], self._kernel_sizes[i])

            elif self._cell_type == 'linear':
                cell = gen_proj(
                    self._config, name,
                    input_dim=input_dim,
                    output_dim=self._hidden_sizes[i],
                    use_bias=True,
                    dtype="linear"
                )
            if torch.cuda.is_available():
                cell = cell.cuda()
            setattr(self, name, cell)
            cells.append(getattr(self, name))
        self.cells = cells

        # if output_dim is None:
        #     self.output_dim = input_dim

    def forward(self, x):
        batch_size_x = None
        if len(x.shape) == 5:
            batch_size_x = x.shape[0]
            x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        input_ = x
        upd_hidden = []
        for layer_idx in range(self._num_layer):
            cell = self.cells[layer_idx]
            if layer_idx != self._num_layer - 1:  # not the last layer
                upd_cell_hidden = F.relu(cell(input_))
            else:
                upd_cell_hidden = cell(input_)
            upd_hidden.append(upd_cell_hidden)
            input_ = upd_cell_hidden
        final_hidden = upd_hidden[-1]
        if batch_size_x is not None:
            output_shape = final_hidden.shape
            final_hidden = final_hidden.view(batch_size_x, -1, output_shape[-3], output_shape[-2], output_shape[-1])
        return final_hidden


class RNNDecoder(DecoderBase):
    '''[scene rep size, type_num, text_rep]'''

    def __init__(self, config, sc_enc_hidden_sizes, type_num, text_enc_hidden_size,
                 text_emb_size, type_embedding, dataset, scene_enc):
        super(RNNDecoder, self).__init__(config, type_embedding)
        self.type_num = type_num
        self._states = None
        self._dataset = dataset
        self._scene_enc = scene_enc
        self.decoder_config = convert_kwargs(self._config.setdefault("decoder_config", {}))
        self.max_len = self.decoder_config.setdefault("max_len", 20)
        self.hot_type = self.decoder_config.setdefault("hot_type", 'onehot')
        self.y_scale, self.x_scale = dataset.scene_shape[:2]
        text_input_size = text_enc_hidden_size + text_emb_size
        self.init_attention_config("spatial_attention_config",
                                   "type_spatial_attention",
                                   "spatial_attention",
                                   input_dim=sc_enc_hidden_sizes,
                                   output_dim=sc_enc_hidden_sizes)

        self.init_attention_config("spatial_attention_config",
                                   "attribute_spatial_attention",
                                   "spatial_attention",
                                   input_dim=sc_enc_hidden_sizes + text_input_size,
                                   output_dim=sc_enc_hidden_sizes)

        self.init_attention_config("type_text_attention_config",
                                   "type_text_attention",
                                   "text_attention",
                                   q_input_dim=self.type_spatial_attention.output_dim + type_num,
                                   k_input_dim=text_input_size,
                                   v_input_dim=text_input_size,
                                   output_dim=text_input_size)

        self.init_attention_config("attribute_text_attention_config",
                                   "attribute_text_attention",
                                   "text_attention",
                                   q_input_dim=type_num,
                                   k_input_dim=text_input_size,
                                   v_input_dim=text_input_size,
                                   output_dim=text_input_size)

        self.attribute_decoder = GeneralDecoder(
            convert_kwargs(self.decoder_config.setdefault("attribute_decoder", {})),
            sc_enc_hidden_sizes + type_num + text_input_size
        )

        self.type_decoder = GeneralDecoder(
            convert_kwargs(self.decoder_config.setdefault("type_decoder", {})),
            sc_enc_hidden_sizes + type_num + text_input_size
        )

        self.avgPooling = torch.nn.AvgPool2d((GRID_SIZE, GRID_SIZE), stride=1)
        sample_config = convert_kwargs(self.decoder_config.setdefault("sample_config", {}))
        self.sample = sample_dict[self.decoder_config.setdefault("sample_type", "multinomial")](sample_config)

        self.obj_text_attention_weights = []
        self.att_text_attention_weights = []

    def init_attention_config(self, config_name, attr_name, attention_name, **kwargs):
        setattr(self, attr_name,
                attention_dict[attention_name](
                    convert_kwargs(self.decoder_config.setdefault(config_name, {})),
                    **kwargs)
                )

    def forward(self,
                scene_rep,
                text_embedding,
                text_enc_output,
                target,
                pad_mask,
                grid_target=None,
                train_no_gt = False):
        self.obj_text_attention_weights = []
        self.att_text_attention_weights = []
        text_rep = torch.cat([text_enc_output, text_embedding], dim=-1)
        if self.training and not train_no_gt:
        # if True:
            return self.run_train_obj_att(pad_mask, scene_rep, target, text_rep, grid_target)
        else:
            return self.run_test_obj_att(pad_mask, target, text_rep)

    def run_test_obj_att(self, pad_mask, target, text_rep):
        time_step = 1
        batch_size = len(pad_mask)
        gen_steps = self.max_len
        scene_reps = []
        result = {}
        last_n_hot_tensor = None
        for t in range(gen_steps):
            if t == 0:
                sample_n_hot_last = torch.zeros((batch_size, time_step, self.type_num)).to(self._device)
                last_n_hot_tensor = sample_n_hot_last.int()
                self.test_mask = torch.zeros((batch_size, 1, self.type_num), dtype=torch.uint8)
                scene_reps.append(self.get_scene_rep(b_s=batch_size))
            else:
                sample_n_hot_last = self.get_batch_one_hot(result['type_samples'][..., -1],
                                                           self.type_num).unsqueeze(1)
                if self.hot_type == 'nhot':
                    sample_n_hot_last = sample_n_hot_last.int()
                    last_n_hot_tensor |= sample_n_hot_last
                    sample_n_hot_last = last_n_hot_tensor.float()
            obj_dec_output = self.obj_attention_decoder(pad_mask, scene_reps[-1], sample_n_hot_last, text_rep)
            sample = self.test_obj_sample(obj_dec_output)
            sample_one_hot = self.get_batch_one_hot(sample, self.type_num)
            att_dec_output = self.att_attention_decoder(pad_mask, scene_reps[-1], sample_one_hot, text_rep)
            result_att = self.att_sample(att_dec_output)
            result_att['type_logits'] = obj_dec_output
            result_att['type_samples'] = sample
            scene_reps.append(self.get_scene_rep(result_att))
            result = self.update_result(result, result_att)
            self.obj_text_attention_weights = [torch.cat(self.obj_text_attention_weights, dim=1)]
            self.att_text_attention_weights = [torch.cat(self.att_text_attention_weights, dim=1)]
            if torch.all(sample == INIT_WORD_DICT[PAD]):
                break
        return result

    def run_train_obj_att(self, pad_mask, scene_rep, target, text_rep, grid_target):
        target_one_hot = self.get_batch_one_hot(values=target, class_num=self.type_num)
        if self.hot_type == 'nhot':
            target_n_hot = self.get_n_hot_tensor(target_one_hot)
        elif self.hot_type == 'onehot':
            target_n_hot = target_one_hot
        target_hot = self.get_minus_hot_tensor(target_n_hot)
        obj_dec_output = self.obj_attention_decoder(pad_mask, scene_rep, target_hot, text_rep)
        '''
        取obj_dec_output的sample作为type的预测值
        '''
        obj_softmax = functional.softmax(obj_dec_output, dim=-1)

        samples = self.sample(obj_softmax)
        att_dec_output = self.att_attention_decoder(pad_mask, scene_rep, target_one_hot, text_rep)
        result = self.att_sample(att_dec_output, grid_target)
        result['type_samples'] = samples
        result['type_logits'] = obj_dec_output

        return result

    def update_result(self, result, result_att):
        for att in result_att.keys():
            if result.get(att, None) is not None:
                result[att] = torch.cat([result[att], result_att[att]], dim=DIM_CAT[att])
            else:
                result[att] = result_att[att]
        return result

    def get_scene_rep(self, result=None, last_canvas_list=None, b_s=None):
        if result == None:
            self.last_canvas_list = []
            '''batch 中的每个都添加一个background'''
            step_img_tensor = self._dataset._get_tensor_from_ids(BACKGROUND_IDS).unsqueeze(0)
            step_img_tensor = step_img_tensor.unsqueeze(0).repeat([b_s] + [1 for _ in list(step_img_tensor.shape)])
        else:
            self.last_canvas_list = self.draw_gen_batch(result, last_canvas_list=self.last_canvas_list)
            canvas_list = self._dataset._getrgb(self.last_canvas_list)
            canvas_list = [self._dataset.resize_util(image) for image in canvas_list]
            step_img_tensor = [transforms.ToTensor()(canvas).unsqueeze(0).to(self._device) for canvas in
                               canvas_list]
            step_img_tensor = torch.cat(step_img_tensor, 0)
            step_img_tensor = step_img_tensor.unsqueeze(1)
        scene_rep = self._dataset._get_resnet_feature(step_img_tensor)
        scene_rep = self._scene_enc(scene_rep)
        return scene_rep

    def _extract_enc_feature(self, shape, sc_hidden_init):
        time_step = shape[1]
        spatial_size = list(shape[-2:])
        sc_hidden_init = sc_hidden_init.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # 增加最后一个维度和time step的维度
        sc_hidden_init = sc_hidden_init.repeat(1, time_step, 1, spatial_size[0], spatial_size[1])
        return sc_hidden_init

    def test_obj_sample(self, obj_dec_output):
        if TEST_DEL_DUP:
            obj_dec_output[self.test_mask] = NEG_INF
        logits = functional.softmax(obj_dec_output, dim=-1)
        sample = self.sample(logits)
        if TEST_DEL_DUP:
            self.test_mask[range(len(sample)), 0, sample] = 1
        return sample

    def att_sample(self, att_dec_output, target=None):
        '''
            att_dec_output的维度是[b_s, t_s, 1+attribute_one_hot, 28, 28]
            找到第一个channel里面值argmax最大的地方作为grid的结果
            找到最大grid对应位置的所有channel取出
            切分成相应的attribute的logits
            取argmax得到attribute
            '''
        '''get the grid channel'''
        grid_logits = att_dec_output[..., :1, :, :]
        '''get pos of max value in grid channel'''
        grid_logits_softmax = functional.softmax(grid_logits, dim=-1)
        grid_logits_flat = grid_logits_softmax.view(-1, GRID_SIZE * GRID_SIZE)
        if target is None:
            max_pos = torch.argmax(grid_logits_flat, dim=-1, keepdim=True)
        else:
            max_pos = target.view(-1, 1)
        max_pos_flat = max_pos.repeat(1, att_dec_output.shape[-3]).view(-1)
        '''get all channel'''
        att_one_hot = att_dec_output.view(-1, GRID_SIZE * GRID_SIZE)[range(len(max_pos_flat)), max_pos_flat]
        att_one_hot = att_one_hot.view(att_dec_output.shape[:-2])
        '''sample all channel'''
        index = 0
        result = {}
        for att in ATT_CHANNEL_DIC.keys():
            if att == 'location':
                index += ATT_CHANNEL_DIC[att]
                continue
            end_index = index + ATT_CHANNEL_DIC[att]
            result[att] = att_one_hot[..., index:end_index]
            index = end_index
        grid_logit_shape = list(grid_logits.shape[:-3])
        assert GRID_TYPE == 'gridx'
        if GRID_TYPE == 'gridx':
            result['grid'] = grid_logits.view(grid_logit_shape + [-1])
        # if GRID_TYPE == 'gridxy':
        #     result['grid_x'] = max_pos.view(list(att_dec_output.shape[:-3])) % GRID_SIZE
        #     result['grid_y'] = max_pos.view(list(att_dec_output.shape[:-3])) // GRID_SIZE
        return result

    def att_attention_decoder(self, pad_mask, scene_rep, target_one_hot, text_rep):
        '''attrbute text attention'''
        att_text_attention_output, att_wight = self.attribute_text_attention(
            q=target_one_hot,
            k=text_rep,
            v=text_rep,
            pad_mask=pad_mask,
            return_weights=True
        )
        self.att_text_attention_weights.append(att_wight.detach())
        '''attrbute spatial attention'''
        att_text_attention_output_rep = self.spatial_replicate(att_text_attention_output)
        target_one_hot_rep = self.spatial_replicate(target_one_hot)

        att_spa_attention_output = self.attribute_spatial_attention(
            torch.cat([scene_rep, att_text_attention_output_rep], dim=-3), scene_rep)
        '''attrbute decoder'''
        att_dec_output = self.attribute_decoder(
            torch.cat([att_spa_attention_output, target_one_hot_rep, att_text_attention_output_rep], dim=-3))
        return att_dec_output

    def spatial_replicate(self, att_text_attention_output):
        new_shape = list(att_text_attention_output.shape)
        new_shape.append(1)
        new_shape.append(1)
        rep_shape = [1 for _ in range(len(att_text_attention_output.shape))]
        rep_shape.append(GRID_SIZE)
        rep_shape.append(GRID_SIZE)
        att_text_attention_output_rep = \
            att_text_attention_output.view(new_shape).repeat(rep_shape)
        return att_text_attention_output_rep

    def obj_attention_decoder(self, pad_mask, scene_rep, target_hot, text_rep):

        '''object spatial attention'''
        obj_spa_att_output = self.type_spatial_attention(scene_rep, scene_rep)
        obj_spa_att_output = self.get_avg_pooling(obj_spa_att_output)
        '''object text attention'''
        obj_text_att_output, att_wight = self.type_text_attention(
            q=torch.cat([obj_spa_att_output, target_hot.float()], dim=-1),
            k=text_rep,
            v=text_rep,
            pad_mask=pad_mask,
            return_weights=True
        )
        self.obj_text_attention_weights.append(att_wight.detach())
        '''object decoder'''
        obj_dec_output = self.type_decoder(torch.cat([obj_spa_att_output, target_hot, obj_text_att_output], dim=-1))
        return obj_dec_output

    def get_minus_hot_tensor(self, target_n_hot):
        target_hot = target_n_hot[..., :-1, :]
        temp_shape = list(target_n_hot.shape)
        temp_shape[-2] = 1
        temp_one_hot = torch.zeros(temp_shape).to(self._device)
        target_hot = torch.cat([temp_one_hot, target_hot], dim=1)
        return target_hot

    def get_avg_pooling(self, input):
        shape = list(input.shape)
        b_s = shape[0]
        t_s = shape[1]
        input = input.view([-1] + shape[-3:])
        return self.avgPooling(input).view(b_s, t_s, -1)

    def get_batch_one_hot(self, values, class_num):  # 需要value最后一维度是平的
        values_shape = list(values.shape)
        values = values.view(-1, 1)
        batch_size = values.shape[0]
        ohtensor = torch.zeros(batch_size, class_num).cuda().scatter_(1, values, 1)
        return ohtensor.view(values_shape + [class_num])

    def get_n_hot_tensor(self, one_hot_tensor):
        one_hot_copy = one_hot_tensor.int()
        n_hot_tensor = torch.zeros(one_hot_copy.shape, dtype=torch.int).to(self._device)
        n_hot_tensor[..., 0, :] = one_hot_copy[..., 0, :]
        for i in range(1, one_hot_copy.shape[-2]):
            n_hot_tensor[..., i, :] = n_hot_tensor[..., i - 1, :] | one_hot_copy[..., i, :]
        return n_hot_tensor.float()

    def draw_gen_batch(self, outputs, batch=None, export_folder=None, name_prefix=None, n=None, last_canvas_list=None):
        if batch is not None:
            texts_batch = self._dataset._tensor2list(batch["text"])
            scene_names_batch = batch["scene"]
        '''draw generated batch'''
        flip_batch = outputs["flip"].argmax(-1).data
        pose_batch = outputs["pose"].argmax(-1).data
        expression_batch = outputs["expression"].argmax(-1).data
        # xs_batch, ys_batch, zs_batch = outputs["x"][..., 0].data, outputs["y"][..., 0].data, outputs["z"].argmax(
        #     -1).data
        zs_batch = outputs["z"].argmax(-1).data
        # xs_batch = (xs_batch * self.x_scale).to(torch.int32).data
        # ys_batch = (ys_batch * self.y_scale).to(torch.int32).data
        types_batch = self._dataset._tensor2list(outputs["type_samples"])
        if GRID_TYPE == 'gridx':
            grids_batch = outputs["grid"].data.argmax(-1)
        elif GRID_TYPE == 'gridxy':
            grids_x_batch = outputs["grid_x"].data.argmax(-1)
            grids_y_batch = outputs["grid_y"].data.argmax(-1)
            grids_batch = torch.cat([grids_x_batch.unsqueeze(-1), grids_y_batch.unsqueeze(-1)], dim=-1)

        batches = (
            zs_batch, grids_batch, flip_batch, types_batch, pose_batch, expression_batch)
        name_prefix = "sample" if name_prefix is None else name_prefix
        if batch == None:
            canvas_list = []
        for i, (zs, grids, fs, ts, ps, es) in enumerate(zip(*batches)):
            if batch is not None:
                '''每个batch画n个，test的时候n=-1全部画'''
                if 0 < n <= i + 2:
                    break
                texts, scene_name = texts_batch[i], scene_names_batch[i]
                local_export_folder = os.path.join(export_folder, "{}_{}".format(name_prefix, i))
                canvas_file = os.path.join(local_export_folder, "scene.png")
                data_file = os.path.join(local_export_folder, "gen_data.txt")
                os.makedirs(local_export_folder, exist_ok=True)
                with open(os.path.join(local_export_folder, "description.txt"), "w") as f:
                    f.write(" ".join(self._dataset.recover_texts(texts)))
            else:
                canvas_file = scene_name = data_file = None
                # canvas_file = os.path.join("/data1/bixiao/Code/Text2SceneFinal/test_pic/sample_{}_{}.png".format(i, 0))
            if last_canvas_list is not None and len(last_canvas_list) != 0:
                last_canvas = last_canvas_list[i]
                # canvas_file = os.path.join(
                    # "/data1/bixiao/Code/Text2SceneFinal/test_pic/sample_{}_{}.png".format(i, len(last_canvas_list)))
            else:
                last_canvas = None
            canvas = self._dataset.draw_from_grid(grids, zs, fs, ts, ps, es, save_file=canvas_file,
                                                  scene_file=scene_name,
                                                  data_file=data_file, last_canvas=last_canvas)
            if batch == None:
                canvas_list.append(canvas)
        if batch == None:
            return canvas_list


decoder_dict = {"rnn": RNNDecoder}

__all__ = ["decoder_dict"]
