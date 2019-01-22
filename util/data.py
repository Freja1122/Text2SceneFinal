import os
import sys
import re
import torch
import pickle
import random
import shutil
import collections
import numpy as np
from shutil import copyfile
from collections import defaultdict
from PIL import Image
from skimage import io
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import numpy_type_map
from torch._six import string_classes, int_classes
import torchvision.transforms as transforms
import cv2
import pickle
import bcolz
import threading
import re

'''import my own module'''
from util.toolkits import convert_kwargs, ProgressBar
from constants import *
from util.grid import grid_dict
from models.Resnet34 import resnet34
from util.toolkits import flat_bts, unflat_bts

# from util.n2imap import *
# from util.grid import grid_dict

'''设置root path'''
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

'''some constants for dataset'''
dotSet = set('.?!\",')
scales = {0: 1.0, 1: 0.7, 2: 0.49}
scales = {0: 1.0, 1: 0.7, 2: 0.49}
object_info_dict = {"source": 0, "type": 1, "pos": 2, "flip": 3, "pose": 4, "expression": 5, "step_scene": 6,
                    "step_scene_tensor": 7}
expression_num = 5
pose_num = 7
channel_num, height, width = 3, 400, 500
threadLock = threading.Lock()

'''设置随机数种子'''
torch.random.manual_seed(142857)
np.random.seed(22)

'''判断是否是图像文件'''


def is_img(file):
    try:
        Image.open(file)
        return True
    except IOError or OSError:
        return False


'''tokenize'''


def tokenize(s):
    s = re.sub(r'[;]', ',', s)
    s = re.sub(r'([.,?!"])', r' \1 ', s)
    s = re.sub(r'(\'s)', r' \1 ', s)
    s = re.sub(r'(n\'t)', r' \1 ', s)
    words = s.split()
    tokens = []
    for w in words:
        if tokens and tokens[-1] == w:
            continue
        tokens.append(w)
    if tokens and tokens[-1] not in dotSet:
        tokens.append('.')
    return tokens


'''删除以.开头的文件，输入为folder name'''


def remove_redundant_files(folder):
    for file in os.listdir(folder):
        if file.startswith("."):
            os.remove(os.path.join(folder, file))


'''valid上下文管理器'''


class valid_context_manager:
    def __init__(self, dataset):
        self._dataset = dataset

    def __enter__(self):
        self._dataset.validating = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._dataset.validating = False


'''test上下文管理器'''


class test_context_manager:
    def __init__(self, dataset):
        self._dataset = dataset

    def __enter__(self):
        self._dataset.testing = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._dataset.testing = False


'''Dataset'''


class Text2CVDataset(Dataset):
    def __init__(self, name, config):
        self._file_path = os.path.dirname(os.path.realpath(__file__))
        self._n_data = None
        self._initialized = False
        self._init_config(name, config)
        self.scene_shape = self.pos_info = self.type_info = None
        self.type_source_dict = self.type_idx_dict = self.idx_type_dict = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._clear_cache = self.config["clear_cache"]
        self.valid = valid_context_manager(self)
        self.test = test_context_manager(self)
        '''
        real_ins_text_list 代表object中和text中交集的instance的list
        gt_real_rate 代表 gt 中交集占比
        '''
        self.real_ins_text_list, self.gt_real_rate = [], []
        self.first_valid = None
        self._step_scene_num = self.config.setdefault("step_scene_num", 66663)
        self.step_scene_shape = (224, 224)
        self.resize_util = transforms.Resize(self.step_scene_shape)

    @property
    def n_cv(self):
        return int(self._n_data * self._cv_ratio)

    @property
    def n_data(self):
        if self._n_data is None or self._n_data <= 0:  # changed by bi
            self._n_data = len(self._base_names) if self.training else len(self._texts)
            # self._n_data = len(self._base_names)
            self._random_indices = np.random.permutation(self._n_data)
        self._n_data = int(self._n_data)
        return self._n_data

    @property
    def n_vocab(self):
        return len(self._word2idx)

    @property
    def n_types(self):
        return len(self.type_idx_dict)

    def _init_config(self, name, config):
        self.config = convert_kwargs(config)
        self.config.setdefault("clear_cache", False)
        self.config.setdefault("load_sources", False)
        self.config.setdefault("preprocess_method", "scale")
        self.config.setdefault("max_words", 0)  # 这个不知道是什么
        self.config.setdefault("max_len", 100)  # 这个是text的长度
        self.training = self.config.setdefault("training", True)  # changed by bi
        self.validating = self.config.setdefault("validating", False)
        self.testing = self.config.setdefault("testing", False)
        self._cv_ratio = float(self.config.setdefault("cv_ratio", 0.1))
        self.n_test = self.config.setdefault("test_num", 100)
        self.sim_type = self.config.setdefault("sim_type", True)
        self._grid_x = grid_dict[GRID_TYPE](positive_xy=True)
        self.check_real_rate = self.config.setdefault("check_real_rate", True)

        # src
        self.parent_path = os.path.abspath(os.path.join(self._file_path, os.pardir))
        src_data_folder = self.config.setdefault("src_data_folder", os.path.join(self.parent_path, "data",
                                                                                 "AbstractScenes_v1.1"))
        tgt_data_folder = self.config.setdefault("tgt_data_folder", os.path.join(self.parent_path, "data", name))
        self.config.setdefault("src_scene_folder", os.path.join(src_data_folder, "RenderedScenes"))
        self.config.setdefault("src_texts_folder",
                               os.path.abspath(os.path.join(src_data_folder, "SimpleSentences_xinyi")))
        self.config.setdefault("combine_texts_list", ["SimpleSentences1_10020.txt", "SimpleSentences2_10020.txt"])
        self.config.setdefault("src_source_folder", os.path.join(src_data_folder, "Pngs"))
        self.config.setdefault("src_object_file", os.path.join(src_data_folder, "Scenes_10020.txt"))

        # tgt
        source_folder = self.config.setdefault("source_folder", "sources")
        scene_folder = self.config.setdefault("scene_folder", "scenes")
        text_folder = self.config.setdefault("text_folder", "texts")
        object_folder = self.config.setdefault("object_folder", "objects")
        step_scene_folder = self.config.setdefault("step_scene_folder", "step_scenes")
        step_scene_tensor_folder = self.config.setdefault("step_scene_tensor_folder", "step_scenes_tensor")
        resnet_folder = self.config.setdefault("resnet_folder", "resnet")

        # step scene
        self._step_scene_name = self.config.setdefault("step_scene_name", 'step_scene_{}_{}.png')
        '''resnet, rgb, None'''
        self._scene_load_type = self.config.setdefault("scene_load_type", 'resnet')
        if self._scene_load_type == 'resnet':
            self.resnet_model = resnet34(pretrained=True, num_classes=1000, zero_init_residual=False).cuda()
            for param in self.resnet_model.parameters():
                param.requires_grad = False
        self._spend_time_gen = self.config.setdefault("spend_time_gen", True)
        self._is_generate_scene = self.config.setdefault("generate_step_scene", True)

        # define folder path
        self._source_folder = os.path.join(tgt_data_folder, source_folder)
        self._scene_folder = os.path.join(tgt_data_folder, scene_folder)
        self._text_folder = os.path.join(tgt_data_folder, text_folder)
        self._object_folder = os.path.join(tgt_data_folder, object_folder)
        self._step_scene_folder = os.path.join(tgt_data_folder, step_scene_folder)
        self._step_scene_tensor_folder = os.path.join(tgt_data_folder, step_scene_tensor_folder)
        self._resnet_folder = os.path.join(tgt_data_folder, resnet_folder)

    def __len__(self):
        if self._cv_ratio <= 0:
            return self.n_data
        n_train = int(self.n_data - self.n_cv - self.n_test)
        return n_train

    def __getitem__(self, index):
        if self.validating:
            index = self._n_data - index % self.n_cv - self.n_test
        elif self.testing:
            index = self._n_data - index % self.n_test - 1
        index = self._random_indices[index]
        if not self._initialized:
            raise ValueError("please call 'make_dataset' method before using Text2CVDataset")
        '''text'''
        n_texts = self._texts_lengths[index]
        chosen_texts = self._texts[index][random.randint(0, n_texts - 1)]
        if self.testing:
            chosen_texts = self._texts[index][0]

        '''scene and objects'''
        scene_name = self._scene_names[index]
        # scene = torch.load(os.path.join(self._scene_folder, "pt", "{}.pt".format(os.path.splitext(scene_name)[0])))
        objects = self._objects[index].copy()
        if self.config["load_sources"]:
            objects = objects.copy()
            source_idx = object_info_dict["source"]
            sources = objects[source_idx]
            for i, source in enumerate(sources):
                objects[source_idx][i] = torch.load(os.path.join(self._source_folder, "pt", source))
        if self._scene_load_type == 'rgb':
            objects_num = len(objects[object_info_dict['source']])
            '''step scene是对应第i个object生成之前的图片，以background开始'''
            scene_tensors = transforms.ToTensor()(self._get_image_from_ids(BACKGROUND_IDS)).unsqueeze(0)
            for i in range(0, objects_num):
                interm_image = self._get_image_from_ids([index, i])
                scene_tensor = transforms.ToTensor()(interm_image).unsqueeze(0)
                scene_tensors = torch.cat([scene_tensors, scene_tensor], 0)
            objects.append(scene_tensors)
        elif self._scene_load_type == 'resnet':
            '''load resnet output tensor'''
            resnet_feature = torch.load(os.path.join(self._resnet_folder, 'resnet_feature_{}.pt'.format(index)))
            objects.append(resnet_feature)
            self.resnet_channel = resnet_feature.shape[-3]
            del resnet_feature

        return {"scene": scene_name, "text": chosen_texts, "objects": objects}

    def _get_image_from_ids(self, ids):
        file_name = self._get_file_name_from_step_ids(ids)
        image = Image.open(os.path.join(self._step_scene_folder, file_name))
        image = self._getrgb(image)
        return image

    def _getrgb(self, image):
        if isinstance(image, list):
            return [m.convert('RGB') for m in image]
        return image.convert('RGB')

    def _get_tensor_from_ids(self, ids):
        image = self._get_image_from_ids(ids)
        image = self.resize_util(image)
        return transforms.ToTensor()(image)

    def _get_file_name_from_step_ids(self, ids):
        return self._step_scene_name.format(ids[0], ids[1])

    def _pop_idx(self, idx):
        self._base_names.pop(idx)
        self._scene_names.pop(idx)
        self._texts = torch.cat([self._texts[:idx], self._texts[idx + 1:]], 0)
        self._objects.pop(idx)
        self._texts_lengths.pop(idx)

    def _get_img(self, file):
        return torch.from_numpy(io.imread(file)).to(self.device)

    def _get_dict(self, words):
        self._word2idx = INIT_WORD_DICT.copy()
        self._word2idx.update({word: i + len(self._word2idx) for i, word in enumerate(words)})

    def make_dataset(self, init_only=False):
        self._initialized = True
        if not init_only:
            self._combine_texts()
            self._copy_data()
        self._scene_names = sorted([
            file for file in os.listdir(self._scene_folder)
            if os.path.isfile(os.path.join(self._scene_folder, file)) and
               is_img(os.path.join(self._scene_folder, file))
        ])
        self._base_names = [os.path.splitext(name)[0] for name in self._scene_names]
        self._sources_names = [
            file for file in os.listdir(self._source_folder)
            if os.path.isfile(os.path.join(self._source_folder, file))
        ]
        self._texts, self._objects = [], []
        print("extracting scenes")
        self._extract_scenes()
        print("extracting texts")
        self._extract_texts()
        print("extracting sources")
        self._extract_sources()
        print("extracting objects")
        self._extract_objects()
        print("extracting step scene")
        self._extract_step_pic()
        print("extracting resnet")
        self._extract_resnet_feature()
        print('extract over')
        return self

    def _extract_scenes(self):
        scene_shape_file = os.path.join(self._scene_folder, "scene_shape.txt")
        print('write scene shape file to', scene_shape_file)
        print('save image using torch to', self._scene_folder)
        if not self.training or os.path.exists(scene_shape_file):
            with open(scene_shape_file, "r") as f:
                self.scene_shape = [int(line.strip()) for line in f]
            return
        scene_pt_folder = os.path.join(self._scene_folder, "pt")
        os.makedirs(scene_pt_folder, exist_ok=True)
        bar = ProgressBar(max_value=len(self._scene_names), name="scene")
        for name in self._scene_names:
            base_name = os.path.splitext(name)[0]
            img_file = os.path.join(self._scene_folder, name)
            pt_file = os.path.join(scene_pt_folder, "{}.pt".format(base_name))
            if self.scene_shape is None:
                self.scene_shape = self._get_img(img_file).shape  # 第一次 [400,500,4]
                with open(scene_shape_file, "w") as f:
                    f.write("\n".join(map(lambda n: str(n), self.scene_shape)))
            if self._clear_cache or not os.path.isfile(pt_file):
                torch.save(self._get_img(img_file), pt_file)
            bar.update()
        print()

    def _extract_texts(self):
        max_len = self.config["max_len"]
        cache_folder = os.path.join(self._text_folder, "_cache_{}".format(max_len))
        os.makedirs(cache_folder, exist_ok=True)
        dict_file = os.path.join(cache_folder, "dict.txt")
        data_file = os.path.join(cache_folder, "texts.pt")
        lengths_file = os.path.join(cache_folder, "lengths.txt")
        if not self._clear_cache and all(os.path.isfile(file) for file in (dict_file, data_file, lengths_file)):
            print("loading dict")
            with open(dict_file, "r") as f:
                words = [line.strip() for line in f]
                self._get_dict(words)
            print("loading lengths")
            with open(lengths_file, "r") as f:
                self._texts_lengths = [int(line.strip()) for line in f]
            print("loading texts tensor")
            self._texts = torch.load(data_file)
        else:
            pad, unk = INIT_WORD_DICT[PAD], INIT_WORD_DICT[UNK]
            max_words = self.config["max_words"]
            bar, counter = ProgressBar(max_value=self.n_data, name="text"), Counter()
            print('dynamic padding and counting words... ')
            for i, name in enumerate(self._base_names):  # n_data为什么是9993
                text_dir = os.path.join(self._text_folder, "{}.txt".format(name))
                with open(text_dir, "r") as f:
                    lines = []
                    for line in f:
                        line = line.strip().split()
                        line_length = len(line)
                        assert line_length <= max_len
                        line = [PAD] * (max_len - line_length) + line
                        lines.append(line)
                        del line
                    if not lines:
                        self._texts.append([[PAD] * max_len])
                    else:
                        self._texts.append(lines)
                        counter.update(sum(lines, []))
                bar.update()
            for init in INIT_WORD_DICT:
                counter.pop(init, None)
            if max_words <= 0:
                words = list(counter.keys())
            else:
                words = [pair[0] for pair in counter.most_common(max_words)]
            self._get_dict(words)
            with open(dict_file, "w") as f:
                f.write("\n".join(words))
            print()
            print("transforming texts from words to index")
            bar = ProgressBar(max_value=self.n_data, name="word2idx")
            for i, sentences in enumerate(self._texts):
                for j, sentence in enumerate(sentences):
                    self._texts[i][j] = [self._word2idx.get(w, unk) for w in sentence]
                bar.update()
            '''_texts_lengths means how many sentences in one text list'''
            self._texts_lengths = [len(lines) for lines in self._texts]
            with open(lengths_file, "w") as f:
                f.write("\n".join(map(lambda n: str(n), self._texts_lengths)))
            n_max_lines = max(self._texts_lengths)
            print('add the lost text (every scene has two sentence with length of max_len) and build tensor')
            for i, lines in enumerate(self._texts):
                lines_length = len(lines)
                if lines_length < n_max_lines:
                    lines += [[pad] * max_len] * (n_max_lines - lines_length)
            self._texts = torch.LongTensor(self._texts)
            print('save tensor to', data_file)
            torch.save(self._texts, data_file)
        print("loading tensor to device")
        print('vocab_size: ', self.n_vocab)
        self._texts = self._texts.to(self.device)
        self._idx2word = {i: w for w, i in self._word2idx.items()}
        self._instance2noun = torch.load(os.path.join(self.parent_path, 'sources/AS/instance2noun.pkl'))
        self._instance2idx = {ins: [self._word2idx.get(w) for w in ws] for ins, ws in self._instance2noun.items()}
        print()

    def _extract_sources(self):
        source_pt_folder = os.path.join(self._source_folder, "pt")
        os.makedirs(source_pt_folder, exist_ok=True)
        bar = ProgressBar(max_value=len(self._sources_names), name="source")
        for name in self._sources_names:
            base_name = os.path.splitext(name)[0]
            img_file = os.path.join(self._source_folder, name)
            pt_file = os.path.join(source_pt_folder, "{}.pt".format(base_name))
            if self._clear_cache or not os.path.isfile(pt_file):
                torch.save(self._get_img(img_file), pt_file)
            bar.update()

    def _extract_resnet_feature(self):
        if self._scene_load_type != 'resnet':
            return
        os.makedirs(self._resnet_folder, exist_ok=True)
        remove_redundant_files(self._resnet_folder)
        resnet_feature_names = os.listdir(self._resnet_folder)
        name_max_idx = len(resnet_feature_names)
        if name_max_idx < self.n_data:
            name_max_idx = max(0, name_max_idx - 1)
            print('get feature from step pic to: ', self._resnet_folder)
            print('generating from idx: {}'.format(name_max_idx))
            bar = ProgressBar(max_value=self.n_data - name_max_idx, name="resnet feature")
            for i in range(name_max_idx, self.n_data):
                objects = self._objects[i]
                objects_num = len(objects[object_info_dict['source']])
                scene_tensors = []
                '''add backgrounud'''
                img_tensor = self._get_tensor_from_ids(BACKGROUND_IDS).unsqueeze(0)
                scene_tensors.append(img_tensor)
                del img_tensor
                for idx in range(0, objects_num):
                    img_tensor = self._get_tensor_from_ids([i, idx]).unsqueeze(0)
                    scene_tensors.append(img_tensor)
                scene_tensors = torch.cat(scene_tensors, dim=0)
                resnet_feature = self._get_resnet_feature(scene_tensors)
                torch.save(resnet_feature,
                           os.path.join(self._resnet_folder, 'resnet_feature_{}.pt'.format(i)))
                del scene_tensors
                bar.update()
            return

    def _get_resnet_feature(self, scene_tensors):
        result_step = self.resnet_model(scene_tensors.to(self.device), layer_num=4)
        batch_size_x, result_step = flat_bts(result_step)
        result_step = torch.nn.functional.interpolate(result_step, scale_factor=(2, 2), mode='bilinear',
                                                      align_corners=None)  # align_corners 应该是什么
        result_step = unflat_bts(batch_size_x, result_step)
        return result_step.data

    def _update_info(self, attr, info):
        attr_complete = "{}_info".format(attr)
        self_attr = getattr(self, attr_complete, None)
        transposed_info = list(zip(*info))
        if self_attr is None:
            self_attr = [set() for _ in range(len(transposed_info))]
        for j, sub_info in enumerate(transposed_info):
            self_attr[j].update(set(sub_info))
        setattr(self, attr_complete, self_attr)

    def _get_type_idx_in_order(self):
        num_list = [8, 10, 1, 1, 6, 10, 7, 15]
        idx_in_num_list = [2, 3]
        idx_dic = INIT_WORD_DICT.copy()
        idx2tuple = {v: k for k, v in INIT_WORD_DICT.items()}
        if self.sim_type:
            idx_dic[str([idx_in_num_list[0], 0])] = len(idx_dic)
            idx_dic[str([idx_in_num_list[1], 0])] = len(idx_dic)
        else:
            for i in range(0, 35):
                idx_dic[str([2, i])] = len(idx_dic)
            for i in range(0, 35):
                idx_dic[str([3, i])] = len(idx_dic)

        for i in range(0, len(num_list)):
            if i in idx_in_num_list:  # origin index rather index after map
                continue
            for j in range(0, num_list[i]):
                idx_dic[str([i, j])] = len(idx_dic)
        idx_dic = {str(k): v for k, v in idx_dic.items()}
        idx2tuple = {v: k for k, v in idx_dic.items()}
        return idx_dic, idx2tuple

    def _extract_objects(self):
        target_info = ["pos", "type"]
        type_dicts_file = os.path.join(self._object_folder, "type.dicts")
        info_files = [os.path.join(self._object_folder, "_{}.info".format(info)) for info in target_info]
        if not self.training:
            with open(type_dicts_file, "rb") as f:
                self.type_source_dict, self.type_idx_dict, self.idx_type_dict = pickle.load(f)
            for info, info_file in zip(target_info, info_files):
                if not self._clear_cache and os.path.isfile(info_file):
                    with open(info_file, "rb") as f:
                        setattr(self, "{}_info".format(info), pickle.load(f))
            # return
        bar = ProgressBar(max_value=self.n_data, name="object")
        extract_info = {}
        if self._clear_cache or not os.path.isfile(type_dicts_file):
            update_dicts = True
            self.type_source_dict = {EOS: None}
            self.type_idx_dict, self.idx_type_dict = self._get_type_idx_in_order()
        else:
            update_dicts = False
            with open(type_dicts_file, "rb") as f:
                self.type_source_dict, self.type_idx_dict, self.idx_type_dict = pickle.load(f)
        print('build data in the format: ', object_info_dict)
        source_idx, type_idx = object_info_dict["source"], object_info_dict["type"]
        for info, info_file in zip(target_info, info_files):
            extract_info[info] = True
            if not self._clear_cache and os.path.isfile(info_file):
                extract_info[info] = False
                with open(info_file, "rb") as f:
                    setattr(self, "{}_info".format(info), pickle.load(f))
        for idx, name in enumerate(self._base_names):
            with open(os.path.join(self._object_folder, "{}.txt".format(name)), "r") as f:
                local_objects = [[] for _ in range(len(object_info_dict) - 2)]  # step scene不在这里加
                for line in f:
                    objects_info = line.strip().split("\t")
                    for i, info in enumerate(objects_info):
                        data = info.split()
                        if i == source_idx:
                            local_objects[i].append("{}.pt".format(os.path.splitext(data[0])[0]))
                        else:
                            lo = list(map(lambda n: int(n), data))
                            local_objects[i].append(lo)
                            del lo
                types, sources = local_objects[type_idx], local_objects[source_idx]
                if update_dicts:
                    for t, s in zip(types, sources):
                        t = str(t)
                        self.type_source_dict[t] = s
                for i, info in enumerate(local_objects):
                    if i == source_idx:
                        local_objects[i] = info
                    else:
                        for tgt_info in target_info:
                            if i == object_info_dict[tgt_info] and extract_info[tgt_info]:
                                self._update_info(tgt_info, info)
                        if i == type_idx:
                            if self.sim_type:
                                for j in range(0, len(info)):
                                    if info[j][0] == 2:
                                        info[j] = [2, 0]
                                    if info[j][0] == 3:
                                        info[j] = [3, 0]
                            info = [self.type_idx_dict[str(line)] for line in info] + [self.type_idx_dict[EOS]]
                            if self.check_real_rate:
                                real_ins_text1 = self._get_real_type_list(info, self._tensor2list(self._texts[idx][0]),
                                                                          True)
                                real_ins_text2 = self._get_real_type_list(info, self._tensor2list(self._texts[idx][1]),
                                                                          True)
                                self.real_ins_text_list.append([real_ins_text1, real_ins_text2])
                                self.gt_real_rate.append([len(real_ins_text1[3]) / len(info),
                                                          len(real_ins_text2[3]) / len(info)])
                        else:
                            if len(info) == 0:
                                continue
                            info.append([0] * len(info[0]))
                        local_objects[i] = torch.LongTensor(info).to(self.device)
            self._objects.append(local_objects)
            del local_objects
            bar.update()
        for info, info_file in zip(target_info, info_files):
            print('write ', info, ' into ', info_file)
            if extract_info[info]:
                with open(info_file, "wb") as f:
                    pickle.dump(getattr(self, "{}_info".format(info)), f)  # ???把什么pickle进去了
        if update_dicts:
            print('write idx_type_dict into ', type_dicts_file)
            self.idx_type_dict = {i: t for t, i in self.type_idx_dict.items()}
            with open(type_dicts_file, "wb") as f:
                pickle.dump((self.type_source_dict, self.type_idx_dict, self.idx_type_dict), f)
        if self.check_real_rate:
            real_rate = sum(self.gt_real_rate, [])
            print('mean num of gt type real num: ', np.mean(real_rate))
        print()

    def _tensor2list(self, t):
        if isinstance(t, list):
            return t
        return t.data.cpu().numpy().tolist()

    def _tensor2numpy(self, t):
        return t.data.cpu().numpy()

    def _extract_step_pic(self):
        out_class = self

        class myThread(threading.Thread):
            def __init__(self, index):
                threading.Thread.__init__(self)
                self.index = index
                self.outer = out_class

            def run(self):
                threadLock.acquire()
                self.outer._draw_step_scene(self.index)
                threadLock.release()

        if self._is_generate_scene == False:
            return
        print('write step scene to', self._step_scene_folder)
        os.makedirs(self._step_scene_folder, exist_ok=True)
        remove_redundant_files(self._step_scene_folder)
        step_scene_names = os.listdir(self._step_scene_folder)
        if 'step_scene_-1_-1.png' not in step_scene_names:
            shutil.copy(os.path.join(self.config["src_source_folder"], 'background.png'),
                        os.path.join(self._step_scene_folder, 'step_scene_-1_-1.png'))
            step_scene_names = os.listdir(self._step_scene_folder)
        if self._scene_load_type == None:
            return
        if self._spend_time_gen == False:
            print('make sure u spend time to gen')
        if len(step_scene_names) != 0:
            step_scene_idxs = [int(re.findall(r"\d+", s)[0]) for s in step_scene_names]
            max_step_idxs = max(step_scene_idxs)
        else:
            max_step_idxs = 0
        '''把name全部加入'''
        for i, objects in enumerate(self._objects):
            objects_num = len(objects[object_info_dict['source']])
            names = []
            for idx in range(0, objects_num):
                names.append([i, idx])
            names.append(BACKGROUND_IDS)
            names_tensor = torch.tensor(names)
            self._objects[i].append(names_tensor)
            del names_tensor
        print('step scene names added')
        print()
        if max_step_idxs == self.n_data - 1:
            return
        print('begin to generate step scene')
        thread_num = 6
        '''生成thread num整数倍object的中间图片'''
        bar = ProgressBar(max_value=((self.n_data - max_step_idxs) // thread_num), name="step scene")
        lsat_num = max_step_idxs
        for i in range(max_step_idxs, len(self._objects), thread_num):
            threads = []
            try:
                for t in range(0, thread_num):
                    thread = myThread(i + t)
                    thread.start()
                    threads.append(thread)
            except:
                print("Error: unable to start thread")
            if len(threads) != 0:
                for t in threads:
                    t.join()
            bar.update()
            lsat_num = i
        '''生成剩下的中间图片'''
        bar = ProgressBar(max_value=self.n_data - lsat_num, name="step scene")
        for i in range(lsat_num, self.n_data):
            self._draw_step_scene(i)
            bar.update()
        print('step scene number: {}'.format(self._step_scene_num))

    def _draw_step_scene(self, i):
        objects = self._objects[i]
        objects_num = len(objects[object_info_dict['source']])
        pos_array = objects[object_info_dict['pos']].data.cpu().numpy()
        f_array = objects[object_info_dict['flip']].data.cpu().numpy()
        t_array = objects[object_info_dict['type']].data.cpu().numpy()
        pose_array = objects[object_info_dict['pose']].data.cpu().numpy()
        expression_array = objects[object_info_dict['expression']].data.cpu().numpy()
        xs, ys, zs, fs, ts, grids, ps, es = [], [], [], [], [], [], [], []
        for idx in range(0, objects_num):
            xs.append(pos_array[idx][0])
            ys.append(pos_array[idx][1])
            zs.append(pos_array[idx][2])
            grids.append(pos_array[idx][3])
            fs.append(f_array[idx])
            ps.append(pose_array[idx])
            es.append(expression_array[idx])
            ts.append(t_array[idx])
            save_file = os.path.join(self._step_scene_folder, self._step_scene_name.format(i, idx))
            self.draw(xs, ys, grids, zs, fs, ts, ps, es, save_file=save_file)
        del xs, ys, zs, fs, ts, grids, pos_array, f_array, t_array

    def _combine_texts(self):
        src_texts_folder = self.config["src_texts_folder"]

        def _sub_combine(src_file):
            end_tokens = set(".?!\"")
            idx, texts, local_texts = None, {}, []
            src_file = os.path.join(src_texts_folder, src_file)
            with open(src_file, "r") as rf:
                for line in rf:
                    line = line.strip().split("\t")
                    if len(line) != 3:
                        continue
                    if idx is None:
                        idx = line[0]
                    if idx != line[0]:
                        texts[int(idx)] = " ".join(local_texts)
                        local_texts = []
                        idx = line[0]
                    if line[-1][-1] not in end_tokens:
                        line[-1] += "."
                    local_texts.append(line[-1])
                texts[int(idx)] = " ".join(local_texts)
            return texts

        file_list = self.config["combine_texts_list"]
        keys, texts_list = set(), [_sub_combine(file) for file in file_list]
        for t in texts_list:
            keys |= t.keys()
        with open(os.path.join(src_texts_folder, "combine.txt"), "w") as f:
            f.write("\n".join(["\t".join([
                t[key] for t in texts_list if key in t
            ]) for key in sorted(keys)]))

    def _preprocess_texts(self, src_texts_folder, tgt_texts_folder):
        max_len = self.config["max_len"]
        src_texts_file = os.path.join(src_texts_folder, "combine.txt")
        tgt_texts_file = os.path.join(tgt_texts_folder, "_processed_{}.txt".format(max_len))
        if not self._clear_cache and os.path.isfile(tgt_texts_file):
            print("loading preprocessed texts from ", tgt_texts_file)
            print()
            with open(tgt_texts_file, "r") as f:
                pop_indices = list(map(lambda n: int(n), f.readline().strip().split()))
                data = [[sub_line.split() for sub_line in line.strip().split("\t")] for line in f]
        else:
            print("preprocessing texts, write from ", src_texts_file, " to ", tgt_texts_file)
            data, pop_indices = [], []
            with open(src_texts_file, "r") as f:
                for i, line in enumerate(f):
                    pop = False
                    local_lines = []
                    for sub_line in line.strip().split("\t"):
                        tokenized = tokenize(sub_line)
                        local_lines.append(tokenized)
                        if len(tokenized) > max_len:
                            pop = True
                            break
                    if pop:
                        pop_indices.append(i)
                    data.append(local_lines)
            with open(tgt_texts_file, "w") as f:
                f.write(" ".join(map(lambda n: str(n), sorted(pop_indices))) + "\n")
                f.write("\n".join(["\t".join([" ".join(sub_line) for sub_line in line]) for line in data]))
            print('tokenize over, combine 6 sentences into two list')
            print()
        return data, set(pop_indices)

    def _copy_data(self):
        pop_indices = set()
        tgt_data_folder = self.config["tgt_data_folder"]
        os.makedirs(tgt_data_folder, exist_ok=True)

        scene_names = scenes_base_names = None
        if self.training:
            src_scene_folder = self.config["src_scene_folder"]
            print("copying scenes from ", src_scene_folder, ' to ', tgt_data_folder)
            os.makedirs(self._scene_folder, exist_ok=True)
            remove_redundant_files(src_scene_folder)
            src_scene_names = os.listdir(src_scene_folder)
            for i, file in enumerate(src_scene_names):
                new_file = file[5:].split("_")
                new_file = "{:04d}_{}".format(int(new_file[0]), new_file[1])
                # noinspection PyTypeChecker
                src_scene_names[i] = (new_file, file)
            scene_names, scenes_base_names = [], []
            for new_file, file in sorted(src_scene_names):
                scene_names.append(os.path.join(self._scene_folder, new_file))
                shutil.copy(os.path.join(src_scene_folder, file), scene_names[-1])
                scenes_base_names.append(os.path.splitext(new_file)[0])
            print('scene name changed: e.g (new_file, file) ', src_scene_names[0])
            print()

        src_texts_folder = self.config["src_texts_folder"]
        print("copying texts from ", src_texts_folder, ' to ', self._text_folder)
        os.makedirs(self._text_folder, exist_ok=True)
        remove_redundant_files(src_texts_folder)
        '''get the tokenized text'''
        texts, texts_pop_indices = self._preprocess_texts(src_texts_folder, self._text_folder)
        text_names = []
        pop_indices |= texts_pop_indices
        if scenes_base_names is None:
            scenes_base_names = ["t{}".format(i) for i in range(len(texts))]
        print()
        ''''every file has two line, related to one image'''
        for i, (base_name, texts_lines) in enumerate(zip(scenes_base_names, texts)):
            text_names.append(os.path.join(self._text_folder, "{}.txt".format(base_name)))
            with open(text_names[-1], "w") as f:
                f.write("\n".join([" ".join(line) for line in texts_lines]))

        src_source_folder = self.config["src_source_folder"]
        print("copying sources from ", src_source_folder, ' to ', self._source_folder)
        print()
        os.makedirs(self._source_folder, exist_ok=True)
        remove_redundant_files(src_source_folder)
        for file in os.listdir(src_source_folder):
            shutil.copy(os.path.join(src_source_folder, file), os.path.join(self._source_folder, file))

        object_names = None
        if self.training:
            src_object_file = self.config["src_object_file"]
            print("copying objects from ", src_object_file, ' to ', self._object_folder)
            print()
            if self.sim_type == False:
                self._object_folder += '_old'
            os.makedirs(self._object_folder, exist_ok=True)
            line_cursor = -1
            cursor, counts, objects = 0, -1, [[] for _ in range(len(scenes_base_names))]
            with open(src_object_file, "r") as f:
                f.readline()
                for line in f:
                    line = line.strip().split("\t")
                    if counts == -1:
                        cursor = 0
                        line_cursor += 1
                        counts = int(line[1])
                        if counts == 0:
                            counts = -1
                            pop_indices.add(line_cursor)
                        continue
                    cursor += 1
                    source, dtype, pos, flip = line[0], line[1:3], line[3:6], line[6]
                    if int(dtype[0]) in HUMAN_IDX_LIST:
                        pose = int(dtype[1]) // expression_num
                        expression = int(dtype[1]) % expression_num
                    else:
                        pose, expression = 0, 0
                    if self._grid_x.name == 'gridx':
                        pos.append(str(self._grid_x.get_grid_id_from_xy(int(pos[0]), int(pos[1]))))
                    elif self._grid_x.name == 'gridxy':
                        x_grid, y_grid = self._grid_x.get_x_y_grid_from_pos(int(pos[0]), int(pos[1]))
                        pos.append(str(x_grid))
                        pos.append(str(y_grid))
                    # noinspection PyTypeChecker
                    objects[line_cursor].append(
                        "\t".join([source, " ".join(dtype), " ".join(pos), flip, str(pose), str(expression)]))
                    if cursor == counts:
                        counts = -1
            object_names = []
            for base_name, obj_list in zip(scenes_base_names, objects):
                object_names.append(os.path.join(self._object_folder, "{}.txt".format(base_name)))
                with open(object_names[-1], "w") as f:
                    f.write("\n".join(obj_list))

        print("deleting {} invalid data".format(len(pop_indices)))
        print('pop_indices: ', pop_indices)
        for idx in sorted(pop_indices):
            for names in (scene_names, text_names, object_names):
                if names is None:
                    continue
                # print(names[idx])
                os.remove(names[idx])
        print('copy done')
        print()

    def draw_from_grid(self, grids, zs, fs, ts, ps, es, scene_file=None, save_file=None, data_file=None,
                       last_canvas=None):
        xs = []
        ys = []
        for grid in grids:
            if self._grid_x.name == 'gridx':
                x, y = self._grid_x.get_center_pos_from_id(grid)
            elif self._grid_x.name == 'gridxy':
                x, y = self._grid_x.get_center_pos_from_x_y_grid_num(grid[0], grid[1])
            xs.append(x)
            ys.append(y)
        return self.draw(xs, ys, grids, zs, fs, ts, ps, es, scene_file=scene_file, save_file=save_file,
                         data_file=data_file, last_canvas=last_canvas)

    def draw(self, xs, ys, grids, zs, fs, ts, ps, es, scene_file=None, save_file=None, data_file=None,
             last_canvas=None):
        def render_img(x, y, z, f, source_type, canvas):
            source = Image.fromarray(
                torch.load(os.path.join(self._source_folder, "pt", source_type)).data.cpu().numpy())
            w, h = source.size
            scale = scales[z.item()]
            if scale != 1:
                w, h = int(w * scale), int(h * scale)
                source.thumbnail((w, h))
            flip = f.item()
            if flip == 1:
                source = source.transpose(Image.FLIP_LEFT_RIGHT)
            if not isinstance(x, int):
                x, y = x.item(), y.item()
            x, y = x - w // 2, y - h // 2
            if x < 0:
                x, source = 0, source.crop((-x, 0, w, h))
                w += x
            if y < 0:
                y, source = 0, source.crop((0, -y, w, h))
            canvas.paste(source, (x, y), source)
            return canvas

        canvas = Image.open(os.path.join(self._source_folder, "background.png"))
        # 根据t对应的pair的第一位排序
        type_list = []
        pic_list = []
        t_no_dup = []
        for i, t in enumerate(ts):
            source_type = self.type_source_dict.get(self.idx_type_dict[t], "unknown")
            # 如果是人物，则绘制相应图像
            if t in HUMAN_PIC_IDX:
                new_index = (ps[i] * expression_num + es[i]).item()
                new_type = self.idx_type_dict[t].replace('0', str(new_index))
                source_type = self.type_source_dict.get(new_type, "unknown")
            pic_list.append(source_type)
            if source_type == "unknown":
                continue
            if source_type is None:
                break

            if t not in t_no_dup:
                t_no_dup.append(t)
            else:
                continue
            type_list.append([i, self.idx_type_dict[t]])
        org_type_list = type_list
        type_list = sorted(type_list, key=lambda x: x[1])
        render_len = len(type_list)
        if last_canvas is not None:
            canvas = last_canvas
        # 先根据深度绘制天空中的物品
        for z_ in range(2, -1, -1):
            for [idx, _] in type_list:
                if pic_list[idx][0] == 's':
                    if zs[idx] == z_:
                        canvas = render_img(xs[idx], ys[idx], zs[idx], fs[idx], pic_list[idx], canvas)
                else:
                    break
        # 根据深度绘制剩余物品
        for z_ in range(2, -1, -1):
            for [idx, _] in type_list:
                if pic_list[idx][0] != 's':
                    try:
                        if zs[idx] == z_:
                            canvas = render_img(xs[idx], ys[idx], zs[idx], fs[idx], pic_list[idx], canvas)
                    except:
                        print('bad')

        if save_file is not None:
            canvas.save(save_file)
        if data_file is not None:
            with open(data_file, 'w') as fw:
                for t in org_type_list:
                    i = int(t[0])
                    write_list = [str(pic_list[i]), str(t[1][1]), str(t[1][3:-1]),
                                  str(xs[i]), str(ys[i]), str(zs[i].item()), str(grids[i].item()), str(fs[i].item()),
                                  str(ps[i].item()),
                                  str(es[i].item())]
                    fw.write('\t'.join(write_list) + '\n')
        if scene_file is not None:
            base_name, ext = os.path.splitext(save_file)
            Image.open(os.path.join(self._scene_folder, scene_file)).save("{}_org.{}".format(base_name, ext))
            if data_file:
                copyfile(os.path.join(self._object_folder, scene_file.replace("png", "txt")),
                         data_file.replace("gen_data.txt", "raw_data.txt"))
        return canvas

    def recover_texts(self, text):
        words = [self._idx2word.get(idx, '<eos>') for idx in text]
        return [w for w in words if w not in INIT_WORD_DICT]

    def recover_types(self, type, gt_text, pad=INIT_WORD_DICT[EOS]):
        if pad in type:
            type = type[:type.index(pad)]
        # type = [t for t in type if t not in INIT_WORD_DICT.values()]
        words = []
        for idx in type:
            inter_words = self._list_intersection(self._instance2noun.get(idx, ['None']), gt_text)
            if len(inter_words) != 0:
                words.append((idx, inter_words[0]))
            else:
                words.append((idx, self._instance2noun.get(idx, ['None'])[0]))
        return words

    '''因为文字不能用numpy，所以统一用list'''

    def _get_real_type_list(self, type_list, gt_word_list, Return_raw_word=False):
        # type_valid_mask = [0 if w in INIT_WORD_DICT.values() else 1 for w in type_list]
        # type_list = list(filter(lambda t: t not in INIT_WORD_DICT.values(), type_list))
        try:
            eos_idx = type_list.index(INIT_WORD_DICT[EOS])
            type_list = type_list[:eos_idx]
        except ValueError:
            pass
        ins_word_list = [self._instance2idx[idx] for idx in type_list]
        type_word_mask_list = [[type_list[i], self._list_intersection(ins_word_list[i], gt_word_list)[0], 1]
                               if len(self._list_intersection(ins_word_list[i], gt_word_list)) != 0
                               else [type_list[i], ins_word_list[i][0], 0]
                               for i in range(0, len(ins_word_list))]
        try:
            type_list, word_list, real_mask = zip(*type_word_mask_list)
            type_list, word_list, real_mask = list(type_list), list(word_list), list(real_mask)
            real_tuple_list = list(filter(lambda t: t[2] == 1, type_word_mask_list))
        except ValueError:
            type_list, word_list, real_mask = [], [], []
            real_tuple_list = []
        if len(real_tuple_list) != 0:
            real_type_list, real_word_list, _ = zip(*real_tuple_list)
        else:
            real_type_list, real_word_list = [], []
        if Return_raw_word:
            sentence = self.recover_texts(gt_word_list)
            ins_word_recover = [self.recover_texts(ins_word) for ins_word in ins_word_list]
            all_word_recover = self.recover_texts(word_list)
            real_word_recover = self.recover_texts(real_word_list)
            return all_word_recover, real_mask, real_word_recover, real_type_list
        '''
        word list: types对应的word list
        real mask：types对应的是否在text中出现的标识
        real_word_list：types出现在text中对应的word
        real_type_list：出现在text中剩下的type
        '''
        return word_list, real_mask, real_word_list, real_type_list

    def _list_union(self, a, b):
        return list(set(a).union(set(b)))

    def _list_intersection(self, a, b):
        return list(set(a).intersection(set(b)))

    def _list_difference(self, a, b):
        return list(set(a).difference(set(b)))


class Text2CVDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(Text2CVDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        print("batch_size: ", self.batch_size)

    @property
    def next_batch(self):
        return next(iter(self))

    def _collate_fn(self, batch, is_text=False, is_objects=False, is_types=False):
        error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
        if is_types:
            return batch
        if is_objects:
            lengths = [len(data[object_info_dict["type"]]) for data in batch]
            new_batch = [[] for _ in range(len(batch[0]))]
            for data in batch:
                for i, attr in enumerate(data):
                    new_batch[i].append(attr)
            max_len = max(lengths)
            need_pad = min(lengths) != max_len
            for i, attr_batch in enumerate(new_batch):
                if i == object_info_dict["source"]:
                    continue
                if not need_pad:
                    new_batch[i] = torch.stack(attr_batch, 0)
                else:
                    shape = [len(batch), max_len]
                    if len(attr_batch[0]) == 0:
                        continue
                    if len(attr_batch[0].shape) == 2:
                        shape.append(attr_batch[0].shape[1])
                    else:
                        for k in range(1, len(attr_batch[0].shape)):
                            shape.append(attr_batch[0].shape[k])
                    if i == object_info_dict["type"]:
                        new_attr = attr_batch[0].new(*shape).fill_(INIT_WORD_DICT[EOS])
                    elif i == object_info_dict["step_scene"]:  # step scene, fill in [-1]
                        new_attr = attr_batch[0].new(*shape).fill_(0)
                    else:
                        new_attr = attr_batch[0].new(*shape).fill_(0)
                    for j, attr in enumerate(attr_batch):
                        new_attr[j][:len(attr)] = attr
                    new_batch[i] = new_attr.data
            new_batch.append(lengths)
            return new_batch
        elem_type = type(batch[0])
        if isinstance(batch[0], torch.Tensor):
            if not is_text:
                lengths = [len(data) for data in batch]
                if min(lengths) != max(lengths):
                    return batch
                return torch.stack(batch, 0)
            pad = INIT_WORD_DICT[PAD]
            max_len = max([torch.sum(data.ne(pad)) for data in batch])
            return torch.stack([data[-max_len:] for data in batch], 0)
        if elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if re.search('[SaUO]', elem.dtype.str) is not None:
                    raise TypeError(error_msg.format(elem.dtype))

                return torch.stack([torch.from_numpy(b) for b in batch], 0)
            if elem.shape == ():  # scalars
                py_type = float if elem.dtype.name.startswith('float') else int
                return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
        if isinstance(batch[0], int_classes):
            return torch.LongTensor(batch)
        if isinstance(batch[0], float):
            return torch.DoubleTensor(batch)
        if isinstance(batch[0], string_classes):
            return batch
        if isinstance(batch[0], collections.Mapping):
            return {key: self._collate_fn(
                [d[key] for d in batch], key == "text", key == "objects") for
                key in batch[0]}
        if isinstance(batch[0], collections.Sequence):
            transposed = zip(*batch)
            return [self._collate_fn(samples, is_text=False) for samples in transposed]
        raise TypeError((error_msg.format(type(batch[0]))))
        # "background.png"


__all__ = ["Text2CVDataset", "Text2CVDataLoader", "object_info_dict"]

if __name__ == '__main__':
    clear_cache_, init_only_ = False, False
    rev_dict = {i: k for k, i in object_info_dict.items()}


    def print_(data_):
        print("=" * 60)
        for k_, v_ in data_.items():
            if k_ == "objects":
                for i_, obj_ in enumerate(v_):
                    k__ = i_ if i_ >= len(rev_dict) else rev_dict[i_]
                    print(k__, "" if isinstance(obj_, list) else obj_.shape)
                    print(obj_)
                    print("-" * 60)
            else:
                print(k_, v_.shape if k_ != "scene" else v_)
                if k_ == "text":
                    print(v_)


    def verbose_(training_):
        if not training_:
            global clear_cache_, init_only_
            clear_cache_, init_only_ = False, True
        dataset_ = Text2CVDataset(DATA_NAME, {
            "clear_cache": clear_cache_, "training": training_}).make_dataset(init_only=init_only_)
        print(len(dataset_))
        loader = Text2CVDataLoader(dataset_, shuffle=False, batch_size=2)
        print("=" * 60)
        print_(loader.next_batch)
        print_(loader.next_batch)
        print("=" * 60)
        print('dataset_.validating', dataset_.validating)
        with dataset_.valid:
            print('dataset_.validating', dataset_.validating)
            print('loader.next_batch')
            print_(loader.next_batch)
        print('dataset_.validating', dataset_.validating)
        print("=" * 60)
        print('dataset_.type_source_dict', dataset_.type_source_dict)
        print('dataset_.scene_shape', dataset_.scene_shape)
        if training_:
            for pos_ in dataset_.pos_info:
                print('len(pos_)', len(pos_))
            for type_ in dataset_.type_info:
                print('type_', type_)
        print('dataset_.idx_type_dict', dataset_.idx_type_dict)


    verbose_(True)
    verbose_(False)
