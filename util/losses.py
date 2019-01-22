import torch
import torch.nn.functional as functional
from constants import GRID_TYPE
from constants import *


class LossBase(torch.nn.Module):
    def __init__(self, config):
        super(LossBase, self).__init__()
        self._config = config

    def forward(self, logits, target, target_mask):
        pass


class MAE(LossBase):
    def forward(self, predictions, target, mask):
        return torch.abs(predictions - target).mean(dim=-1)[mask].mean()


class MSE(LossBase):
    def forward(self, predictions, target, mask):
        return torch.pow(predictions - target, 2).mean(dim=-1)[mask].mean()


class CrossEntropy(LossBase):
    def forward(self, logits, target, mask=None, return_all=False):
        logits_flat, target_flat = logits.view(-1, logits.shape[-1]), target.view(-1, 1)
        mask, log_prob_flat = mask.contiguous().view(-1, 1), functional.log_softmax(logits_flat, dim=-1)
        nll_losses_old = -log_prob_flat.gather(dim=-1, index=target_flat)
        temp = torch.sum(mask)
        nll_losses = nll_losses_old[mask]
        if not return_all:
            return nll_losses.mean()
        return log_prob_flat, nll_losses, mask


class LabelSmoothCrossEntropy(CrossEntropy):
    def __init__(self, config):
        super(LabelSmoothCrossEntropy, self).__init__(config)
        self._eps = self._config.setdefault("eps", 0.1)

    def forward(self, logits, target, mask, return_all=False):
        log_prob_flat, nll_losses, mask = super(
            LabelSmoothCrossEntropy, self).forward(logits, target, mask, True)
        smooth_losses = -log_prob_flat.sum(dim=-1, keepdim=True)[mask]
        nll_loss, smooth_loss = nll_losses.mean(), smooth_losses.mean()
        eps = self._eps / log_prob_flat.size(-1)
        loss = (1. - self._eps) * nll_loss + eps * smooth_loss
        if not return_all:
            return loss
        return log_prob_flat, nll_losses, smooth_losses, mask


class BasicLoss(LossBase):
    def __init__(self, config, type_head_num, att_head_num):
        super(BasicLoss, self).__init__(config)
        self.reg_loss = loss_dict[self._config.setdefault("reg_loss", "mae")](self._config)
        self.z_loss = loss_dict[self._config.setdefault("z_loss", "label_smooth_cross_entropy")](self._config)
        self.flip_loss = loss_dict[self._config.setdefault("flip_loss", "cross_entropy")](self._config)
        self.grid_loss_type = 'cross_entropy'
        self.grid_loss = loss_dict[self._config.setdefault("grid_loss", self.grid_loss_type)](self._config)
        self.type_loss = loss_dict[self._config.setdefault("type_loss", "cross_entropy")](self._config)
        self.pose_loss = loss_dict[self._config.setdefault("pose_loss", "cross_entropy")](self._config)
        self.expression_loss = loss_dict[self._config.setdefault("expression_loss", "cross_entropy")](self._config)
        self.wt = self._config.setdefault("weight_type", 8)
        self.wp = self._config.setdefault("weight_position", 2)
        self.wa = self._config.setdefault("weight_attribute", 1)
        self.wh = self._config.setdefault("weight_human", 2)
        self.mask_type = self._config.setdefault("mask_type", "all")
        self.attention_loss = True
        assert type_head_num == att_head_num
        self.head_num = type_head_num
        self.usegthuman = True

    def forward(self, batch, outputs, obj_attention_weights=None, att_attention_weights=None, return_all=False):
        mask = outputs["mask"]
        text_mask = outputs["text_mask"]
        type_logits = outputs["type_logits"]
        type_targets = outputs["type_targets"]
        type_samples = outputs["type_samples"]
        if self.usegthuman == True:
            boy_mask = type_targets == 4
            girl_mask = type_targets == 5
        else:
            boy_mask = type_samples == 4
            girl_mask = type_samples == 5
        human_mask = (boy_mask + girl_mask)
        human_mask *= mask
        pose = outputs["pose"]
        expression = outputs["expression"]
        pose_target = outputs["pose_target"]
        expression_target = outputs["expression_target"]
        flip = outputs["flip"]
        flip_target = outputs["flip_target"]
        if self.mask_type == "all":
            mask1 = torch.ones_like(mask)
        else:
            mask1 = mask
        if CLOSE_OBJ:
            loss = 0
        else:
            loss = self.wt * self.type_loss(type_logits, type_targets, mask1)
        if GRID_TYPE == 'gridxy':
            grid_x = outputs["grid_x"]
            grid_y = outputs["grid_y"]
            grid_target_x = outputs["grid_x_target"]
            grid_target_y = outputs["grid_y_target"]
            loss += self.wp * self.grid_loss(grid_x, grid_target_x, mask)
            loss += self.wp * self.grid_loss(grid_y, grid_target_y, mask)

        elif GRID_TYPE == 'gridx':
            grid = outputs.get("grid")
            grid_target = outputs.get("grid_target")
            loss += self.wp * self.grid_loss(grid, grid_target, mask)
        z = outputs["z"]
        # x, y, z = outputs["x"], outputs["y"], outputs["z"]
        x_target, y_target, z_target = outputs["x_target"], outputs["y_target"], outputs["z_target"]
        loss += self.wa * self.flip_loss(flip, flip_target, mask)
        loss += self.wh * self.pose_loss(pose, pose_target, human_mask)
        loss += self.wh * self.expression_loss(expression, expression_target, human_mask)
        loss += self.wa * self.z_loss(z, z_target, mask)

        if self.attention_loss and len(obj_attention_weights) != 0:
            attention_loss = self.attention_penalty(mask, obj_attention_weights, text_mask)
            loss += attention_loss
        if self.attention_loss and len(att_attention_weights) != 0:
            attention_loss = self.attention_penalty(mask, att_attention_weights, text_mask)
            loss += attention_loss
        return loss

    def attention_penalty(self, mask, general_attention_weights, text_mask):
        attention_weights = general_attention_weights[0]
        t_s = attention_weights.shape[1]
        s_l = attention_weights.shape[2]
        com_mask = self.get_com_mask(mask, text_mask).float()
        # weight_s_sum = torch.sum(attention_weights, 2)#=1
        attention_weights_mask = attention_weights * com_mask
        # weight_s_sum1 = torch.sum(attention_weights_mask, 2)#=1
        '''得到不同时间步对于一个单词的关注程度'''
        weight_t_sum = torch.sum(attention_weights_mask, 1)  # [256*4, 52]
        '''对所有单词做正则'''
        text_mask_1 = self.get_text_mask(text_mask).float()
        weight_t_minus = (1 - weight_t_sum) * text_mask_1.float()  # [256*4, 52]
        weight_t_squre = torch.pow(weight_t_minus, 2)
        weight_w_sum = torch.sum(weight_t_squre, 1)  # [256*4]
        weight_mean = torch.mean(weight_w_sum)  # [1]
        attention_loss = self.wa * weight_mean
        return attention_loss

    def get_com_mask(self, tsmask, textmask):
        a1 = tsmask.unsqueeze(-1)
        b1 = textmask.unsqueeze(-2)
        c = a1 * b1
        shape = [-1] + [s for s in c.shape[1:]]
        # return torch.cat([c] * self.head_num, dim=1).view(shape)
        return torch.cat([c] * self.head_num, dim=0)

    def get_text_mask(self, textmask):
        shape = [-1] + [s for s in textmask.shape[1:]]
        return torch.cat([textmask] * self.head_num, dim=0)
        # return torch.cat([textmask] * self.head_num, dim=1).view(shape)


loss_dict = {
    "basic": BasicLoss,
    "mae": MAE, "mse": MSE,
    "cross_entropy": CrossEntropy, "label_smooth_cross_entropy": LabelSmoothCrossEntropy,
}

__all__ = ["loss_dict"]
