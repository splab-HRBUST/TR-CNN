'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
'''
import math, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools import *

class GANLoss(nn.Module):

    # 定义判别器和生成器的损失函数
    def discriminator_loss(self, real_outputs, fake_outputs):
        """
        判别器的损失函数
        :param real_outputs: 判别器对真实数据的输出
        :param fake_outputs: 判别器对生成数据的输出
        :return: 判别器的损失
        """
        real_loss = torch.mean(
            F.binary_cross_entropy_with_logits(real_outputs, torch.ones_like(real_outputs)))
        fake_loss = torch.mean(
            F.binary_cross_entropy_with_logits(fake_outputs, torch.zeros_like(fake_outputs)))
        return real_loss + fake_loss

    def generator_loss(self, fake_outputs):
        """
        生成器的损失函数
        :param fake_outputs: 判别器对生成数据的输出
        :return: 生成器的损失
        """
        return torch.mean(
            F.binary_cross_entropy_with_logits(fake_outputs, torch.ones_like(fake_outputs)))


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

        self.AAM = AAMsoftmax(1211, 192, m=0.2, s=30)

    def forward(self, x, label):
        _, out1 = self.AAM(x, label)
        x = nn.functional.normalize(x)
        sp, sn = self.convert_label_to_similarity(x, label)
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)
        delta_p = 1 - self.m
        delta_n = self.m
        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma
        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        return loss, out1

    def convert_label_to_similarity(self, x, label):
        similarity_matrix = x @ x.transpose(1, 0)
        label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

        positive_matrix = label_matrix.triu(diagonal=1)
        negative_matrix = label_matrix.logical_not().triu(diagonal=1)

        similarity_matrix = similarity_matrix.view(-1)
        positive_matrix = positive_matrix.view(-1)
        negative_matrix = negative_matrix.view(-1)
        return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class AAMsoftmax(nn.Module):
    def __init__(self, n_class, dim, m, s):
        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, dim), requires_grad=False)
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss = self.ce(output, label)
        return loss, (output.detach(), )
        # return 0, (output.detach(), )


# class Control_Contrastive(nn.Module):
#     def __init__(self, n_class, dim, m, s):
#         super(Control_Contrastive, self).__init__()
#         self.m = m
#         self.m_center = m  #  0.2  11°
#         self.s = s
#         self.s_center = s
#         self.s_neg = 8
#
#         self.wsim_center = torch.nn.Parameter(torch.FloatTensor(n_class, dim), requires_grad=True)
#         self.wsim_sample = torch.nn.Parameter(torch.FloatTensor(n_class, dim), requires_grad=True)
#
#         self.n_class = n_class
#         self.sample_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
#         self.center_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
#
#         nn.init.xavier_normal_(self.wsim_center, gain=1)
#         nn.init.xavier_normal_(self.wsim_sample, gain=1)
#
#         self.cos_m = math.cos(self.m)
#         self.cos_m_center = math.cos(self.m_center)
#         self.sin_m = math.sin(self.m)
#         self.sin_m_center = math.sin(self.m_center)
#
#         self.th = math.cos(self.m)
#         # self.th = 0.95
#         self.th_center = math.cos(self.m_center)
#         # self.th_center = 0.92
#         self.mm = math.sin(math.pi - self.m) * self.m
#         # self.mm = math.sin(math.pi - self.m) * self.m
#         self.mm_center = math.sin(math.pi - self.m_center) * self.m_center
#         # self.mm_center = math.sin(math.pi - self.m_center) * (self.m_center ** 2)
#
#         self.soft_plus = nn.Softplus()
#
#     def forward(self, x1, x2, label=None):
#         # l2loss = 0
#         # logit_neg_loss = 0
#         # x1: updown  x2: mid   wsim_center: class  wsim_sample : sample
#         x1, x2 = F.normalize(x1, dim=1), F.normalize(x2, dim=1)
#         sim_center = F.linear(x1, F.normalize(self.wsim_center))
#         sim_sample = F.linear(x2, F.normalize(self.wsim_sample))
#
#         # 样本中心
#         labels_center, full_unique_index = label.unique(return_inverse=True)
#         avg_center = []
#         avg_neg = []
#         for labels in labels_center:
#             avg_center.append(sim_center[labels == label].mean(dim=0))
#             avg_neg.append(torch.cat((x1, x2), dim=1)[labels == label].mean(dim=0))
#
#         cosine_center = torch.stack(avg_center)
#         x_noSpk = torch.stack(avg_neg)
#
#     # 转的方向不一样
#         sine_center = torch.sqrt((1.0 - torch.mul(cosine_center, cosine_center)).clamp(0, 1))
#         phi_center_add = cosine_center * self.cos_m_center - sine_center * self.sin_m_center
#         # phi_unique_minus = cosine_center * self.cos_m_center + sine_center * self.sin_m_center
#         # phi_unique_high = torch.where((phi_center_add - phi_unique_minus) > 0, phi_center_add, phi_unique_minus)
#         # phi_unique_low = torch.where((phi_center_add - phi_unique_minus) < 0, phi_center_add, phi_unique_minus)
#         # phi_unique = torch.where((cosine_center - self.th_center) > 0, cosine_center - self.mm_center, phi_unique)
#         phi_unique = torch.where((cosine_center - self.th_center) > 0, cosine_center - self.mm_center, phi_center_add)
#         one_hot_center = torch.zeros_like(cosine_center)
#         one_hot_center.scatter_(1, labels_center.view(-1, 1), 1)
#         output_center = (one_hot_center * phi_unique) + ((1.0 - one_hot_center) * cosine_center)
#         output_center = output_center * self.s_center
#
#         # 样本单个
#         # cosine = F.normalize(torch.matmul(q, self.wv) * k, p=1, dim=-1)
#         cosine = sim_sample
#         sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
#         phi_add = cosine * self.cos_m - sine * self.sin_m
#         # phi_minus = cosine * self.cos_m + sine * self.sin_m
#         # phi_high = torch.where((phi_add - phi_minus) > 0, phi_add, phi_minus)
#         # phi_low = torch.where((phi_add - phi_minus) < 0, phi_add, phi_minus)
#         phi = torch.where((cosine - self.th) > 0, cosine - self.mm, phi_add)
#         # phi = torch.where((cosine - self.th) > 0, cosine - self.mm, phi)
#         one_hot = torch.zeros_like(cosine)
#         one_hot.scatter_(1, label.view(-1, 1), 1)
#         output_sample = (one_hot * phi) + ((1.0 - one_hot) * cosine)
#         output_sample = output_sample * self.s
#
#         aam_sample_loss = self.sample_ce(output_sample, label)
#         aam_center_loss = self.center_ce(output_center, labels_center)
#
#
#         logit_neg = x_noSpk
#         logit_neg = torch.matmul(logit_neg, logit_neg.T)  # [-2, 2] [-1, 1]
#         logit_neg = logit_neg * (1 - torch.eye(logit_neg.size(0), device=logit_neg.device))
#         # 2025年2月11日02:36:21 在大数据集表现不好，微调参数 +0.5
#         logit_neg = logit_neg[logit_neg != 0]
#
#     # 大的更大 小的更小，从而比值最小
#         # try 1 SOTA
#         # logit_neg = torch.abs(logit_neg)  # -2,2  0,4
#         # max_logit_neg = max(logit_neg).detach()
#         # logit_p = torch.where((torch.abs(F.normalize(logit_neg, dim=0)) - self.m) > 0, self.s_neg * logit_neg, 1 / self.s_neg * logit_neg)
#         # logit_neg_loss = self.soft_plus(torch.logsumexp(logit_p, dim=0) + self.m_neg)
#         # # try 2
#         # logit_neg = torch.abs(logit_neg)  # -2,2  0,2
#         # max_logit_neg = max(logit_neg)
#         # logit_neg = logit_neg / max_logit_neg
#         # logit_neg = torch.where((logit_neg - 0.1) > 0, self.s_neg * logit_neg, 1 / 8 * logit_neg)
#         # logit_neg_loss = self.soft_plus(torch.logsumexp(max_logit_neg * logit_neg, dim=0) -
#         #                                 torch.logsumexp((2 - max_logit_neg) * (1 - logit_neg), dim=0) + max_logit_neg)  # math.log(1+math.exp(-2.5)) = 0.07888973429254956
#
#         # # try 3
#         # new neg 2025年2月24日00:39:01
#         # logit_neg = torch.abs(logit_neg)  # -2,2  0,2
#         max_logit_neg = max(logit_neg)
#         median_logit_neg = logit_neg.median()
#         min_logit_neg = logit_neg.min()
#         rate_max_neg = logit_neg / max_logit_neg
#
#         # sp = (2 - max_logit_neg) + (1 - logit_neg)
#         # 目的 左边小 右边大
#
#         spk_l = 2 - max_logit_neg  # [4,0] max_logit_neg 控制在 self.m 下 变大
#         spk_r = max_logit_neg + rate_max_neg  # [-3,3]  变小
#         al = torch.clamp_min(2-spk_l.detach() + self.m, min=0.)
#         ar = torch.clamp_min(spk_r.detach() + self.m, min=0.)
#         delta_l = 2 - self.m
#         delta_r = self.m
#         logit_l = -al * (spk_l - delta_l) * self.s_neg  # 小于 delta_l
#         logit_r = ar * (spk_r - delta_r) * self.s_neg  # 大于 delta_r
#         logit_neg_loss = self.soft_plus(torch.logsumexp(logit_l, dim=0) + torch.logsumexp(logit_r, dim=0))
#
#         # logit_neg_loss = self.soft_plus(torch.logsumexp(max_logit_neg + logit_neg, dim=0) -
#         #                                 torch.logsumexp((2 - max_logit_neg) + (1 - logit_neg),
#         #                                                 dim=0) + max_logit_neg)
#
#         # if self.wq.grad is not None and self.wk.grad is not None:
#         #     self.wk.grad, self.wq.grad = F.normalize(self.wk.grad + 0.5 * self.wq.grad, dim=1), 0.001 * self.wq.grad
#         sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + "\t neg: %.4f, center: %.4f, sample: %.4f, max_logit_neg: %.4f, median_logit_neg:%.4f, min_logit_neg:%.4f\n" %
#                          (logit_neg_loss, aam_center_loss, aam_sample_loss, max_logit_neg, median_logit_neg, min_logit_neg))
#         return logit_neg_loss + aam_center_loss + aam_sample_loss, output_sample.detach(), output_center[full_unique_index].detach()
#         # return 0.1 * logit_neg_loss + 2 * aam_center_loss + aam_sample_loss + l2loss, output.detach(), output_center[full_unique_index].detach()
    # 论文
class Control_Contrastive(nn.Module):
    def __init__(self, n_class, dim, m, s):
        super(Control_Contrastive, self).__init__()
        self.pc = 0
        self.m = m
        self.m_center = m  # 0.2  11°
        self.s = s
        self.s_neg = 1
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, dim), requires_grad=True)

        self.n_class = n_class
        self.sample_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.center_ce = nn.CrossEntropyLoss(label_smoothing=0.1)

        nn.init.xavier_normal_(self.weight, gain=1)

        self.cos_m = math.cos(self.m)
        self.cos_m_center = math.cos(self.m_center)
        self.sin_m = math.sin(self.m)
        self.sin_m_center = math.sin(self.m_center)

        self.th = math.cos(self.m)
        # self.th = 0.95
        self.th_center = math.cos(self.m_center)
        # self.th_center = 0.92
        self.mm = math.sin(math.pi - self.m) * self.m
        # self.mm = math.sin(math.pi - self.m) * self.m
        self.mm_center = math.sin(math.pi - self.m_center) * self.m_center
        # self.mm_center = math.sin(math.pi - self.m_center) * (self.m_center ** 2)

        self.soft_plus = nn.Softplus()

    def forward(self, x, label=None):
        # l2loss = 0
        logit_neg_loss = 0
        aam_center_loss = 0
        # x1: updown  x2: mid   wsim_center: class  wsim_sample : sample
        x1 = F.normalize(x, dim=1)
        x2 = x1.clone()
        weight = F.normalize(self.weight)
        cosine = F.linear(x1, weight)
        avg_center = []
        avg_neg = []

        labels_center, full_unique_index = label.unique(return_inverse=True)
        for labels in labels_center:
            avg_center.append(F.linear(x2[labels == label].mean(dim=0), weight))
            avg_neg.append(torch.cat((x1, x2), dim=1)[labels == label].mean(dim=0))

        cosine_center = torch.stack(avg_center)
        sine_center = torch.sqrt((1.0 - torch.mul(cosine_center, cosine_center)).clamp(0, 1))
        phi_center_add = cosine_center * self.cos_m_center - sine_center * self.sin_m_center
        phi_unique = torch.where((cosine_center - self.th_center) > 0, cosine_center - self.mm_center, phi_center_add)
        one_hot_center = torch.zeros_like(cosine_center)
        one_hot_center.scatter_(1, labels_center.view(-1, 1), 1)
        output_center = (one_hot_center * phi_unique) + ((1.0 - one_hot_center) * cosine_center)
        output_center = output_center * self.s
        logit_neg = torch.stack(avg_neg)

        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi_add = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, cosine - self.mm, phi_add)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output_sample = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output_sample = output_sample * self.s

        aam_sample_loss = 0.5 * self.sample_ce(output_sample, label)
        aam_center_loss = 0.5 * self.center_ce(output_center, labels_center)

        logit_neg = torch.matmul(logit_neg, logit_neg.T)  # [-2, 2] [-1, 1]
        logit_neg = logit_neg * (1 - torch.eye(logit_neg.size(0), device=logit_neg.device))
        logit_neg = logit_neg[logit_neg != 0]

        max_logit_neg = max(logit_neg)
        median_logit_neg = logit_neg.median()
        min_logit_neg = logit_neg.min()
        # rou
        rate_max_neg = logit_neg / max_logit_neg
        logit_neg_loss = self.soft_plus(torch.logsumexp(self.s_neg * logit_neg, dim=0) -
                                        torch.logsumexp((2 - max_logit_neg) * (1-rate_max_neg),
                                                        dim=0) + max_logit_neg)

        # if self.wq.grad is not None and self.wk.grad is not None:
        #     self.wk.grad, self.wq.grad = F.normalize(self.wk.grad + 0.5 * self.wq.grad, dim=1), 0.001 * self.wq.grad
        if self.pc % 10 == 0:
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + "\t Loss n: %.4f, c: %.4f, s: %.4f,\t\t\t neg: max_n: %.4f, median_n:%.4f, min_n:%.4f\n" %
                             (logit_neg_loss, aam_center_loss, aam_sample_loss, max_logit_neg, median_logit_neg, min_logit_neg))
        self.pc += 1
        return logit_neg_loss + aam_center_loss + aam_sample_loss, (output_sample.detach(), output_center[full_unique_index].detach())
        # return 0.1 * logit_neg_loss + 2 * aam_center_loss + aam_sample_loss + l2loss, output.detach(), output_center[full_unique_index].detach()


# 此时去掉neg loss
# class Control_Contrastive(nn.Module):
#     def __init__(self, n_class, dim, m, s):
#         super(Control_Contrastive, self).__init__()
#         self.m = m
#         self.m_center = m  #  0.2  11°
#         self.s = s
#         self.s_center = s
#         self.s_neg = 1  # 默认 8
#
#         self.wsim_center = torch.nn.Parameter(torch.FloatTensor(n_class, dim), requires_grad=True)
#         self.wsim_sample = torch.nn.Parameter(torch.FloatTensor(n_class, dim), requires_grad=True)
#
#         self.n_class=n_class
#         self.sample_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
#         self.center_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
#
#         nn.init.xavier_normal_(self.wsim_center, gain=1)
#         nn.init.xavier_normal_(self.wsim_sample, gain=1)
#
#         self.cos_m = math.cos(self.m)
#         self.cos_m_center = math.cos(self.m_center)
#         self.sin_m = math.sin(self.m)
#         self.sin_m_center = math.sin(self.m_center)
#
#         self.th = math.cos(self.m)
#         # self.th = 0.95
#         self.th_center = math.cos(self.m_center)
#         # self.th_center = 0.92
#         self.mm = math.sin(math.pi - self.m) * self.m
#         # self.mm = math.sin(math.pi - self.m) * self.m
#         self.mm_center = math.sin(math.pi - self.m_center) * self.m_center
#         # self.mm_center = math.sin(math.pi - self.m_center) * (self.m_center ** 2)
#
#         self.soft_plus = nn.Softplus()
#
#     def forward(self, x1, x2, label=None):
#         l2loss = 0
#         # logit_neg_loss = 0
#         # x1: updown  x2: mid   wsim_center: class  wsim_sample : sample
#         x1, x2 = F.normalize(x1, dim=1), F.normalize(x2, dim=1)
#         sim_center = F.linear(x1, F.normalize(self.wsim_center))
#         sim_sample = F.linear(x2, F.normalize(self.wsim_sample))
#
#         # 样本中心
#         labels_center, full_unique_index = label.unique(return_inverse=True)
#         avg_center = []
#         # avg_neg = []
#         for labels in labels_center:
#             avg_center.append(sim_center[labels == label].mean(dim=0))
#             # avg_neg.append(torch.cat((x1, x2), dim=1)[labels == label].mean(dim=0))
#
#         cosine_center = torch.stack(avg_center)
#         # x_noSpk = torch.stack(avg_neg)
#
#     # 转的方向不一样
#         sine_center = torch.sqrt((1.0 - torch.mul(cosine_center, cosine_center)).clamp(0, 1))
#         phi_center_add = cosine_center * self.cos_m_center - sine_center * self.sin_m_center
#         # phi_unique_minus = cosine_center * self.cos_m_center + sine_center * self.sin_m_center
#         # phi_unique_high = torch.where((phi_center_add - phi_unique_minus) > 0, phi_center_add, phi_unique_minus)
#         # phi_unique_low = torch.where((phi_center_add - phi_unique_minus) < 0, phi_center_add, phi_unique_minus)
#         # phi_unique = torch.where((cosine_center - self.th_center) > 0, cosine_center - self.mm_center, phi_unique)
#         phi_unique = torch.where((cosine_center - self.th_center) > 0, cosine_center - self.mm_center, phi_center_add)
#         one_hot_center = torch.zeros_like(cosine_center)
#         one_hot_center.scatter_(1, labels_center.view(-1, 1), 1)
#         output_center = (one_hot_center * phi_unique) + ((1.0 - one_hot_center) * cosine_center)
#         output_center = output_center * self.s_center
#
#         # 样本单个
#         # cosine = F.normalize(torch.matmul(q, self.wv) * k, p=1, dim=-1)
#         cosine = sim_sample
#         sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
#         phi_add = cosine * self.cos_m - sine * self.sin_m
#         # phi_minus = cosine * self.cos_m + sine * self.sin_m
#         # phi_high = torch.where((phi_add - phi_minus) > 0, phi_add, phi_minus)
#         # phi_low = torch.where((phi_add - phi_minus) < 0, phi_add, phi_minus)
#         phi = torch.where((cosine - self.th) > 0, cosine - self.mm, phi_add)
#         # phi = torch.where((cosine - self.th) > 0, cosine - self.mm, phi)
#         one_hot = torch.zeros_like(cosine)
#         one_hot.scatter_(1, label.view(-1, 1), 1)
#         output_sample = (one_hot * phi) + ((1.0 - one_hot) * cosine)
#         output_sample = output_sample * self.s
#
#         aam_sample_loss = self.sample_ce(output_sample, label)
#         aam_center_loss = self.center_ce(output_center, labels_center)
#
#
#     #     logit_neg = x_noSpk
#     #     logit_neg = torch.matmul(logit_neg, logit_neg.T)  # [-2, 2] [-1, 1]
#     #     logit_neg = logit_neg * (1 - torch.eye(logit_neg.size(0), device=logit_neg.device))
#     #     logit_neg = logit_neg[logit_neg != 0]
#     #
#     # # 大的更大 小的更小，从而比值最小
#     #     # try 1 SOTA
#     #     # logit_neg = torch.abs(logit_neg)  # -2,2  0,4
#     #     # max_logit_neg = max(logit_neg).detach()
#     #     # logit_p = torch.where((torch.abs(F.normalize(logit_neg, dim=0)) - self.m) > 0, self.s_neg * logit_neg, 1 / self.s_neg * logit_neg)
#     #     # logit_neg_loss = self.soft_plus(torch.logsumexp(logit_p, dim=0) + self.m_neg)
#     #     # # try 2
#     #     logit_neg = torch.abs(logit_neg)  # -2,2  0,2
#     #     max_logit_neg = self.s_neg * max(logit_neg)
#     #     logit_neg = logit_neg / max_logit_neg
#     #     # logit_neg = torch.where((logit_neg - 0.1) > 0, self.s_neg * logit_neg, 1 / 8 * logit_neg)
#     #     logit_neg_loss = self.soft_plus(torch.logsumexp(max_logit_neg * logit_neg, dim=0) -
#     #                                     torch.logsumexp((2 - max_logit_neg) * (1-logit_neg), dim=0) + max_logit_neg)  # math.log(1+math.exp(-2.5)) = 0.07888973429254956
#     #
#
#         # if self.wq.grad is not None and self.wk.grad is not None:
#         #     self.wk.grad, self.wq.grad = F.normalize(self.wk.grad + 0.5 * self.wq.grad, dim=1), 0.001 * self.wq.grad
#         sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + "\t neg: %.4f, center: %.4f, sample: %.4f, max_logit_neg: %.4f\n" % (0, aam_center_loss, aam_sample_loss, 0))
#         return aam_center_loss + aam_sample_loss + l2loss, output_sample.detach(), output_center[full_unique_index].detach()
#         # return 0.1 * logit_neg_loss + 2 * aam_center_loss + aam_sample_loss + l2loss, output.detach(), output_center[full_unique_index].detach()


class AAM_Control_Contrastive(nn.Module):
    def __init__(self, n_class, dim, m, s):
        super(AAM_Control_Contrastive, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, dim), requires_grad=True)
        self.weight_m = torch.nn.Parameter(torch.FloatTensor(n_class, dim), requires_grad=True)
        self.weight_n = torch.nn.Parameter(torch.FloatTensor(n_class, dim), requires_grad=True)
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.soft_plus = nn.Softplus()
        nn.init.xavier_normal_(self.weight, gain=1)
        nn.init.xavier_normal_(self.wq, gain=1)
        nn.init.uniform_(self.wk)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th1 = math.cos(self.m)
        self.th2 = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
    def forward(self, x, label=None):

        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

        cosine_m = F.linear(F.normalize(x), F.normalize(self.weight_m))
        x_m = cosine_m[label_matrix]

        cosine_n = F.linear(F.normalize(x), F.normalize(self.weight_m))

        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        sine_m = torch.sqrt((1.0 - torch.mul(cosine_m, cosine_m)).clamp(0, 1))

        phi = cosine * self.cos_m - sine * self.sin_m
        phi_m = cosine * self.cos_m - sine * self.sin_m

        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        phi_m = torch.where((cosine_m - self.th) > 0, cosine_m, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        AAM_loss = self.ce(output, label)



        B, D = x.size()
        v = F.linear(F.normalize(x), F.normalize(self.w[label, :]))
        q = F.linear(F.normalize(x), F.normalize(self.wq[label, :]))
        k = F.linear(F.normalize(x), F.normalize(self.wk[label, :]))
        similarity_matrix = q * k

        label_matrix = label.unsqueeze(1) == label.unsqueeze(0)
        ap_m = torch.tensor([torch.mean(similarity_matrix[i, label_matrix[i]]) for i in range(B)], device=x.device)
        ap = torch.diag(similarity_matrix)
        an = torch.where(label_matrix.logical_not()==True, similarity_matrix, 0)
        # an = similarity_matrix[label_matrix.logical_not()]
        # ap = torch.clamp_min(- ap.detach() + 1 + self.m1, min=0.)
        # an = torch.clamp_min(an.detach() + self.m1, min=0.)

        cosine_ap =  ap.clamp(0, 1)
        sine_ap = torch.sqrt((1.0 - cosine_ap).clamp(0, 1))
        cosine_ap_m = ap_m.clamp(0, 1)
        sine_ap_m = torch.sqrt((1.0 - cosine_ap_m).clamp(0, 1))
        cosine_an =  an.clamp(0, 1)
        sine_an = torch.sqrt((1.0 - cosine_an).clamp(0, 1))

# cos(a+b+m)
        phi_pm_cosin = cosine_ap * cosine_ap_m - sine_ap * sine_ap_m
        phi_pm_sine = torch.sqrt((1.0 - phi_pm_cosin).clamp(0, 1))
        phi_pm = phi_pm_cosin * self.cos_m - phi_pm_sine * self.sin_m

        logit_neg = 1 - phi_pm
        # logit_neg = self.s//3 * logit_neg
        # logit_neg = phi_pm

        phi_nm_sine = sine_an * cosine_ap_m + cosine_an * sine_ap_m
        phi_nm_cosin = torch.sqrt((1.0 - phi_nm_sine).clamp(0, 1))
        phi_nm = phi_nm_sine * self.cos_m - phi_nm_cosin * self.sin_m

        logit_n = phi_nm[label_matrix.logical_not()]
        # logit_n = torch.where((logit_n - self.th) > 0, phi_pm, phi_pm_cosin)

        # logit_neg = ap_m.detach() - ap.detach() + delta_p
        # logit_n = ap_m.detach() - an.detach() + delta_n
        Control_Contrastive_loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_neg, dim=0))
        return AAM_loss + Control_Contrastive_loss



class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

        self.sigmod = nn.Sigmoid()
        self.MSE_criterion = nn.MSELoss()
    def forward(self, x, y):
        # x = torch.exp(x)
        # y = torch.exp(y)
        loss = self.MSE_criterion(x, y)
        return loss
class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.KL_criterion    = nn.KLDivLoss(reduction='batchmean', log_target=False)

    def forward(self, input, target):
        # input = F.normalize(input)
        # target = F.normalize(target)

        # input = F.softmax(input)
        input = F.log_softmax(input)
        target = F.log_softmax(target, dim=1)
        # target = F.log_softmax(target, dim=1)
        kl_loss = self.KL_criterion(input, target)
        return kl_loss
