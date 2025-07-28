'''
This is the ECAPA-TDNN model.
This model is modified and combined based on the following three projects:
  1. https://github.com/clovaai/voxceleb_trainer/issues/86
  2. https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
  3. https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py
  4. https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/model.py#L31
  5. # 构件 介绍  yzx#   Param : C(1024,512)
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class wav_pre_process_fbank_extract(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min=20, f_max=7600, power=2,
                                                 window_fn=torch.hamming_window, n_mels=80),
        )

        # self.spectrogram = torchaudio.transforms.Spectrogram(
        #     n_fft=256,
        #     win_length=250,
        #     hop_length=100,
        #     pad=0,
        #     window_fn=torch.hamming_window,
        #     power=2.0,
        #     normalized=False,
        #     center=True,
        #     pad_mode="reflect",
        #     onesided=True,
        # )
        self.specaug = FbankAug()

    def forward(self, x):
        with torch.no_grad():  # X正负约各一半
            x = self.torchfbank(x) + 1e-6  # all X>0
            # cepstral mean subtraction
            x = x.log()  # 正/负：40~80
            x = x - torch.mean(x, dim=-1, keepdim=True)  # X正负约各一半
            # no mask
        return x

class wav_pre_process_fbank(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min=20, f_max=7600, power=2,
                                                 window_fn=torch.hamming_window, n_mels=80),
        )

        # self.spectrogram = torchaudio.transforms.Spectrogram(
        #     n_fft=256,
        #     win_length=250,
        #     hop_length=100,
        #     pad=0,
        #     window_fn=torch.hamming_window,
        #     power=2.0,
        #     normalized=False,
        #     center=True,
        #     pad_mode="reflect",
        #     onesided=True,
        # )
        self.specaug = FbankAug()

    def forward(self, x, aug=True):
        with torch.no_grad():  # X正负约各一半
            x = self.torchfbank(x) + 1e-12  # all X>0
            # cepstral mean subtraction
            x = x.log()  # 正/负：40~80
            x = x - torch.mean(x, dim=-1, keepdim=True)  # X正负约各一半

            # 做2s特征
            if aug == True:
                x = self.specaug(x)

            # x = self.spectrogram(x) + 1e-6
            # x = x.log()
            # x = F.normalize(x, 2, dim=2)
            # if aug == True:
            #    x = self.specaug(x)
        return x

class wav_pre_process_stft(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # self.torchfbank = torch.nn.Sequential(
        #     PreEmphasis(),
        #     torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
        #                                          f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80),
        # )

        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=512,
            win_length=400,
            hop_length=160,
            pad=0,
            window_fn=torch.hamming_window,
            power=None,
            normalized=False,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.specaug = FbankAug(freq_mask_width=(0, 32), time_mask_width=(0, 5))

    def forward(self, x, aug):
        with torch.no_grad():  # X正负约各一半
            # x = self.stft(x) + 1e-6
            x = self.stft(x)[:, 0:-1, :].real

            if aug == True:
                x = self.specaug(x)

            # x = self.spectrogram(x) + 1e-6
            # x = x.log()
            # x = F.normalize(x, 2, dim=2)
            # if aug == True:
            #    x = self.specaug(x)
        return x

# 预处理语音 定义
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
class PreEmphasis(nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x.unsqueeze(1)
        x = F.pad(x, (1, 0), 'reflect')
        return F.conv1d(x, self.flipped_filter).squeeze(1)
class FbankAug(nn.Module):
    # ecapa def __init__(self, freq_mask_width=(0, 10), time_mask_width=(0, 5), aug=False):
    #  code def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10), aug=False):

    def __init__(self, freq_mask_width=(0, 10), time_mask_width=(0, 5), aug=False):
    # def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10), aug=False):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        self.aug = aug
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x, aug=False):
        self.aug = aug
        if self.aug:
            x = self.mask_along_axis(x, dim=2)
            x = self.mask_along_axis(x, dim=1)
        return x
class wav_pre_process_melfbank(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80),
        )

        # self.spectrogram = torchaudio.transforms.Spectrogram(
        #     n_fft=256,
        #     win_length=250,
        #     hop_length=100,
        #     pad=0,
        #     window_fn=torch.hamming_window,
        #     power=2.0,
        #     normalized=False,
        #     center=True,
        #     pad_mode="reflect",
        #     onesided=True,
        # )
        self.specaug = FbankAug()

    def forward(self, x, aug):
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            # 离散余弦变换（DCT）, so not mfcc
            # Mel Spectrogram 保存了更多的时间 - 频率信息  同理 引出升维
            x = x.log()
            # cepstral mean subtraction
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

            # x = self.spectrogram(x) + 1e-6
            # x = x.log()
            # x = F.normalize(x, 2, dim=2)
            # if aug == True:
            #    x = self.specaug(x)
        return x
class Complex_wav_pre_process_melfbank(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.pre = PreEmphasis()
        # self.spectrogram = torchaudio.transforms.Spectrogram(
        #     n_fft=512,
        #     win_length=400,
        #     hop_length=160,
        #     pad=0,
        #     window_fn=torch.hamming_window,
        #     power=None,
        #     onesided=False,
        # )
        # self.mel_scale = torchaudio.transforms.MelScale(
        #     n_mels=80, sample_rate=16000, f_min=20, f_max=7600, n_stft=512  # 复数全取 实数 n_stft//2 +1
        # )

        self.melSpec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400,
                                                            hop_length=160, \
                                                            f_min=20, f_max=7600, window_fn=torch.hamming_window,
                                                            n_mels=80)

        self.specaug = FbankAug()

    def forward(self, x, aug):
        with torch.no_grad():
            x = self.pre(x)
            # x = torch.complex(x, torch.zeros_like(x))
            # x = self.spectrogram(x)
            # x_real = self.mel_scale(x.real) + 1e-6
            # x_imag = self.mel_scale(x.imag) + 1e-6
            # x = torch.complex(x_real, x_imag)

            x = self.melSpec(x)
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)
        return x

# QKVCNN 方便跑实验 改为PConv  分别用n=1,2 去模拟
# class TR_Conv(nn.Module):
#     # unit_FZ 一定要是奇数 不然不好处理
#     def __init__(self, in_channels=1, out_channels=1, m=2, n=1, groups=None, dilation=1, bias=True):
#         super().__init__()
#         self.dilation = dilation
#         self.dilationParam = 1  # 固定步长 进行跳帧拼接
#         # 主要参数
#         self.m = m
#         self.groups = groups if groups is not None else 2 ** n
#         # self.boundary = 1
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.unit_FZ = 1 + math.floor(2 ** (n-1))  # x1  x1,0  此时为2  FRAME ZERO
#         self.kernel_size = self.unit_FZ * (m - 1) + 1
#         # self.numpad = self.kernel_size // 2
#         # 只有 self.kernel_size == 3 多的线性和在偶数位上   ==5 ==7 ==均在奇数位置  因为 NUM//2 +1 一定为奇数
#         self.conv1d = nn.Conv1d(in_channels=in_channels,
#                                 out_channels=out_channels, kernel_size=self.kernel_size, stride=1, padding=self.unit_FZ-1, dilation=self.dilationParam, groups=self.groups, bias=bias)
#         self.bn = nn.BatchNorm1d(out_channels)
#         # self.channel_wight = torch.nn.Parameter(torch.FloatTensor(1, self.out_channels), requires_grad=True)
#         # self.relu = nn.ReLU()
#         # self.tanh = nn.Tanh()
#         # self.sigmoid = nn.Sigmoid()
#
#
#         # stdv = 1. / math.sqrt(self.conv1d.weight.size(1))
#         # self.conv1d.weight.data.uniform_(-stdv, stdv)
#
#         # nn.init.uniform_(self.conv1d.weight, a=0.0, b=1.0)
#         # nn.init.xavier_normal_(self.conv1d.weight, gain=1)
#
#         # full, x:R1, R1→R2
#     def pad_num(self, T, x):
#         if (T - x.shape[-1]) % 2 == 0:
#             padnum = (T - x.shape[-1]) // 2, (T - x.shape[-1]) // 2
#         else:
#             padnum = (T - x.shape[-1]) // 2, (T - x.shape[-1]) // 2 + 1
#         return padnum
#     def forward(self, x):  # v, r k=1, k=3
#
#         assert (
#                 x.ndim == 2 or x.ndim == 3
#         ), "x.shape must be 2 or 3  --err from yzx"
#         if x.ndim == 2:
#             x.unsqueeze_(-2)
#         BX, _, TX = x.shape
#
#         taylor_eq = torch.zeros(BX, self.out_channels, TX, device=x.device)
#         for i in range(self.dilation):
#             B, C, T = x[:, :, i::self.dilation].shape
#             x_zero_x_dila_input = torch.zeros(B, C, self.unit_FZ * (T-1) + 1, device=x.device)
#             x_zero_x_dila_input[:, :, 0::self.unit_FZ] = x[:, :, i::self.dilation]
#             x_zero_x_dila_out = self.conv1d(x_zero_x_dila_input)  # 1006+(4*2)-20
#             taylor_eq_list = []
#             # miu = torch.clamp(torch.abs(mu), min=1)  # 2024年12月18日22:21:44
#             # 获取初始化的泰勒展开元素
#
#             # n==2 来作差等价qkv
#             for j in range(self.unit_FZ):
#                 # 元素本身
#                 if j == 0:
#                     x_m_mk = x_zero_x_dila_out[:, :, self.unit_FZ-1::self.unit_FZ]
#                     taylor_eq_list.append(F.pad((x_m_mk), self.pad_num(T, x_m_mk), "reflect"))
#                 else:
#                     x_m1_tom_km1 = x_zero_x_dila_out[:, :, j-1+self.unit_FZ::self.unit_FZ] - x_zero_x_dila_out[:, :, :-self.unit_FZ][:, :, j-1::self.unit_FZ]
#                     taylor_eq_list.append(F.pad(x_m1_tom_km1, self.pad_num(T, x_m1_tom_km1), "reflect"))
#             # qkvn2_123 2025年2月22日16:40:39
#             # taylor_eq_dia = torch.softmax(taylor_eq_list[-2]*taylor_eq_list[-1], dim=1) * taylor_eq_list[0]
#             # qkvn2_312 2025年2月22日16:40:52
#             taylor_eq_dia = torch.softmax(taylor_eq_list[0]*taylor_eq_list[-1], dim=1) * taylor_eq_list[-2]
#
#             # # n==1 来直接等价qkv
#             # for j in range(self.unit_FZ):
#             #     # 元素本身
#             #     if j == 0:
#             #         x_m_mk = x_zero_x_dila_out[:, :, self.unit_FZ - 1::self.unit_FZ]
#             #         taylor_eq_list.append(F.pad((x_m_mk), self.pad_num(T, x_m_mk), "reflect"))
#             #     else:
#             #         q_element = x_zero_x_dila_out[:, :, j - 1 + self.unit_FZ::self.unit_FZ]
#             #         k_element = x_zero_x_dila_out[:, :, :-self.unit_FZ][:, :, j - 1::self.unit_FZ]
#             #         taylor_eq_list.append(F.pad(q_element, self.pad_num(T, q_element), "reflect"))
#             #         taylor_eq_list.append(F.pad(k_element, self.pad_num(T, k_element), "reflect"))
#             # taylor_eq_dia = torch.softmax(taylor_eq_list[1] * taylor_eq_list[2], dim=1) * taylor_eq_list[0]
#
#             taylor_eq[:, :, i::self.dilation] = taylor_eq_dia
#         return taylor_eq

# 卷积 定义  TR_Conv: Powers-of-Two Interpolation Convolution
class TR_Conv(nn.Module):
    # unit_FZ 一定要是奇数 不然不好处理
    def __init__(self, in_channels=1, out_channels=1, m=1, n=0, groups=1, dilation=1, drop=0,
                 bias=True, attention=False, head_paramisloate=False,
                 concatenate=False, G_add=False, wide_erf=True):
        super().__init__()
        # concatenate : true  并联 False    给一个参数G_add   组相加/组并联
        self.wide_erf = wide_erf
        # self.G_add = G_add
        self.G_add = not concatenate
        self.concatenate = concatenate
        self.attention = attention
        self.head_paramisloate = head_paramisloate
        self.all_m = True
        # self.cat_Ms = True  # Master-Slave
        self.taylor = False
        self.drop = nn.Dropout(drop)
        # self.relu = nn.ReLU()

        self.m = m
        self.n = n

        if m == 1 and n == 0:
            self.conv1d = nn.Conv1d(in_channels=in_channels,
                                    out_channels=out_channels, kernel_size=m, stride=1,
                                    padding=dilation * math.floor(m / 2), dilation=dilation, groups=groups,
                                    bias=bias)
            print(
                "'{}-Conc1d' object m:{} n:{} kernel_size:{}, groups:{}, in_channels:{}, out_channels:{}".format(
                    type(self).__name__, m, n, m, groups, in_channels, out_channels))
        else:
            # 初始化
            self.zero_num = math.floor(2 ** (n - 1))
            self.unit_FZ = 1 + self.zero_num  # 固定卷积核在空0的形状
            self.out_channels = out_channels

            if self.head_paramisloate and self.zero_num > 0:  # head 就是为了保持头之间独立
                assert (
                    # n & (n - 1) == 0  判断是否2的幂次方
                        self.zero_num > 1 or (m == 1 and self.zero_num == 0)
                ), "self.zero_num > 1， m-1个权重 对应m个元素 不同的权重对应算子：+/-" \
                   "  --  err pairs head_paramisloate  from yzx"
                self.head = self.zero_num // 2 if self.concatenate and self.zero_num > 0 else 1
            elif self.zero_num > 0:
                assert (
                    # n & (n - 1) == 0  判断是否2的幂次方
                        self.zero_num > 0 or (m == 1 and self.zero_num == 0)
                ), "self.zero_num > 0， m-1个权重 对应m个元素 同样的权重对应算子：+/-" \
                   "  --  err one head_paramisloate  from yzx"
                self.head = self.zero_num if self.zero_num > 0 else 1  # 默认为原始卷积
            else:
                self.head = 1

            if self.concatenate and self.zero_num > 0:  # 输出 out_channels * self.head
                assert (out_channels % self.head == 0) or (
                            m == 1 and n == 0), "concatenate： out_channels % self.head !=0  -- err from yzx"
                self.Pout_channels = out_channels // (self.head * 2)  # 每个头的每个单元都可以操作  因为维度相同
                self.groups = 1
            else:
                if self.G_add:
                    self.Pout_channels = out_channels
                    self.groups = self.head
                else:  # (G_cat)
                    self.Pout_channels = out_channels // self.head // 2
                    self.groups = groups
            #  通用版本 再一次时序 depth-wise 学习
            assert (
                    n > 0 or (n == 0 and m > 0)
            ), "n head(equal to n order param, shared param) at least must be 1   -- err for atention from yzx"
            # Q_list = []
            # K_list = []
            # for i in range(out_channels // self.Pout_channels * 2):
            #     # 等价于 m1_left  m1_right m
            #     Q_list.append(nn.Conv1d(in_channels=self.Pout_channels, out_channels=self.Pout_channels,
            #                             kernel_size=1, groups=self.Pout_channels))
            #     K_list.append(nn.Conv1d(in_channels=self.Pout_channels, out_channels=self.Pout_channels,
            #                             kernel_size=1, groups=self.Pout_channels))
            #     # V_list.append(nn.Conv1d(in_channels=self.Pout_channels, out_channels=self.Pout_channels,
            #     #                         kernel_size=1, groups=self.Pout_channels))
            # self.Q = nn.ModuleList(Q_list)
            # self.K = nn.ModuleList(K_list)
            # self.V = nn.ModuleList(V_list)
            # 为注意力设置的
            # self.V_bn = nn.BatchNorm1d(out_channels)
            self.gelu = QuickGELU()
            self.tanh = nn.Tanh()
            # self.drop = nn.Dropout(0.5)

            self.dilation = dilation
            if self.m > 0:
                self.kernel_size = self.unit_FZ * (m - 1) + 1
            else:
                self.kernel_size = self.unit_FZ
            self.in_channels = in_channels

            self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=self.Pout_channels,
                                    kernel_size=self.kernel_size, stride=1, padding=self.zero_num,
                                    dilation=1, groups=self.groups, bias=bias)
            print("'{}' object m:{} n:{} zero_num:{} kernel_size:{}, head:{},"
                  " groups:{}, cat:{}, head_paramisloate:{}, in_channels:{},"
                  " out_channels:{} Pout_channels:{} wide_erf:{}".format(
                type(self).__name__, self.m, self.n, self.zero_num, self.kernel_size,
                self.head, self.groups, self.concatenate,
                self.head_paramisloate, in_channels,
                self.out_channels, self.Pout_channels, self.wide_erf))

    def forward(self, x):  # v, r k=1, k=3

        assert (
                x.ndim == 2 or x.ndim == 3
        ), "x.shape must be 2 or 3  --err from yzx"
        if x.ndim == 2:
            x.unsqueeze_(-2)
        BX, _, TX = x.shape
        if self.n == 0:
            taylor_eq = self.conv1d(x)
        else:
            # 初始化最终结果
            taylor_eq = torch.zeros(BX, self.out_channels, TX, device=x.device)
            for i in range(self.dilation):  # 这是一个缺陷
                B, C, T = x[:, :, i::self.dilation].shape
                # 初始化 
                x_zero_x_dila_input = torch.zeros(B, C, self.unit_FZ * (T-1) + 1, device=x.device)
                taylor_eq_list = []
                
                x_zero_x_dila_input[:, :, 0::self.unit_FZ] = x[:, :, i::self.dilation]
                x_zero_x_dila_out = self.conv1d(x_zero_x_dila_input)  # 1006+(4*2)-20
                
                # 取 k = m 部分 只有一个
                x_m_mk = x_zero_x_dila_out[:, :, self.unit_FZ - 1::self.unit_FZ]
                x_m_mk = self.pad_num(T, x_m_mk, "x_m_mk")
                taylor_eq_list.append(x_m_mk)


                # fixme  记得处理这个元素
                # 取 富余 信息 深层网络 进行存储 获取多个卷积之后相邻元素时域的  和 / 差 / 最大 etc
                if self.wide_erf:
                    x_m_mk_unfold_m = self.unfold(x_m_mk, B, T)
                    x_m_mk_unfold_m = x_m_mk_unfold_m.squeeze(0).max(dim=-2)[0]

                j = 0
                tmp_head = 0
                while j < self.zero_num:
                    partial_m1_r_1 = x_zero_x_dila_out[:, :, j + self.unit_FZ::self.unit_FZ]
                    partial_m1_l_1 = x_zero_x_dila_out[:, :, :-self.unit_FZ][:, :, j::self.unit_FZ]

                    partial_m1_r_1 = self.pad_num(T, partial_m1_r_1, "partial_m1_r_1")
                    partial_m1_l_1 = self.pad_num(T, partial_m1_l_1, "partial_m1_l_1")

                    if self.head_paramisloate:  # 独立
                        partial_m1_r_2 = x_zero_x_dila_out[:, :, j + 1 + self.unit_FZ::self.unit_FZ]
                        partial_m1_l_2 = x_zero_x_dila_out[:, :, :-self.unit_FZ][:, :, j + 1::self.unit_FZ]
                        partial_m1_r_2 = self.pad_num(T, partial_m1_r_2, "partial_m1_r_2")
                        partial_m1_l_2 = self.pad_num(T, partial_m1_l_2, "partial_m1_l_2")
                        if self.attention:  # 再加一次参数
                            Q_m1_tom_km1 = partial_m1_r_1 - partial_m1_l_1
                            K_m1_tom_km1 = partial_m1_r_2 + partial_m1_l_2
                        else:  # 直接用zero后的参数
                            op_minus_m1tom_km1 = partial_m1_r_1 - partial_m1_l_1
                            op_add_m1tom_km1 = partial_m1_r_2 + partial_m1_l_2
                        j += 2
                    elif self.all_m:  # 共享 默认为True 基本单元为m个元素 ，在通道上就会少一半 多 2^(n-1)个通道 补多少0
                        if self.attention:  # m-1参数相同 构造数据m不同
                            Q_m1_tom_km1 = partial_m1_r_1 - partial_m1_l_1
                            K_m1_tom_km1 = partial_m1_r_1 + partial_m1_l_1
                        else:
                            op_minus_m1tom_km1 = partial_m1_r_1 - partial_m1_l_1
                            op_add_m1tom_km1 = partial_m1_r_1 + partial_m1_l_1
                        j += 1
                    else:   # 共享 相当于没加算子
                        if self.attention:
                            Q_m1_tom_km1 = partial_m1_r_1
                            K_m1_tom_km1 = partial_m1_l_1
                        else:
                            op_minus_m1tom_km1 = partial_m1_r_1
                            op_add_m1tom_km1 = partial_m1_l_1
                        j += 1
                        # 最后一层处理
                    if self.attention:
                        #  2025年3月1日02:13:00 需要注意力调整
                        Q_m1_tom_km1 = self.Q[tmp_head](Q_m1_tom_km1)
                        K_m1_tom_km1 = self.K[tmp_head](K_m1_tom_km1)
                        V_m_Km = taylor_eq_list[0]
                        # taylor_eq_list.append(self.gelu(Q_m1_tom_km1 + K_m1_tom_km1) * V_m_Km +
                        #                       self.drop(x_m_mk_unfold_m))
                        #  no V_bn → nan 2025年3月3日05:44:09
                        taylor_eq_list.append(self.tanh(Q_m1_tom_km1 + K_m1_tom_km1) * self.V_bn(V_m_Km))
                    else:
                        # todo
                        # depth_conv_minus = self.Q[tmp_head](op_minus_m1tom_km1)
                        # depth_conv_add = self.K[tmp_head](op_add_m1tom_km1)
                        depth_conv_minus = op_minus_m1tom_km1
                        depth_conv_add = op_add_m1tom_km1
                        depth_conv_m = taylor_eq_list[0]
                         # C 版本  T 版本 一体放缩 depth_conv_m
                        if self.G_add:  # 控制相加的时候约去
                            depth_conv_minus = self.gelu(depth_conv_minus)

                        # 控制更大感受野
                        if self.wide_erf:
                            taylor_eq_list.append(
                                depth_conv_minus + 1 / (self.head) * depth_conv_m + 1 / (self.head * 2) * self.drop(
                                    x_m_mk_unfold_m))
                            taylor_eq_list.append(depth_conv_add + 1 / (self.head) * depth_conv_m + 1 / (self.head * 2) * self.drop(x_m_mk_unfold_m))
                        else:
                            taylor_eq_list.append(
                                depth_conv_minus + 1 / (self.head) * depth_conv_m + 1 / (self.head * 2))
                            taylor_eq_list.append(
                                depth_conv_add + 1 / (self.head) * depth_conv_m + 1 / (self.head * 2))
                # 对于 中间的taylor_eq_list 进行处理
                if self.taylor:
                    for i in range(len(taylor_eq_list)):
                        taylor_eq_list[i] = taylor_eq_list[i] / math.factorial(i+1)
                if self.concatenate:  # 根据相拼接得到输出通道
                    taylor_eq_dia = torch.stack(taylor_eq_list[1:], dim=-2).view(B, -1, T)  # dim C
                    taylor_eq[:, :, i::self.dilation] = taylor_eq_dia
                elif self.G_add:  # 直接相加
                    # taylor_eq_dia = torch.sum(torch.stack(taylor_eq_list[1:], dim=0), dim=0)
                    taylor_eq_dia = torch.sum(torch.stack(taylor_eq_list[1:], dim=0), dim=0)
                    taylor_eq[:, :, i::self.dilation] = taylor_eq_dia
                else:  # 直接输出通道最后相拼接  得到多倍的输出通道 head*2
                    taylor_eq_dia = torch.stack(taylor_eq_list[1:], dim=2)
                    return taylor_eq_dia
                # taylor_eq_dia = torch.mean(torch.stack(taylor_eq_list, dim=0), dim=0)

        return taylor_eq


    def reset_param(self):
        # stdv = 1. / math.sqrt(self.conv1d.weight.size(1))
        # self.conv1d.weight.data.uniform_(-stdv, stdv)

        # nn.init.uniform_(self.conv1d.weight, a=0.0, b=1.0)
        # nn.init.xavier_normal_(self.conv1d.weight, gain=1)
        pass
    def pad_num(self, T, x, x_name):
        assert (
            x.shape[-1] <= T
        ), x_name + " shape[-1] > T, so no need pad -- err for yzx "
        if T > x.shape[-1]:
            if (T - x.shape[-1]) % 2 == 0:
                padnum = (T - x.shape[-1]) // 2, (T - x.shape[-1]) // 2
            else:
                padnum = (T - x.shape[-1]) // 2, (T - x.shape[-1]) // 2 + 1
            # return F.pad(x, padnum, mode='constant', value=0)
            return F.pad(x, padnum, 'reflect')
        else:
            return x
    def unfold(self, x_m_mk, B, T):
        return F.unfold(F.pad(x_m_mk.unsqueeze(0),
                                                   (math.floor((self.m - 1) / 2), math.ceil((self.m - 1) / 2)),
                                                   mode='constant', value=0),
                                             kernel_size=(self.Pout_channels, self.m), dilation=(1, 1), stride=(1, 1)) \
                            .view(1, B, self.Pout_channels, self.m, T)
class Unified_Meta_PConv(nn.Module):
    # unit_FZ 一定要是奇数 不然不好处理
    def __init__(self, in_channels=1, out_channels=1, m=2, n=1, groups=None, dilation=1, bias=True):
        super().__init__()
        self.dilation = dilation
        self.dilationParam = 1  # 固定步长 进行跳帧拼接
        # 主要参数
        self.m = m
        self.bnX_bold = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.groups = groups if groups is not None else 2 ** n
        self.boundary = math.sqrt(math.ceil((m-1)/2))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unit_FZ = 1 + math.floor(2 ** (n-1))  # x1  x1,0  此时为2  FRAME ZERO
        self.kernel_size = self.unit_FZ * (m - 1) + 1
        # self.numpad = self.kernel_size // 2
        # 只有 self.kernel_size == 3 多的线性和在偶数位上   ==5 ==7 ==均在奇数位置  因为 NUM//2 +1 一定为奇数
        self.conv1d = nn.Conv1d(in_channels=in_channels,
                                out_channels=out_channels, kernel_size=self.kernel_size, stride=1, padding=self.unit_FZ-1, dilation=self.dilationParam, groups=self.groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        # self.channel_wight = torch.nn.Parameter(torch.FloatTensor(1, self.out_channels), requires_grad=True)
        # self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        # self.sigmoid = nn.Sigmoid()


        # stdv = 1. / math.sqrt(self.conv1d.weight.size(1))
        # self.conv1d.weight.data.uniform_(-stdv, stdv)

        # nn.init.uniform_(self.conv1d.weight, a=0.0, b=1.0)
        # nn.init.xavier_normal_(self.conv1d.weight, gain=1)

        # full, x:R1, R1→R2
    def pad_num(self, T, x):
        if (T - x.shape[-1]) % 2 == 0:
            padnum = (T - x.shape[-1]) // 2, (T - x.shape[-1]) // 2
        else:
            padnum = (T - x.shape[-1]) // 2, (T - x.shape[-1]) // 2 + 1
        return padnum
    def forward(self, x):  # v, r k=1, k=3

        assert (
                x.ndim == 2 or x.ndim == 3
        ), "x.shape must be 2 or 3  --err from yzx"
        if x.ndim == 2:
            x.unsqueeze_(-2)
        BX, _, TX = x.shape

        taylor_eq = torch.zeros(BX, self.out_channels, TX, device=x.device)
        taylor_eq = self.bnX_bold(taylor_eq)
        taylor_eq = self.relu(taylor_eq)
        for i in range(self.dilation):
            B, C, T = x[:, :, i::self.dilation].shape
            x_zero_x_dila_input = torch.zeros(B, C, self.unit_FZ * (T-1) + 1, device=x.device)
            x_zero_x_dila_input[:, :, 0::self.unit_FZ] = x[:, :, i::self.dilation]
            x_zero_x_dila_out = self.conv1d(x_zero_x_dila_input)  # 1006+(4*2)-20
            taylor_eq_list = []
            # miu = torch.clamp(torch.abs(mu), min=1)  # 2024年12月18日22:21:44
            # 获取初始化的泰勒展开元素
            for j in range(self.unit_FZ):
                # 元素本身
                if j == 0:
                    x_m_mk = x_zero_x_dila_out[:, :, self.unit_FZ-1::self.unit_FZ]
                    # taylor_eq_list.append(F.pad((x_m_mk), self.pad_num(T, x_m_mk), "reflect"))
                    taylor_eq_list.append(F.pad((x_m_mk), self.pad_num(T, x_m_mk), "constant", 0))
                # 导数基本单元 直接计算一阶导
                    x_m_mk_unfold_m = F.unfold(F.pad(taylor_eq_list[0].detach().unsqueeze(0), (math.floor((self.m-1) / 2), math.ceil((self.m-1) / 2), 0, 0)),
                                           kernel_size=(self.out_channels, self.m), dilation=(1, 1), stride=(1, 1))\
                        .view(1, B, self.out_channels, self.m, T)
                    # x_m_mk_unfold_m: requires_grad False 减少计算
                    x_m_mk_unfold_m = (taylor_eq_list[0].unsqueeze(-2).detach() - x_m_mk_unfold_m.squeeze(0)).mean(dim=-2)  # 求均值 找变化率
                    x_m_mk_unfold_m = x_m_mk_unfold_m.clamp(min=1, max=self.boundary)
                    # x_m_mk_unfold_m = x_m_mk_unfold_m.clamp(min=0.9, max=1)
                else:
                    x_m1_tom_km1 = x_zero_x_dila_out[:, :, j-1+self.unit_FZ::self.unit_FZ] - x_zero_x_dila_out[:, :, :-self.unit_FZ][:, :, j-1::self.unit_FZ]
                    # x_m1_tom_km1 = 1 / (self.unit_FZ-1) * x_m1_tom_km1
                    # taylor_eq_list.append(F.pad(x_m1_tom_km1, self.pad_num(T, x_m1_tom_km1), "reflect") / x_m_mk_unfold_m)
                    taylor_eq_list.append(F.pad(x_m1_tom_km1, self.pad_num(T, x_m1_tom_km1),  "constant", 0) / x_m_mk_unfold_m)
            if len(taylor_eq_list) > 2:
                # 递归计算多阶导数：
                n_order = math.floor(math.log2(len(taylor_eq_list)-1)) + 1  # 加上上面计算的一阶导
                running_order = 2  # 将要算的阶数
                while running_order <= n_order:
                    cal_order_list = taylor_eq_list[-2 ** (n_order - running_order + 1):]
                    for k in range(2**(n_order-running_order)):  # 导数的运行次数
                          # 截取需要的数据列表
                        x_k_order = cal_order_list[2**(n_order-running_order) + k] - cal_order_list[2**(n_order-running_order)-k-1]
                        x_k_order = 1/2**(n_order-running_order) * x_k_order  # 不做任何约束
                        # taylor_eq_list.append(F.pad((math.factorial(running_order) * x_k_order), self.pad_num(T, x_k_order), "reflect") / x_m_mk_unfold_m)
                        taylor_eq_list.append(F.pad((math.factorial(running_order) * x_k_order), self.pad_num(T, x_k_order), "constant", 0) / x_m_mk_unfold_m)
                        # taylor_eq_list.append(nn.AdaptiveAvgPool1d(T)((miu-tmp)/math.factorial(run_num+1) * tmp))
                    running_order += 1

        # 包含多阶导数  暂时先按照泰勒 求和吧
            # fill = torch.mean(torch.stack(taylor_eq_list, dim=0), dim=0)
            taylor_eq_dia = torch.mean(torch.stack(taylor_eq_list, dim=0), dim=0)
            # taylor_eq_dia = F.normalize(taylor_eq_dia, dim=-1)
            taylor_eq[:, :, i::self.dilation] = taylor_eq_dia
        return taylor_eq
class DGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super().__init__()
        self.Adj_Dynamic = nn.Sequential(
            nn.PReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.BatchNorm1d(nfeat),
        )
        self.pReLU = nn.PReLU()
        # self.cov = GPT_IConv(nfeat, nfeat, bias=False)
        # nn.init.xavier_normal_(self.cov.weight)
    def forward(self, x):
        adj = self.Adj(x)
        adj = self.pReLU(adj)
        return adj
    def Adj(self, x):
        n = x.shape[1]
        I = torch.eye(n, n, requires_grad=False).to(x.device)
        Adj_Dynamic = self.Adj_Dynamic(x)
        sim = torch.matmul(Adj_Dynamic, Adj_Dynamic.permute(0, 2, 1))
        # 来自 长度相同， 等分乘积最大  99 21  88 64  77 49
        # A = torch.where(sim > (0.81+0.16), torch.ones_like(sim), torch.zeros_like(sim)) +\
        #     torch.where(sim > (0.64+0.25), torch.ones_like(sim), torch.zeros_like(sim)) +\
        #     torch.where(sim > (0.49+0.36), torch.ones_like(sim), torch.zeros_like(sim)) +\
        #     + 3 * I
        A = torch.where(sim > (0.81), torch.ones_like(sim), torch.zeros_like(sim)) + \
            + I
        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A, D_hat))
        # A_hat = self.cov(A_hat)
        return A_hat

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(-1)
        D_hat = torch.diag_embed(torch.pow(D, -0.5))
        return D_hat

class Unified_Meta_DGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super().__init__()
        self.Adj_Dynamic = nn.Sequential(
            nn.PReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.BatchNorm1d(nfeat),
        )
        self.pReLU = nn.PReLU()
        # self.cov = GPT_IConv(nfeat, nfeat, bias=False)
        # nn.init.xavier_normal_(self.cov.weight)
    def forward(self, x):
        adj = self.Adj(x)
        adj = self.pReLU(adj)
        return adj
    def Adj(self, x):
        n = x.shape[1]
        I = torch.eye(n, n, requires_grad=False).to(x.device)
        Adj_Dynamic = self.Adj_Dynamic(x)
        sim = torch.matmul(Adj_Dynamic, Adj_Dynamic.permute(0, 2, 1))
        # 来自 长度相同， 等分乘积最大  99 21  88 64  77 49
        # A = torch.where(sim > (0.81+0.16), torch.ones_like(sim), torch.zeros_like(sim)) +\
        #     torch.where(sim > (0.64+0.25), torch.ones_like(sim), torch.zeros_like(sim)) +\
        #     torch.where(sim > (0.49+0.36), torch.ones_like(sim), torch.zeros_like(sim)) +\
        #     + 3 * I
        A = torch.where(sim > (0.81), torch.ones_like(sim), torch.zeros_like(sim)) + \
            torch.where((1-sim) > (0.81), torch.ones_like(sim), torch.zeros_like(sim)) + \
            + 2 * I
        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A, D_hat))
        # A_hat = self.cov(A_hat)
        return A_hat

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(-1)
        D_hat = torch.diag_embed(torch.pow(D, -0.5))
        return D_hat

# SE  定义
class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x

class New_FwSEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.dim = -1
        self.fw_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 1)),
            nSqueeze(self.dim),
            TR_Conv(channels, bottleneck, m=1, n=2, concatenate=True),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck),  # I remove this layer
            TR_Conv(bottleneck, channels, m=1, n=2, concatenate=True),
            nn.Sigmoid(),
        )
        self.se_w = nn.Parameter(torch.FloatTensor(2, 2), requires_grad=True)
        nn.init.xavier_normal_(self.se_w, gain=1)

    def forward(self, input):
        x = self.fw_se(input)
        x = torch.matmul(x, self.se_w)
        return input * x.unsqueeze(self.dim)
# class New_Ponv2D(nn.Module):
#     def __init__(self, m=3, in_dim=256, freq_fuse_num=2):
#         super().__init__()
#         self.Ponv2D_1 = TR_Conv(m=m, n=1, in_channels=in_dim, out_channels=in_dim, G_add=False, groups=in_dim//freq_fuse_num)
#         self.Ponv_fuse = TR_Conv(m=1, n=0, in_channels=2 * in_dim, out_channels=in_dim, groups=in_dim//freq_fuse_num)
#         self.Pconv_bn1 = nn.BatchNorm1d( 2 * in_dim)
#         self.Pconv_bn2 = nn.BatchNorm1d(in_dim)
#         self.relu = nn.ReLU()
#     def forward(self, x):
#         B, _, T = x.shape
#         x = self.Ponv2D_1(x)
#         x = self.relu(x)
#         x = self.Pconv_bn1(x)
#         x = self.Ponv_fuse(x.view(B, -1, T))  # 算子挨着连接 主副 主是m  负为+-
#         x = self.relu(x)
#         x = self.Pconv_bn2(x)
#         return x
class New_Ponv2D(nn.Module):
    def __init__(self, m=3, n=1, in_dim=80, out_dim=512//4, concatenate=True):
        super().__init__()
        self.Ponv_ft = TR_Conv(m=m, n=n, in_channels=in_dim, out_channels=out_dim, concatenate=concatenate)
        # self.head = math.floor(2 ** (n - 1))
        # self.Ponv_ft = TR_Conv(m=m, n=n, in_channels=in_dim, out_channels=out_dim//(2 * self.head), groups=16)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.Ponv_ft(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class nSqueeze(nn.Module):
    def __init__(self, dim=None):
        super(nSqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        if self.dim is not None:
            return x.squeeze(dim=self.dim)
        return x.squeeze()
class New_SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.dim = -1
        self.new_se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            TR_Conv(channels, bottleneck, m=1, n=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # speechbrain  remove this layer
            TR_Conv(bottleneck, channels, m=1, n=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.new_se(input)
        return input * x

class UpChannel_lr_pn(nn.Module):

    def __init__(self, in_dim, out_dim, groups=None):
        super().__init__()
        self.upChannel1 = nn.Conv1d(in_dim, out_dim, kernel_size=5, stride=1, padding=2)
        self.upChannel2 = nn.Conv1d(in_dim, out_dim, kernel_size=5, stride=1, padding=2)
        self.upChannel3 = nn.Conv1d(in_dim, out_dim, kernel_size=5, stride=1, padding=2)
        self.upChannel4 = nn.Conv1d(in_dim, out_dim, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.pReLU_up1 = nn.PReLU()
        self.pReLU_up2 = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.bn4 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x1 = self.relu(x)
        x1 = self.upChannel1(x1 + x)
        x1 = self.elu(x1)
        x1 = self.bn1(x1)
        # x1 = self.dropout(x1)

        x2 = self.relu(-x)
        x2 = self.upChannel2(x2 - x)
        x2 = self.elu(x2)
        x2 = self.bn2(x2)
        # x2 = self.dropout(x2)

        x3 = self.pReLU_up1(x)
        x3 = self.upChannel3(x3 + x)
        x3 = self.elu(x3)
        x3 = self.bn3(x3)
        # x3 = self.dropout(x3)

        x4 = self.pReLU_up2(-x)
        x4 = self.upChannel4(x4 - x)
        x4 = self.elu(x4)
        x4 = self.bn4(x4)

        return x1, x2, x3, x4

class UpChannel_TDNN_PConv(nn.Module):

    def __init__(self, in_dim, out_dim, m=5, n=0, groups=None):
        super().__init__()
        self.upChannel1 = TR_Conv(in_dim, out_dim, m=m, n=n, groups=groups)
        self.upChannel2 = TR_Conv(in_dim, out_dim, m=m, n=n, groups=groups)
        self.upChannel3 = TR_Conv(in_dim, out_dim, m=m, n=n, groups=groups)
        self.upChannel4 = TR_Conv(in_dim, out_dim, m=m, n=n, groups=groups)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.bn4 = nn.BatchNorm1d(out_dim)

# NLLlr/LR  +/-
    def forward(self, x):
        x1 = self.upChannel1(x)
        x1 = self.relu(x1)
        x1 = self.bn1(x1)
        # x1 = self.dropout(x1)

        x2 = self.upChannel2(x)
        x2 = self.relu(x2)
        x2 = self.bn2(x2)
        # x2 = self.dropout(x2)

        x3 = self.upChannel3(x)
        x3 = self.relu(x3)
        x3 = self.bn3(x3)
        # x3 = self.dropout(x3)

        x4 = self.upChannel4(x)
        x4 = self.relu(x4)
        x4 = self.bn4(x4)

        return x1, x2, x3, x4

    def load_teacher_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        # exclude =  ['speaker_encoder.torchfbank', 'speaker_encoder.fc6', 'speaker_encoder.bn6']
        exclude = ['speaker_encoder.torchfbank']
        for name, param in loaded_state.items():
            # if any(name.startswith(prefix) for prefix in exclude):
            #     continue  # 如果名字以任何一个前缀开头，则跳过
            if 'speaker_encoder' in name:
                key = name.split('.', 1)[1]  # 去掉 'speaker_encoder.' 前缀
                if key in self_state:
                    self_state[key].copy_(param)
                else:
                    print(f"Warning: {key} not found in state_dict.")

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
        self_state[name].copy_(param)

class UpChannel_lr_pn_PConv(nn.Module):

    def __init__(self, in_dim, out_dim, m=5, n=0, groups=1, concatenate=False, attention=False, head_paramisloate=False):
        super().__init__()
        self.upChannel1 = TR_Conv(in_dim, out_dim, m=m, n=n, groups=groups, concatenate=concatenate, attention=attention, head_paramisloate=head_paramisloate)
        self.upChannel2 = TR_Conv(in_dim, out_dim, m=m, n=n, groups=groups, concatenate=concatenate, attention=attention, head_paramisloate=head_paramisloate)
        self.upChannel3 = TR_Conv(in_dim, out_dim, m=m, n=n, groups=groups, concatenate=concatenate, attention=attention, head_paramisloate=head_paramisloate)
        self.upChannel4 = TR_Conv(in_dim, out_dim, m=m, n=n, groups=groups, concatenate=concatenate, attention=attention, head_paramisloate=head_paramisloate)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.pReLU_up1 = nn.PReLU()
        self.pReLU_up2 = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.bn4 = nn.BatchNorm1d(out_dim)

# NLLlr/LR  +/-
    def forward(self, x):
        x1 = self.relu(x)
        x1 = self.upChannel1(x1 + x)
        x1 = self.elu(x1)
        x1 = self.bn1(x1)
        # x1 = self.dropout(x1)

        x2 = self.relu(-x)
        x2 = self.upChannel2(x2 - x)
        x2 = self.elu(x2)
        x2 = self.bn2(x2)
        # x2 = self.dropout(x2)

        x3 = self.pReLU_up1(x)
        x3 = self.upChannel3(x3 + x)
        x3 = self.elu(x3)
        x3 = self.bn3(x3)
        # x3 = self.dropout(x3)

        x4 = self.pReLU_up2(-x)
        x4 = self.upChannel4(x4 - x)
        x4 = self.elu(x4)
        x4 = self.bn4(x4)

        return x1, x2, x3, x4

    def load_teacher_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        # exclude =  ['speaker_encoder.torchfbank', 'speaker_encoder.fc6', 'speaker_encoder.bn6']
        exclude = ['speaker_encoder.torchfbank']
        for name, param in loaded_state.items():
            # if any(name.startswith(prefix) for prefix in exclude):
            #     continue  # 如果名字以任何一个前缀开头，则跳过
            if 'speaker_encoder' in name:
                key = name.split('.', 1)[1]  # 去掉 'speaker_encoder.' 前缀
                if key in self_state:
                    self_state[key].copy_(param)
                else:
                    print(f"Warning: {key} not found in state_dict.")

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
        self_state[name].copy_(param)

class New_UpChannel_lr_pn_PConv(nn.Module):

    def __init__(self, in_dim, out_dim, m=5, n=0, groups=1, concatenate=False, attention=False, head_paramisloate=False):
        super().__init__()
        self.upChannel1 = TR_Conv(in_dim, out_dim, m=m, n=n, groups=groups, concatenate=concatenate, attention=attention, head_paramisloate=head_paramisloate)
        self.upChannel2 = TR_Conv(in_dim, out_dim, m=m, n=n, groups=groups, concatenate=concatenate, attention=attention, head_paramisloate=head_paramisloate)
        self.upChannel3 = TR_Conv(in_dim, out_dim, m=m, n=n, groups=groups, concatenate=concatenate, attention=attention, head_paramisloate=head_paramisloate)
        self.upChannel4 = TR_Conv(in_dim, out_dim, m=m, n=n, groups=groups, concatenate=concatenate, attention=attention, head_paramisloate=head_paramisloate)
        self.relu = nn.ReLU()
        self.gelu = QuickGELU()
        self.pReLU_up1 = nn.PReLU(init=0.998)
        self.pReLU_up2 = nn.PReLU(init=0.998)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.bn4 = nn.BatchNorm1d(out_dim)

# NLLlr/LR  +/-
    def forward(self, x):
        # 均表示距离 2:1 2：1.5动态 时序上处理数据
        x1 = 1e-3 * self.relu(x)
        x1 = self.upChannel1(x1 + x)
        x1 = self.gelu(x1)
        x1 = self.bn1(x1)
        # x1 = self.dropout(x1)

        x2 = 1e-3 * self.relu(-x)
        x2 = self.upChannel2(x2 - x)
        x2 = self.gelu(x2)
        x2 = self.bn2(x2)
        # x2 = self.dropout(x2)

        x3 = self.pReLU_up1(x)
        x3 = self.upChannel3(1/2 * (x3 + x))
        x3 = self.gelu(x3)
        x3 = self.bn3(x3)
        # x3 = self.dropout(x3)

        x4 = self.pReLU_up2(-x)
        x4 = self.upChannel4(1/2 * (x4 - x))
        x4 = self.gelu(x4)
        x4 = self.bn4(x4)

        #  获取全局信息 变回 正负
        B = x4.shape[0]
        T = x4.shape[-1]
        # 交叉拼接
        x5 = torch.stack((x1, x2), dim=-2).view(B, -1, T)  # no lR 正负在一个量纲  +*2  -*1
        x6 = torch.stack((x3, x4), dim=-2).view(B, -1, T)  # no lR 正负在一个量纲  +*2  -*1
        x7 = torch.stack((x1, x3), dim=-2).view(B, -1, T)  # no lR 正负在一个量纲  +*2  -*1
        x8 = torch.stack((x2, x4), dim=-2).view(B, -1, T)
 
        # 直接拼接
        # x5 = torch.cat((x1, -x2), dim=1)  # no lR 正负在一个量纲  +*2  -*1
        # x6 = torch.cat((x3, -x4), dim=1)  # LR 正负在一个量纲  +*2  -*微调1
        # x7 = torch.cat((x1, x3), dim=1)  # 正负分布不变 +*2    - ：一个不变*1一个*微调1
        # x8 = torch.cat((-x2, -x4), dim=1)  # 正负分布不变 -*2   + ：一个不变*1一个*微调1
        return x5, x6, x7, x8

    def load_teacher_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        # exclude =  ['speaker_encoder.torchfbank', 'speaker_encoder.fc6', 'speaker_encoder.bn6']
        exclude = ['speaker_encoder.torchfbank']
        for name, param in loaded_state.items():
            # if any(name.startswith(prefix) for prefix in exclude):
            #     continue  # 如果名字以任何一个前缀开头，则跳过
            if 'speaker_encoder' in name:
                key = name.split('.', 1)[1]  # 去掉 'speaker_encoder.' 前缀
                if key in self_state:
                    self_state[key].copy_(param)
                else:
                    print(f"Warning: {key} not found in state_dict.")

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
        self_state[name].copy_(param)


# Bottle2neck 定义  DGCN/TR_Conv
class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super().__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.se(out)
        out += residual
        return out
class New_Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, m=3, n=1, dilation=None, scale=8, concatenate=False, attention=False, head_paramisloate=False):
        super().__init__()
        assert inplanes % scale == 0
        assert planes % scale == 0
        width = int(math.floor(planes / scale))
        self.conv1 = TR_Conv(inplanes, width * scale, m=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(TR_Conv(width, width, m=m, dilation=dilation, n=n, concatenate=concatenate, attention=attention, head_paramisloate=head_paramisloate))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        # self.conv3 = TR_Conv(width * scale, planes, m=1)
        # self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.conv3 = TR_Conv(width * scale, planes, m=1, n=0)
        self.bn3 = nn.BatchNorm1d(width * scale)
        self.width = width
        self.se = New_SEModule(channels = planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        # out = torch.sum(out, dim=-2)
        # out = torch.max(out, dim=-2)[0]
        out += residual
        return out

# 网络结构(BASE: ECAPA) 单个： 两个： 定义   需要跑的实验
class Ecapa_Tdnn_Model(nn.Module):

    def __init__(self, C, out_dim=192, in_dim=80, **kwargs):
        super().__init__()

        # self.wav_pre_process = wav_pre_process_melfbank()

        self.conv1 = nn.Conv1d(in_dim, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        # hidden dim 256  para number = 14.7291MB C 1024
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            # nn.Tanh(),  # I add this layer
            nn.Conv1d(128, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, out_dim)
        self.bn6 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        # x = self.wav_pre_process(x, aug=aug)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                              torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)
        return x,

    def load_teacher_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        # exclude =  ['speaker_encoder.torchfbank', 'speaker_encoder.fc6', 'speaker_encoder.bn6']
        # exclude = ['speaker_encoder.torchfbank']
        for name, param in loaded_state.items():
            # if any(name.startswith(prefix) for prefix in exclude):
            #     continue  # 如果名字以任何一个前缀开头，则跳过
            if 'speaker_encoder' in name:
                key = name.split('.', 1)[1]  # 去掉 'speaker_encoder.' 前缀
                if key in self_state:
                    self_state[key].copy_(param)
                else:
                    print(f"Warning: {key} not found in teacher state_dict.")

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)

class New_LF_Block(nn.Module):

    def __init__(self, m=5, n=1, in_dim=80, hid_dim=80, out_dim=512):
        super().__init__()
        # self.relu = nn.ReLU()
        self.LFB_layer1 = New_Ponv2D(m=m, n=n, in_dim=in_dim, out_dim=out_dim//4)
        self.LFB_layer2 = New_Ponv2D(m=m, n=n+1, in_dim=in_dim, out_dim=out_dim//4)
        self.LFB_layer3 = New_Ponv2D(m=m, n=n+2, in_dim=in_dim, out_dim=out_dim//4)
        self.LFB_layer4 = New_Ponv2D(m=m, n=n+3, in_dim=in_dim, out_dim=out_dim//4)
        self.LFB_bn = nn.BatchNorm1d(out_dim)
        self.LFB_se = New_SEModule(out_dim)
        # self.LFB_relu = nn.ReLU()
    def forward(self, x):
        x1 = self.LFB_layer1(x)
        x2 = self.LFB_layer2(x)
        x3 = self.LFB_layer3(x)
        x4 = self.LFB_layer4(x)
        x = self.LFB_bn(torch.cat((x1, x2, x3, x4), dim=1))
        x = self.LFB_se(x)
        return x

class New_STFT_LF_Block(nn.Module):

    def __init__(self, in_dim=256, m=3):
        super().__init__()
        self.relu = nn.ReLU()
        self.layer1_2d = New_LF_Block(m=m, in_dim=in_dim, freq_fuse_num=256)
        self.layer2_2d = New_LF_Block(m=m, in_dim=in_dim, freq_fuse_num=128)
        self.layer3_2d = New_LF_Block(m=m, in_dim=in_dim, freq_fuse_num=64)
        self.layer4_2d = New_LF_Block(m=m, in_dim=in_dim, freq_fuse_num=32)
        self.layer5_2d = New_LF_Block(m=m, in_dim=in_dim, freq_fuse_num=16)
        self.connect_layer = TR_Conv(m=2, n=4, in_channels=in_dim, out_channels=192*16, concatenate=True)
    def forward(self, x):
        x = self.layer1_2d(x)
        x = self.layer2_2d(x)
        x = self.layer3_2d(x)
        x = self.connect_layer(x).view(x.shape[0], -1, x.shape[-1])
        return x

# 基于stft的架构
# class New_Ecapa_Tdnn_Model(nn.Module):
#
#     def __init__(self, C, out_dim=192, in_dim=in_dim, **kwargs):
#         super().__init__()
#         mn = 1
#         head_paramisloate = False  # only head_paramisloate = True n = 2 独立 n = 1 共享  称为simple  add
#         concatenate = True  # only concatenate = True n = 2 独立 n = 1 共享 需要多头来支撑
#         attention = False  #
#         n = 2  # m
#
#         scale = 8
#         # self.wav_pre_process = wav_pre_process_melfbank()
#         self.upChannel = New_STFT_LF_Block(in_dim=in_dim, m=3)
#         # self.upChannel = TR_Conv(in_dim, C, m=5, n=n, concatenate=concatenate, attention=attention, head_paramisloate=head_paramisloate)
#         # self.upChannel = New_UpChannel_lr_pn_PConv(80, C//2, m=5, n=n, concatenate=concatenate, attention=attention, head_paramisloate=head_paramisloate)
#
#         self.relu = nn.ReLU()
#         self.bn1 = nn.BatchNorm1d(3072)
#         # self.layer1 = New_Bottle2neck(C, C, m=3, dilation=2, scale=scale, n=n, concatenate=concatenate, attention=attention, head_paramisloate=head_paramisloate)
#         # self.layer2 = New_Bottle2neck(C, C, m=3, dilation=3, scale=scale, n=n, concatenate=concatenate, attention=attention, head_paramisloate=head_paramisloate)
#         # self.layer3 = New_Bottle2neck(C, C, m=3, dilation=4, scale=scale, n=n, concatenate=concatenate, attention=attention, head_paramisloate=head_paramisloate)
#         # self.layer4 = New_Bottle2neck(C, C, m=3, dilation=5, scale=scale, n=n, concatenate=concatenate, attention=attention, head_paramisloate=head_paramisloate)
#         # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
#
#         self.layer5 = TR_Conv(3072, 1024,  m=1, n=1)
#         self.bn5 = nn.BatchNorm1d(1024)
#         self.layer6 = TR_Conv(1024, 1024,  m=1, n=1)
#         self.bn6 = nn.BatchNorm1d(1024)
#
#         self.layer7 = TR_Conv(1024, 1024,  m=3, n=1)
#         self.bn7 = nn.BatchNorm1d(1024)
#
#         self.layer8 = TR_Conv(1024, 1024,  m=1, n=1)
#         self.bn8 = nn.BatchNorm1d(1024)
#         self.se = SEModule(1024)
#         self.layer9 = TR_Conv(1024, 1536,  m=1, n=1)
#
#         # self.layer5 = nn.Sequential(
#         #     TR_Conv(4 * C, 128,  m=mn, n=mn-1),
#         #     nn.ReLU(),
#         #     nn.BatchNorm1d(128),
#         #     nn.Tanh(),  # I add this layer
#         #     TR_Conv(128, 1536,  m=mn, n=mn-1),
#         # )
#         self.attention = nn.Sequential(
#             TR_Conv(4608, 128,  m=1, n=0),
#             nn.ReLU(),
#             nn.BatchNorm1d(128),
#             nn.Tanh(),  # I add this layer
#             TR_Conv(128, 1536,  m=1, n=0),
#             nn.Softmax(dim=2),
#         )
#         self.bn10 = nn.BatchNorm1d(3072)
#         # self.drop = nn.Dropout(0.25)
#         self.fc11 = nn.Linear(3072, dim)
#         # self.fc6 = TR_Conv(3072, dim, m=mn, n=mn-1)
#         self.bn11 = nn.BatchNorm1d(dim)
#
#     def forward(self, x):
#         # x = self.wav_pre_process(x, aug=aug)
#         x = self.upChannel(x)
#         x = self.relu(x)
#         x = self.bn1(x)
#
#         x = self.layer5(x)
#         x = self.relu(x)
#         x = self.bn5(x)
#         x = self.layer6(x)
#         x = self.relu(x)
#         x = self.bn6(x)
#
#         x = self.layer7(x)
#         x = self.relu(x)
#         x = self.bn7(x)
#
#         x = self.layer8(x)
#         x = self.relu(x)
#         x = self.bn8(x)
#         x = self.se(x)
#
#         x = self.layer9(x)
#         x = self.relu(x)
#
#         t = x.size()[-1]
#
#         global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
#                               torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)
#
#         w = self.attention(global_x)
#
#         mu = torch.sum(x * w, dim=2)
#         sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
#
#         x = torch.cat((mu, sg), 1)
#         x = self.bn10(x)
#         x = self.fc11(x)
#         x = self.bn11(x)
#
#         # x1, x2, x3, x4 = self.upChannel(x)
#         # x5 = self.layer1(x1)
#         # x6 = self.layer2(x2)
#         # x7 = self.layer3(x3)
#         # x8 = self.layer4(x4)
#         # x = self.layer5(torch.cat((x5, x6, x7, x8), dim=1))
#         # x = self.relu(x)
#         # global_m5 = torch.cat((x1, x2, x3, x4), dim=1)
#         # w = self.attention(global_m5)
#         # mu = torch.sum(x * w, dim=2, keepdim=True)
#         # sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2, keepdim = True) - mu ** 2).clamp(min=1e-12))
#         #
#         # # mu = torch.sum(x, dim=2)
#         # # sg = torch.sqrt((torch.sum((x ** 2), dim=2) - mu ** 2).clamp(min=1e-12))
#         #
#         # x = torch.cat((mu, sg), 1)[:, :, 0]
#         # x = self.bn5(x)
#         # # x = self.drop(x)
#         # x = self.fc6(x)
#         # # x = self.fc6(x)[:, :, 0]
#
#         # x = self.bn6(x)
#         return x,
#
#     def load_teacher_parameters(self, path):
#         self_state = self.state_dict()
#         loaded_state = torch.load(path)
#         # exclude =  ['speaker_encoder.torchfbank', 'speaker_encoder.fc6', 'speaker_encoder.bn6']
#         # exclude = ['speaker_encoder.torchfbank']
#         for name, param in loaded_state.items():
#             # if any(name.startswith(prefix) for prefix in exclude):
#             #     continue  # 如果名字以任何一个前缀开头，则跳过
#             if 'speaker_encoder' in name:
#                 key = name.split('.', 1)[1]  # 去掉 'speaker_encoder.' 前缀
#                 if key in self_state:
#                     self_state[key].copy_(param)
#                 else:
#                     print(f"Warning: {key} not found in state_dict.")
#
#     def load_parameters(self, path):
#         self_state = self.state_dict()
#         loaded_state = torch.load(path)
#         for name, param in loaded_state.items():
#             origname = name
#             if name not in self_state:
#                 name = name.replace("module.", "")
#                 if name not in self_state:
#                     print("%s is not in the model." % origname)
#                     continue
#             if self_state[name].size() != loaded_state[origname].size():
#                 print("Wrong parameter length: %s, model: %s, loaded: %s" % (
#                     origname, self_state[name].size(), loaded_state[origname].size()))
#                 continue
#             self_state[name].copy_(param)
class New_Ecapa_Tdnn_Model(nn.Module):

    def __init__(self, C, out_dim=192, in_dim=80, n=1, **kwargs):
        super().__init__()
        # mn = 1
        head_paramisloate = False  # only head_paramisloate = True n = 2 独立 n = 1 共享  称为simple  add
        concatenate = True  # only concatenate = True n = 2 独立 n = 1 共享 需要多头来支撑
        attention = False  #
        n = n  # m
        scale = 8
        # self.upChannel = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        # self.upChannel = New_LF_Block(m=5, n=n, in_dim=in_dim, out_dim=C)
        # self.upChannel1 = New_LF_Block(m=3, n=n, in_dim=C, out_dim=C)
        # self.upChannel2 = New_LF_Block(m=3, n=n, in_dim=C, out_dim=C)
        self.upChannel = TR_Conv(in_dim, C, m=5, n=n, concatenate=concatenate, attention=attention, head_paramisloate=head_paramisloate)
        # self.upChannel = New_UpChannel_lr_pn_PConv(80, C//2, m=5, n=n, concatenate=concatenate, attention=attention, head_paramisloate=head_paramisloate)

        self.relu = nn.ReLU()
        self.bn1_1 = nn.BatchNorm1d(C)
        # self.bn1_2 = nn.BatchNorm1d(C)
        # self.bn1_3 = nn.BatchNorm1d(C)
        self.layer1 = New_Bottle2neck(C, C, m=3, dilation=2, scale=scale, n=n, concatenate=concatenate, attention=attention, head_paramisloate=head_paramisloate)
        self.layer2 = New_Bottle2neck(C, C, m=3, dilation=3, scale=scale, n=n, concatenate=concatenate, attention=attention, head_paramisloate=head_paramisloate)
        self.layer3 = New_Bottle2neck(C, C, m=3, dilation=4, scale=scale, n=n, concatenate=concatenate, attention=attention, head_paramisloate=head_paramisloate)
        # self.layer4 = New_Bottle2neck(C, C, m=3, dilation=5, scale=scale, n=n, concatenate=concatenate, attention=attention, head_paramisloate=head_paramisloate)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer5 = TR_Conv(3 * C, 1536,  m=1, n=3, concatenate=concatenate, wide_erf=True)

        # downm2n4_2
        # self.layer5 = nn.Conv1d(3 * C, 1536,  kernel_size=1)
        # self.layer5 = nn.Sequential(
        #     TR_Conv(4 * C, 128,  m=mn, n=mn-1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(128),
        #     nn.Tanh(),  # I add this layer
        #     TR_Conv(128, 1536,  m=mn, n=mn-1),
        # )
        self.attention = nn.Sequential(
            TR_Conv(4608, 128,  m=1, n=0),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Tanh(),  # I add this layer
            TR_Conv(128, 1536,  m=1, n=0),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        # self.drop = nn.Dropout(0.25)
        self.fc6 = nn.Linear(3072, out_dim)
        # self.fc6 = TR_Conv(3072, dim, m=mn, n=mn-1)
        self.bn6 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        # x = self.wav_pre_process(x, aug=aug)
        x = self.upChannel(x)
        x = self.relu(x)
        x = self.bn1_1(x)

        # x11 = self.upChannel1(x)
        # x11 = self.relu(x11)
        # x11 = self.bn1_2(x11)
        #
        # x = self.upChannel2(x11+x)
        # x = self.relu(x)
        # x = self.bn1_3(x)

        # x_2 = self.upChannel2(x)
        # x_2 = self.relu(x_2)
        # x_2 = self.bn1_2(x_2)
        #
        # x_3 = self.upChannel3(x)
        # x_3 = self.relu(x_3)
        # x_3 = self.bn1_3(x_3)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer5(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                              torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        # x1, x2, x3, x4 = self.upChannel(x)
        # x5 = self.layer1(x1)
        # x6 = self.layer2(x2)
        # x7 = self.layer3(x3)
        # x8 = self.layer4(x4)
        # x = self.layer5(torch.cat((x5, x6, x7, x8), dim=1))
        # x = self.relu(x)
        # global_m5 = torch.cat((x1, x2, x3, x4), dim=1)
        # w = self.attention(global_m5)
        # mu = torch.sum(x * w, dim=2, keepdim=True)
        # sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2, keepdim = True) - mu ** 2).clamp(min=1e-12))
        #
        # # mu = torch.sum(x, dim=2)
        # # sg = torch.sqrt((torch.sum((x ** 2), dim=2) - mu ** 2).clamp(min=1e-12))
        #
        # x = torch.cat((mu, sg), 1)[:, :, 0]
        # x = self.bn5(x)
        # # x = self.drop(x)
        # x = self.fc6(x)
        # # x = self.fc6(x)[:, :, 0]

        # x = self.bn6(x)
        return x,

    def load_teacher_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        # exclude =  ['speaker_encoder.torchfbank', 'speaker_encoder.fc6', 'speaker_encoder.bn6']
        # exclude = ['speaker_encoder.torchfbank']
        for name, param in loaded_state.items():
            # if any(name.startswith(prefix) for prefix in exclude):
            #     continue  # 如果名字以任何一个前缀开头，则跳过
            if 'speaker_encoder' in name:
                key = name.split('.', 1)[1]  # 去掉 'speaker_encoder.' 前缀
                if key in self_state:
                    self_state[key].copy_(param)
                else:
                    print(f"Warning: {key} not found in state_dict.")

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)

# (base: 一层bottleneck {dilation:2}) 两个网络(updown and mid) 三个损失函数(AAM Center Neg)处理
class PConv_DGCN_Double_Model(nn.Module):

    def __init__(self, in_dim=80, C=1024, out_dim=192, n=1):
        super().__init__()
        self.scale = 8
        head_paramisloate = False  # only head_paramisloate = True n = 2 独立 n = 1 共享  称为simple  add
        concatenate = True  # only concatenate = True n = 2 独立 n = 1 共享 需要多头来支撑
        attention = False  # only attention = True n = 2 独立 n = 1 共享
        n = 1


        self.relu = nn.ReLU()

        self.upChannel = UpChannel_lr_pn_PConv(in_dim, C//2, n=n,  concatenate=concatenate, attention=attention,
                        head_paramisloate=head_paramisloate)
        self.layer1_1 = New_Bottle2neck(C, C, m=3, dilation=1, scale=8, n=n, concatenate=concatenate, attention=attention,
                        head_paramisloate=head_paramisloate)
        self.layer2_1 = New_Bottle2neck(C, C, m=3, dilation=2, scale=8, n=n, concatenate=concatenate, attention=attention,
                        head_paramisloate=head_paramisloate)
        self.layer3_1 = New_Bottle2neck(C, C, m=3, dilation=3, scale=8, n=n, concatenate=concatenate, attention=attention,
                        head_paramisloate=head_paramisloate)
        self.layer4_1 = New_Bottle2neck(C, C, m=3, dilation=4, scale=8, n=n, concatenate=concatenate, attention=attention,
                        head_paramisloate=head_paramisloate)

        self.layer_eight_up = nn.Sequential(
            TR_Conv(C * 2, 128, m=2, n=n),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            # nn.Dropout(dropout),
            nn.Tanh(),  # I add this layer
            TR_Conv(128, C // 2, m=2, n=n),
            nn.BatchNorm1d(C // 2),
        )
        self.layer_eight_down = nn.Sequential(
            TR_Conv(C * 2, 128, m=2, n=n),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            # nn.Dropout(dropout),
            nn.Tanh(),  # I add this layer
            TR_Conv(128, C // 2, m=2, n=n),
            nn.BatchNorm1d(C // 2),
        )
        self.layer_eight_mid = nn.Sequential(
            TR_Conv(C * 4, 128, m=2, n=n),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            # nn.Dropout(dropout),
            nn.Tanh(),  # I add this layer
            TR_Conv(128, C, m=2, n=n),
            nn.BatchNorm1d(C),
        )
        self.bn1_updown = nn.BatchNorm1d(C * 2)
        self.fc_updown = nn.Linear(C * 2, out_dim)
        self.bn2_updown = nn.BatchNorm1d(out_dim)
        self.bn1_mid = nn.BatchNorm1d(C * 2)
        self.fc_mid = nn.Linear(C * 2, out_dim)
        self.bn2_mid = nn.BatchNorm1d(out_dim)
        # self.softmaxOut = nn.Softmax(dim=1)

    def forward(self, x):
        x1, x2, x3, x4 = self.upChannel(x)
        x5 = torch.cat((x1, -x2), dim=1)  # no lR 正负在一个量纲  +*2  -*1
        x6 = torch.cat((x3, -x4), dim=1)  # LR 正负在一个量纲  +*2  -*微调1
        x7 = torch.cat((x1, x3), dim=1)  # 正负分布不变 +*2    - ：一个不变*1一个*微调1
        x8 = torch.cat((-x2, -x4), dim=1)  # 正负分布不变 -*2   + ：一个不变*1一个*微调1

        # multidata   +-
        x5, x6 = self.layer1_1(x5), self.layer2_1(x6)
        x7, x8 = self.layer3_1(x7), self.layer4_1(x8)

        x_up = torch.cat((x5, x7), dim=1)
        x_down = torch.cat((x6, x8), dim=1)
        x_mid = torch.cat((x5, x6, x7, x8), dim=1)  # 8 的中心点  S S反 合成 8
        x_up = self.layer_eight_up(x_up)
        x_up = self.relu(x_up)
        x_down = self.layer_eight_down(x_down)
        x_down = self.relu(x_down)
        x_mid = self.layer_eight_mid(x_mid)
        x_mid = self.relu(x_mid)
        x_updown = torch.cat((x_up, x_down), dim=1)

        mu_updown = torch.sum(x_updown, dim=2)
        mu_mid = torch.sum(x_mid, dim=2)
        sg_updown = torch.sqrt((torch.sum((x_updown ** 2), dim=2) - mu_updown ** 2).clamp(min=1e-4))
        sg_mid = torch.sqrt((torch.sum((x_mid ** 2), dim=2) - mu_mid ** 2).clamp(min=1e-4))

        x_updown_cat = self.bn1_updown(torch.cat((mu_updown, sg_updown), dim=1))
        x_updown_cat = self.fc_updown(x_updown_cat)
        x_updown_cat = self.bn2_updown(x_updown_cat)

        x_mid_cat = self.bn1_mid(torch.cat((mu_mid, sg_mid), dim=1))
        x_mid_cat = self.fc_mid(x_mid_cat)
        x_mid_cat = self.bn2_mid(x_mid_cat)
        # 仅仅用于算损失
        x_cat = torch.cat((x_updown_cat, x_mid_cat), dim=1)

        return x_updown_cat, x_mid_cat, x_cat, x_updown_cat * x_mid_cat, x_updown_cat - x_mid_cat, x_updown_cat + x_mid_cat

class New_PConv_DGCN_Double_Model(nn.Module):

    def __init__(self, in_dim=80, C=1024, out_dim=192, n=1):
        super().__init__()
        self.scale = 8
        head_paramisloate = False
        concatenate = True
        attention = False
        n = 1

        self.relu = nn.ReLU()

        self.upChannel = New_UpChannel_lr_pn_PConv(80, C//2, m=5, n=n, concatenate=concatenate, attention=attention, head_paramisloate=head_paramisloate)

        self.layer1 = New_Bottle2neck(C, C, m=3, dilation=1, scale=1, n=n, concatenate=concatenate, attention=attention,
                        head_paramisloate=head_paramisloate)
        self.layer1_1 = New_Bottle2neck(C, C, m=3, dilation=1, scale=1, n=n, concatenate=concatenate, attention=attention,
                                      head_paramisloate=head_paramisloate)
        self.layer2 = New_Bottle2neck(C, C, m=3, dilation=2, scale=2, n=n, concatenate=concatenate, attention=attention,
                        head_paramisloate=head_paramisloate)

        self.layer3 = New_Bottle2neck(C, C, m=3, dilation=3, scale=4, n=n, concatenate=concatenate, attention=attention,
                        head_paramisloate=head_paramisloate)
        self.layer4 = New_Bottle2neck(C, C, m=3, dilation=4, scale=8, n=n, concatenate=concatenate, attention=attention,
                        head_paramisloate=head_paramisloate)

        self.layer_eight_up = TR_Conv(2 * C, 1536//2,  m=1, n=3)
        self.layer_eight_down = TR_Conv(2 * C, 1536,  m=1, n=3)
        self.layer_eight_mid = TR_Conv(3 * C, 1536//2, m=1, n=3)
        self.bn1_updown = nn.BatchNorm1d(C * 2)
        self.fc_updown = nn.Linear(C * 2, out_dim)
        self.bn2_updown = nn.BatchNorm1d(out_dim)
        self.bn1_mid = nn.BatchNorm1d(C * 2)
        self.fc_mid = nn.Linear(C * 2, out_dim)
        self.bn2_mid = nn.BatchNorm1d(out_dim)
        # self.softmaxOut = nn.Softmax(dim=1)

    def forward(self, x):
        x1, x2, x3, x4 = self.upChannel(x)
        x5 = torch.cat((x1, -x2), dim=1)  # no lR 正负在一个量纲  +*2  -*1
        x6 = torch.cat((x3, -x4), dim=1)  # LR 正负在一个量纲  +*2  -*微调1
        x7 = torch.cat((x1, x3), dim=1)  # 正负分布不变 +*2    - ：一个不变*1一个*微调1
        x8 = torch.cat((-x2, -x4), dim=1)  # 正负分布不变 -*2   + ：一个不变*1一个*微调1

        # multidata   +-
        x5, x6 = self.layer1_1(x5), self.layer2_1(x6)
        x7, x8 = self.layer3_1(x7), self.layer4_1(x8)

        x_up = torch.cat((x5, x7), dim=1)
        x_down = torch.cat((x6, x8), dim=1)
        x_mid = torch.cat((x5, x6, x7, x8), dim=1)  # 8 的中心点  S S反 合成 8
        x_up = self.layer_eight_up(x_up)
        x_up = self.relu(x_up)
        x_down = self.layer_eight_down(x_down)
        x_down = self.relu(x_down)
        x_mid = self.layer_eight_mid(x_mid)
        x_mid = self.relu(x_mid)
        x_updown = torch.cat((x_up, x_down), dim=1)

        mu_updown = torch.sum(x_updown, dim=2)
        mu_mid = torch.sum(x_mid, dim=2)
        sg_updown = torch.sqrt((torch.sum((x_updown ** 2), dim=2) - mu_updown ** 2).clamp(min=1e-4))
        sg_mid = torch.sqrt((torch.sum((x_mid ** 2), dim=2) - mu_mid ** 2).clamp(min=1e-4))

        x_updown_cat = self.bn1_updown(torch.cat((mu_updown, sg_updown), dim=1))
        x_updown_cat = self.fc_updown(x_updown_cat)
        x_updown_cat = self.bn2_updown(x_updown_cat)

        x_mid_cat = self.bn1_mid(torch.cat((mu_mid, sg_mid), dim=1))
        x_mid_cat = self.fc_mid(x_mid_cat)
        x_mid_cat = self.bn2_mid(x_mid_cat)
        # 仅仅用于算损失
        x_cat = torch.cat((x_updown_cat, x_mid_cat), dim=1)

        return x_updown_cat, x_mid_cat, x_cat, x_updown_cat * x_mid_cat, x_updown_cat - x_mid_cat, x_updown_cat + x_mid_cat
class Free_Unified_Meta_Model(nn.Module):

    def __init__(self, in_dim=80, C=512, out_dim=192, dropout=0.3):
        super().__init__()
        self.layer1 = TR_Conv(in_channels=80, out_channels=C, m=5, n=4, attention=True, head_paramisloate=True, concatenate=True)
        self.layer2 = TR_Conv(in_channels=C*3, out_channels=C, m=1, n=0)
        self.layer3 = TR_Conv(in_channels=C, out_channels=C, m=3, n=2, attention=True, head_paramisloate=True, concatenate=True)
        self.layer4 = TR_Conv(in_channels=C*2, out_channels=3*C, m=1, n=0)

        self.bn1 = nn.BatchNorm1d(C * 3)
        self.bn2 = nn.BatchNorm1d(C * 3)
        self.bn3 = nn.BatchNorm1d(C * 3)

        self.gelu = QuickGELU()
        self.bn4 = nn.BatchNorm1d(C * 3)
        self.fc = nn.Linear(C * 3, out_dim)
        self.bn5 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.gelu(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.gelu(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.fc(x)
        x = self.bn5(x)
        return x, x, x,