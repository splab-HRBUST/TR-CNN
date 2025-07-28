import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
                x_zero_x_dila_input = torch.zeros(B, C, self.unit_FZ * (T - 1) + 1, device=x.device)
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
                    else:  # 共享 相当于没加算子
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
                            taylor_eq_list.append(
                                depth_conv_add + 1 / (self.head) * depth_conv_m + 1 / (self.head * 2) * self.drop(
                                    x_m_mk_unfold_m))
                        else:
                            taylor_eq_list.append(
                                depth_conv_minus + 1 / (self.head) * depth_conv_m + 1 / (self.head * 2))
                            taylor_eq_list.append(
                                depth_conv_add + 1 / (self.head) * depth_conv_m + 1 / (self.head * 2))
                # 对于 中间的taylor_eq_list 进行处理
                if self.taylor:
                    for i in range(len(taylor_eq_list)):
                        taylor_eq_list[i] = taylor_eq_list[i] / math.factorial(i + 1)
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

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)