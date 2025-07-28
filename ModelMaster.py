'''
This part is used to train the speaker model and evaluate the performances
'''

import sys
import time

import math, pandas as pd
import numpy as np
import torch.optim
import torchaudio
import tqdm
from model import wav_pre_process_fbank, wav_pre_process_stft, Ecapa_Tdnn_Model, New_Ecapa_Tdnn_Model, PConv_DGCN_Double_Model
from loss import AAMsoftmax, Control_Contrastive, CircleLoss, MSELoss, KLDivLoss
from tools import *
# from torchsummary import summary


class ModelMaster(nn.Module):
    def __init__(self, lr, lr_decay, C, n_class, dim, m, s, n=1, truncate_epoch=37, **kwargs):
        super(ModelMaster, self).__init__()
        ## ECAPA-TDNN
        self.train_flag = True
        self.truncate_epoch = truncate_epoch
        self.truncate_flag = False
        self.n_class = n_class
        self.in_dim = 80


        # self.wav_pre_process = wav_pre_process_stft().cuda()
        self.wav_pre_process = wav_pre_process_fbank().cuda()
        self.teacher_flag = False
        if kwargs.get('teacher') is not None:
            self.guidance_extrator = kwargs.get('teacher').cuda()
            self.teacher_flag = True
            print("加载指导模型 self.guidance_extrator:{}!!!".format(self.guidance_extrator._get_name()))
        else:
            self.guidance_extrator = None
        # W2V2Config = kwargs.get('W2V2Config')
        # self.wav2Vec2Model = kwargs.get('Wav2Vec2Model').cuda()
        # self.feature_extractor = kwargs.get('feature_extractor')
        # self.WavLMForXVector = kwargs.get('WavLMForXVector').cuda()

        # self.wav2Vec2Model = Wav2Vec2Model(config=W2V2Config).cuda()
        # self.wav2Vec2Model.freeze_feature_encoder()

        # self.GPT_IConv_NN = Ecapa_Tdnn_Model(C=C, out_dim=dim, in_dim=self.in_dim).cuda()
        self.GPT_IConv_NN = New_Ecapa_Tdnn_Model(C=C, out_dim=dim, in_dim=self.in_dim, n=n).cuda()
        # self.GPT_IConv_NN = PConv_DGCN_Double_Model(C=C, out_dim=dim, in_dim=self.in_dim).cuda()
        # self.GPT_IConv_NN = MaskSelect_Model_base(C=C, dim=dim).cuda()
        # self.GPT_IConv_NN = calParam_PConv_DGCN_Double_Model(C=C, out_dim=dim, n=n).cuda()
        # self.GPT_IConv_NN = PConv_DGCN_Double_Model(C=C, out_dim=dim, n=n).cuda()
        # self.GPT_IConv_NN = Free_Unified_Meta_Model(C=C, out_dim=dim).cuda()
        # summary(self.GPT_IConv_NN, [(80, 202)])  # 1429层  2117层
        self.teacher_Contrastive_loss = MSELoss().cuda()
        # self.teacher_Contrastive_loss = KLDivLoss().cuda()
        # self.Control_Contrastive_loss = AAMsoftmax(n_class=n_class, dim=dim, m=m, s=s).cuda()
        self.Control_Contrastive_loss = Control_Contrastive(n_class=n_class, dim=dim, m=m, s=s).cuda()
        # self.Control_Contrastive_loss = CircleLoss(m=0.35, gamma=60).cuda()
        # self.BCELoss    = MSELoss().cuda()
        # self.optim           = torch.optim.SGD(self.parameters(), lr = 0.2, weight_decay = 1e-4, momentum=0.9)
        # self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
        self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-4)
        # self.optim           = torch.optim.Adam(self.GPT_IConv_NN.parameters(), lr = lr, weight_decay = 2e-5)
        self.optimLoss           = torch.optim.Adam(self.Control_Contrastive_loss.parameters(), lr = lr, weight_decay = 2e-5)
        # self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 1e-4)
        # self.scheduler       = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, eta_min = test_step, gamma=lr_decay)
        # self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lr_decay)

        # self.scheduler       = torch.optim.lr_scheduler.CyclicLR(self.optim, base_lr=1e-8, max_lr=1e-3, step_size_up=10*983 * 3 * 2 //2, mode="triangular2", cycle_momentum=False)
        self.scheduler       = torch.optim.lr_scheduler.CyclicLR(self.optim, base_lr=1e-8, max_lr=1e-3, step_size_up=120000//10, mode="triangular2", cycle_momentum=False)

        self.schedulerLoss       = torch.optim.lr_scheduler.CyclicLR(self.optimLoss, base_lr=1e-8, max_lr=1e-3, step_size_up=120000//10, mode="triangular2", cycle_momentum=False)
        # self.schedulerLoss       = torch.optim.lr_scheduler.CyclicLR(self.optim, base_lr=1e-8, max_lr=1e-3, step_size_up=10*983 * 3 * 2 //2, mode="triangular2", cycle_momentum=False)

        # self.schedulerLoss       = torch.optim.lr_scheduler.StepLR(self.optimLoss, step_size = test_step, gamma=lr_decay)
        # print(time.strftime("%m-%d %H:%M:%S") + " ecapa  para number = %.4fMB"%(sum(param.numel() for param in self.ecapa.parameters()) / 1024 / 1024))
        # 冻结
        # self.frozen_needed_parameters(flag= True)
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡")
        print(time.strftime("%m-%d %H:%M:%S") + " GPT_IConv_DGCN  para number = %.4fMB"%(sum(param.numel() for param in self.GPT_IConv_NN.parameters()) / 1024 / 1024))
        # print(time.strftime("%m-%d %H:%M:%S") + " Control_Contrastive_loss  para number = %.4fMB"%(sum(param.numel() for param in self.Control_Contrastive_loss.parameters()) / 1024 / 1024))
        # print(time.strftime("%m-%d %H:%M:%S") + " WavLMForXVector  para number = %.4fMB"%(sum(param.numel() for param in self.WavLMForXVector.parameters()) / 1024 / 1024))
        print(time.strftime("%m-%d %H:%M:%S") + " Total_Model  para number = %.4fMB"%(sum(param.numel() for param in self.parameters()) / 1024 / 1024))
        # 计算模型大小 和 参数计算
        self.macs_params(input_size=(1, self.in_dim, 202), device="cuda:0")
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡\n")
    def forward(self, wav, aug=False, label=None):
        # score_G, score_D, ce = 0, 0, 0
        loss, circle, sv = 0, 0, 0
        predict_embedding = 0, 0
        wav = wav.cuda()
        x = self.wav_pre_process(wav, aug=aug)
        if self.teacher_flag:
            assert (self.guidance_extrator is not None), "self.guidance_extrator is None -- err from yzx"
            with torch.no_grad():
                teacher_tuple = self.guidance_extrator(wav),
        # 	x = x.cpu().numpy()
        # 	x = self.feature_extractor(x, return_tensors="pt", sampling_rate=16000)
        #
        # 	for i in x:
        # 		x[i] = x[i].cuda()
        # 	x = self.WavLMForXVector(**x).emds_dict
        # x  = x.cuda()
            # x = self.WavLMForXVector(**x).hidden_states[-1]
            # logits = self.WavLMForXVector(**x).logits
            # hidden_states_fir = self.WavLMForXVector(**x).hidden_states[0]
            # hidden_states_last = self.WavLMForXVector(**x).hidden_states[-1]
            # hidden_states_fir_channel = torch.cat((torch.mean(hidden_states_fir, dim=2), torch.sqrt(torch.var(hidden_states_fir, dim=2).clamp(min=1e-4))), dim=1)  # B * 200
            # hidden_states_last_channel = torch.cat((torch.mean(hidden_states_last, dim=2), torch.sqrt(torch.var(hidden_states_last, dim=2).clamp(min=1e-4))), dim=1)  # B * 200
            # hidden_states_fir_frame = torch.cat((torch.mean(hidden_states_fir, dim=1), torch.sqrt(torch.var(hidden_states_fir, dim=1).clamp(min=1e-4))), dim=1)  # B * 200
            # hidden_states_last_frame = torch.cat((torch.mean(hidden_states_last, dim=1), torch.sqrt(torch.var(hidden_states_last, dim=1).clamp(min=1e-4))), dim=1)  # B * 200

        # del x
        # emds_dict = torch.nn.functional.normalize(emds_dict, dim=-1).cpu()

        # x, w2v_last_hidden_state = x.last_hidden_state, x.last_hidden_state
        # w2v_last_hidden_state = torch.cat((torch.mean(w2v_last_hidden_state, dim=2), torch.sqrt(torch.var(w2v_last_hidden_state, dim=2).clamp(min=1e-4))), dim=1)  # B * 200
        #  x1: eight_updown   x2: eight_mid

        # with torch.jit.optimized_execution(True):
        #     self.guidance_extrator.half()
        #     teacher_tuple = self.guidance_extrator(wav),

        embedding_tuple = self.GPT_IConv_NN(x)
        if self.train_flag:
            # loss, predict_embedding = self.Control_Contrastive_loss(embedding_tuple[0], label)  # speaker_AAMsoftmax_loss
            sv, predict_embedding = self.Control_Contrastive_loss(embedding_tuple[0], label)  # speaker_AAMsoftmax_loss
            # loss = self.teacher_Contrastive_loss(embedding_tuple[0], teacher_tuple[0])  # speaker_AAMsoftmax_loss


            # predict_embedding = torch.ones(wav.shape[0], self.n_class, device=wav.device, requires_grad=False),
            # sv, prec_embedding, prec_embedding_center = self.Control_Contrastive_loss(embedding_tuple[0], embedding_tuple[1], label)

            # score_G = self.speaker_kl_loss(o_s_fake_G, o_t_real)
            # score_D = self.BCELoss(o_s_fake_G, o_t_real)
            # score_D_2 = self.BCELoss(o_s_fake_D, fake)
            # score_D = 0.7 * score_D_1 + 0.3 * score_D_2
        return sv + loss + circle, ("acc", "acc_center"), predict_embedding, embedding_tuple,  # 返回向量均是1024

    def utt2spk(self, spk, cohort_embds):
        cohort_dict = {}
        for spk_cohort, cohort in cohort_embds.items():
            # 个人定制 yzx 2025年4月4日23:55:07
            spk_cohort = spk_cohort.split('/')[0]  # 处理成 如 id10284
            utter = torch.mean(cohort[1][0], dim=0, keepdim=True)
            if cohort_dict.get(spk_cohort) is None:
                cohort_dict[spk_cohort] = utter
            else:
                cohort_dict[spk_cohort] = torch.cat((cohort_dict[spk_cohort], utter), dim=0)

        # 求均值
        for spk_cohort, utter in cohort_dict.items():
            cohort_dict[spk_cohort] = torch.mean(cohort_dict[spk_cohort], dim=0, keepdim=True)
        return spk, cohort_dict
    def get_cohort_embds(self, spk, cohort_embds):
        spk, cohort_embds = self.utt2spk(spk, cohort_embds)

        # 处理成dict
        # filt_cohort_embds = {k: v for k, v in center_per_spk.items() if k not in [enr, tst]}
        # 单独处理成tensor
        spk_cohort_embds = [v for k, v in cohort_embds.items() if k not in spk]
        filt_cohort_embds = torch.stack(spk_cohort_embds, dim=-2).squeeze()
        assert (len(filt_cohort_embds.shape) == 2), "as normal need 2 dim of spk utt_mean  -- err from yzx"
        return filt_cohort_embds

    # 三种距离 cos(默认)：最好  oushi：也是变差  mahadun:出奇的不好，差的离谱
    def score_as_normal(self, enr_embd, tst_embd, cohort_embds_enr, cohort_embds_tst, score, adaptive_cohort_size=500, mode='cos'):
        # 个性化定制 yzx
        adaptive_cohort_size = min(adaptive_cohort_size, cohort_embds_enr.shape[0])
        if mode is None:
            mode = 'cos'
        if mode=='cos':
            e_c = torch.matmul(enr_embd, cohort_embds_enr.T).mean(0)
        elif mode == 'oushi':
            e_c = -1.0 * torch.cdist(enr_embd, cohort_embds_enr, p=2).mean(0)
        else:  # mahadun
            e_c = -1.0 * torch.cdist(enr_embd, cohort_embds_enr, p=1).mean(0)

        e_c = torch.topk(e_c, adaptive_cohort_size)[0]

        e_c_m = torch.mean(e_c)
        e_c_s = torch.std(e_c)

        if mode=='cos':
            t_c = torch.matmul(tst_embd, cohort_embds_tst.T).mean(0)
        elif mode == 'oushi':
            t_c = -1.0 * torch.cdist(tst_embd, cohort_embds_tst, p=2).mean(0)
        else:  # mahadun
            t_c = -1.0 * torch.cdist(tst_embd, cohort_embds_tst, p=1).mean(0)

        t_c = torch.topk(t_c, k=adaptive_cohort_size)[0]

        t_c_m = torch.mean(t_c)
        t_c_s = torch.std(t_c)

        normscore_e = (score - e_c_m) / e_c_s
        normscore_t = (score - t_c_m) / t_c_s

        newscore = (normscore_e + normscore_t) / 2
        # newscore = newscore.item()   # 不需要直接转化为float格式 注释
        return newscore
    def train_network(self, epoch, loader, predict_num=1):
        # if not None:
        # 	err_file = open(err_save_path, "a+")
        self.train()
        # self.guidance_extrator.eval()
        ## Update the learning rate based on the current epcoh
        # self.scheduler.step(epoch - 1)  # StepLR 位置
        # self.schedulerLoss.step(epoch - 1)
        index, losses = 0, 0
        acc_dict = {}
        lr = self.optim.param_groups[0]['lr']
        # lrLoss = self.optimLoss.param_groups[0]['lr']
        # if ((self.truncate_epoch + 1) == epoch):
        # 	self.optim = torch.optim.SGD(self.parameters(), lr=0.001, weight_decay=2e-5)
        # 	self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=0.97)
            # self.GPT_IConv_NN.MaskSelect.MaskSelect_Mask = True
        num_r = loader.dataset.train_length // loader.batch_size
        print("batch num:{}, batch size:{}".format(num_r, loader.batch_size))
        for num, (data, labels) in enumerate(loader, start=1):
            # print("num:{}, lr:{}".format(num, self.optim.param_groups[0]['lr']))
            self.zero_grad()
            labels = torch.LongTensor(labels).cuda()
            loss, predict_name, predict, emds_dict = self.forward(data, aug=True, label=labels)
            if num == 1:  # 初始化
                assert (len(predict_name) == len(predict)), "给定的名称和预测不相等！！！ -err from yzx"
                for pname in predict_name:
                    acc_dict[pname] = torch.tensor([0.], device=emds_dict[0].device)
            for idx_acc, key_acc in enumerate(acc_dict):
                prec = accuracy(predict[idx_acc].detach(), labels.detach(), topk=(1,))[0]
                acc_dict[key_acc] += prec
            loss.backward()
            self.optim.step()
            # self.scheduler.step(epoch=11238*(epoch-1)+(num-1))  # CyclicLR 之前使用位置
            self.scheduler.step(epoch=num_r*(epoch-1)+(num-1))  # CyclicLR 位置
            self.schedulerLoss.step(epoch=num_r*(epoch-1)+(num-1))  # CyclicLR 位置

            # self.schedulerLoss.step()
            index += len(labels)
            losses += loss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                             " epoch [%2d] Lr: %.8f, Training: %.2f%%" % (epoch, self.optim.param_groups[0]['lr'], 100 * (num / loader.__len__())) + \
                             " Loss: %.5f" % (
                             losses / (num)))
            for k in acc_dict.keys():
                sys.stderr.write(" , %s: %2.2f%% " % (k, acc_dict[k]/index*len(labels)))
            sys.stderr.write("\n")
            sys.stderr.flush()

        top1_acc_list = [acc_dict[k] / index * len(labels) for k in acc_dict.keys()]
        return losses/num, lr, top1_acc_list, len(emds_dict)

    # test1.wav test2.wav 0  分别读取
    # 1.去重  2.放入testLoader 送入model 3.按照dict读
    def eval_network(self, testLoader, emb_kind_num=1, eval_split_num=5, err_save_path=None):
        # if err_save_path is not None:
        # 	err_file = open(err_save_path, "a+")
        as_normal = True
        d_mode = 'cos'
        as_normal_dict = False
        self.eval()
        emds_dict = {}
        labels = []
        dataset_len = len(testLoader.dataset)
        eval_split_num = testLoader.dataset.eval_split_num
        pc = math.floor(dataset_len/testLoader.batch_size) + 1
        print("测试：mininterval={}, maxinterval={}, total={}".format(20, 60, pc))
        # for num, (full_utter, split_utt, file, start_frame) in tqdm.tqdm(enumerate(testLoader, start=1), mininterval=20, maxinterval=60, total=pc):
        for num, (full_utter, split_utt, file, start_frame) in enumerate(testLoader, start=1):
            with torch.no_grad():
                full_utter_list = []
                for _ in range(len(file)):
                    tmp_full_utter = full_utter[0][:, start_frame[0]:].squeeze_(1).cuda()
                    full_utter = full_utter[1::]
                    start_frame = start_frame[1::]
                    _, _, _, full_utter_list_one = self.forward(tmp_full_utter, aug=False, label=labels)
                    if num == 1:  # 初始化
                        emb_kind_num = len(full_utter_list_one)
                    full_utter_list.append([F.normalize(j, p=2, dim=1) for j in full_utter_list_one])
                del full_utter
                del start_frame
                split_utt = split_utt.view(-1, split_utt.shape[-1]).squeeze_(0).cuda()
                _, _, _, sample2 = self.forward(split_utt, aug=False, label=labels)
                del split_utt
                sample2 = [F.normalize(j, p=2, dim=1) for j in sample2]
                for idx, val in enumerate(file):  # 测试的种类数
                    emds_dict[val] = [full_utter_list[idx], [j[idx*eval_split_num:(idx+1)*eval_split_num] for j in sample2]]
        scores, labels = [], []
        scores_full, scores_spilt = [], []
        as_scores = []
        as_scores_full, as_scores_spilt = [], []
        multi_test_EERS, multi_test_minDCFS = {}, {}
        as_multi_test_EERS, as_multi_test_minDCFS = {}, {}

        # np.save('exps/emds_dict/emds_dict_ecapa2_2s.npy', emds_dict, allow_pickle=True)
        # np.save('exps/emds_dict/emds_dict_ecapa2_2sAndFull.npy', emds_dict, allow_pickle=True)
        # quit()
        # 调试用  当然也可加快训练速度 保存测试的嵌入向量
        if emds_dict is None:
            emds_dict = np.load('exps/emds_dict/test_emds_dict_TRCNN_2s.npy', allow_pickle=True).item()
        # top500 这里直接加载
        if as_normal:
            imposter_emds = torch.tensor(np.load('exps/as_normal/train_emds_Center_TRCNN_3s.npy', allow_pickle=True))
        global_cohort_embds_dict = {}
        adaptive_cohort_size = 500
        for ll in range(len(testLoader.dataset.test_list_path)):
            lines = pd.read_csv(testLoader.dataset.test_list_path[ll], delimiter=' ')
            name = testLoader.dataset.test_list_path[ll].split('/')[-1].split('.')[0]
            for _, line in lines.iterrows():
                labels.append(int(line[0]))
                utt_row_enr = emds_dict[line[1]]
                utt_row_tst = emds_dict[line[2]]
                # 只获取 冒认者 去重
                if as_normal and as_normal_dict:
                    spk_enr = line[1].split('/')[0]
                    spk_tst = line[2].split('/')[0]
                    if global_cohort_embds_dict.get(spk_enr) is None:
                        cohort_embds = self.get_cohort_embds(spk_enr, cohort_embds=imposter_emds)
                        global_cohort_embds_dict[spk_enr] = cohort_embds
                    if global_cohort_embds_dict.get(spk_tst) is None:
                        cohort_embds = self.get_cohort_embds(spk_tst, cohort_embds=imposter_emds)
                        global_cohort_embds_dict[spk_tst] = cohort_embds

                for j in range(emb_kind_num):
                    score_full = torch.mean(torch.matmul(utt_row_enr[0][j], utt_row_tst[0][j].T))  # 计算整段
                    score_split_utt = torch.mean(torch.matmul(utt_row_enr[1][j], utt_row_tst[1][j].T))  # 计算五段
                    score = (score_full + score_split_utt) / 2
                    scores.append(score.detach().cpu().numpy())
                    scores_full.append(score_full.detach().cpu().numpy())
                    scores_spilt.append(score_split_utt.detach().cpu().numpy())
                    if as_normal:
                        if as_normal and as_normal_dict:
                            as_score_full = self.score_as_normal(utt_row_enr[0][j], utt_row_tst[0][j], + \
                                global_cohort_embds_dict[spk_enr], global_cohort_embds_dict[spk_tst], score_full,
                                                                 adaptive_cohort_size=adaptive_cohort_size, mode=d_mode)
                            as_score_spilt = self.score_as_normal(utt_row_enr[1][j], utt_row_tst[1][j], + \
                                global_cohort_embds_dict[spk_enr], global_cohort_embds_dict[spk_tst], score_split_utt,
                                                                  adaptive_cohort_size=adaptive_cohort_size, mode=d_mode)
                        else:
                            imposter_emds = imposter_emds.to(score_full.device)
                            as_score_full = self.score_as_normal(utt_row_enr[0][j], utt_row_tst[0][j], + \
                                imposter_emds, imposter_emds, score_full,
                                                                 adaptive_cohort_size=adaptive_cohort_size, mode=d_mode)
                            as_score_spilt = self.score_as_normal(utt_row_enr[1][j], utt_row_tst[1][j], + \
                                imposter_emds, imposter_emds, score_split_utt,
                                                                  adaptive_cohort_size=adaptive_cohort_size, mode=d_mode)
                        as_score = (as_score_full + as_score_spilt) / 2
                        as_scores_full.append(as_score_full.detach().cpu().numpy())
                        as_scores_spilt.append(as_score_spilt.detach().cpu().numpy())
                        as_scores.append(as_score.detach().cpu().numpy())
            EERS, minDCFS = [], []
            as_EERS, as_minDCFS = [], []
            if as_normal:
                as_EERS, as_minDCFS = [], []
            for i in range(emb_kind_num):
            # Coumpute EER and minDCF
                # 没有as normal
                EER = tuneThresholdfromScore(scores[i::emb_kind_num], labels, [1, 0.1])[1]
                EER_full = tuneThresholdfromScore(scores_full[i::emb_kind_num], labels, [1, 0.1])[1]
                EER_split_utt = tuneThresholdfromScore(scores_spilt[i::emb_kind_num], labels, [1, 0.1])[1]
                EERS.append(EER)
                fnrs, fprs, thresholds = ComputeErrorRates(scores[i::emb_kind_num], labels)
                minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, 1, 1)
                fnrs_full, fprs_full, thresholds_full = ComputeErrorRates(scores_full[i::emb_kind_num], labels)
                minDCF_full, _ = ComputeMinDcf(fnrs_full, fprs_full, thresholds_full, 0.01, 1, 1)
                fnrs_split_utt, fprs_split_utt, thresholds_split_utt = ComputeErrorRates(scores_spilt[i::emb_kind_num], labels)
                minDCF_split_utt, _ = ComputeMinDcf(fnrs_split_utt, fprs_split_utt, thresholds_split_utt, 0.01, 1, 1)
                print("没有 as normal ！！！")
                print(
                "{} emb number {}: EER: {:.3f} EER_full: {:.3f}, EER_split_utt: {:.3f}, minDCF: {:.6f}, minDCF_full: {:.6f}, minDCF_split_utt: {:.6f}".format(
                    name, i + 1, EER, EER_full, EER_split_utt, minDCF, minDCF_full, minDCF_split_utt))
                minDCFS.append(minDCF)
                if as_normal:
                    # as normal
                    as_EER = tuneThresholdfromScore(as_scores[i::emb_kind_num], labels, [1, 0.1])[1]
                    as_EER_full = tuneThresholdfromScore(as_scores_full[i::emb_kind_num], labels, [1, 0.1])[1]
                    as_EER_split_utt = tuneThresholdfromScore(as_scores_spilt[i::emb_kind_num], labels, [1, 0.1])[1]
                    as_EERS.append(as_EER)
                    as_fnrs, as_fprs, as_thresholds = ComputeErrorRates(as_scores[i::emb_kind_num], labels)
                    as_minDCF, _ = ComputeMinDcf(as_fnrs, as_fprs, as_thresholds, 0.01, 1, 1)
                    as_fnrs_full, as_fprs_full, as_thresholds_full = ComputeErrorRates(as_scores_full[i::emb_kind_num], labels)
                    as_minDCF_full, _ = ComputeMinDcf(as_fnrs_full, as_fprs_full, as_thresholds_full, 0.01, 1, 1)
                    as_fnrs_split_utt, as_fprs_split_utt, as_thresholds_split_utt = ComputeErrorRates(as_scores_spilt[i::emb_kind_num], labels)
                    as_minDCF_split_utt, _ = ComputeMinDcf(as_fnrs_split_utt, as_fprs_split_utt, as_thresholds_split_utt, 0.01, 1, 1)
                    print("应用 as normal ！！！")
                    print(
                        "{} emb number {}: as_EER: {:.3f} as_EER_full: {:.3f}, as_EER_split_utt: {:.3f}, as_minDCF: {:.6f}, as_minDCF_full: {:.6f}, as_minDCF_split_utt: {:.6f}".format(
                            name, i + 1, as_EER, as_EER_full, as_EER_split_utt, as_minDCF, as_minDCF_full, as_minDCF_split_utt))
                    as_minDCFS.append(as_minDCF)
            multi_test_EERS[name] = EERS
            multi_test_minDCFS[name] = minDCFS
            as_multi_test_EERS[name] = as_EERS
            as_multi_test_minDCFS[name] = as_minDCFS
        if as_normal:
            return (multi_test_EERS, as_multi_test_EERS), (multi_test_minDCFS, as_multi_test_minDCFS), emb_kind_num, as_normal
        else:
            return (multi_test_EERS, ), (multi_test_minDCFS, ), emb_kind_num, as_normal

    # test1.wav test2.wav 0 分别读取
    # 1.去重复放入dict  2.按照dict读取
    def eval_network1(self, eval_list, eval_path, emb_kind_num=1):
        emb_kind_num = 1
        self.eval()
        files = []
        emds_dict = {}
        lines = open(eval_list).read().splitlines()
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()

        # for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
        for idx, file in enumerate(setfiles):
            audio, _  = torchaudio.load(os.path.join(eval_path, file))
            audio.squeeze_(0)
            # Full utterance
            data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()

            # Spliited utterance matrix
            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            feats = []
            # seperate 5 vector
            startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])
            feats = numpy.stack(feats, axis = 0).astype(float)
            data_2 = torch.FloatTensor(feats).cuda()
            # Speaker emds_dict
            with torch.no_grad():
                # data1  1*n
                self.train_flag = False
                _, _, full_utter_list = self.forward(data_1, aug=False)
                _, _, sample2 = self.forward(data_2, aug=False)
                tmpList = []
                if len(full_utter_list) != emb_kind_num:
                    emb_kind_num = len(full_utter_list)
                for i in range(emb_kind_num):
                    tmp1 = nn.functional.normalize(full_utter_list[i], p=2, dim=1)
                    tmp2 = nn.functional.normalize(sample2[i], p=2, dim=1)
                    tmpList.append([tmp1, tmp2])
                emds_dict[file] = tmpList
        scores, labels = [], []
        EERS, minDCFS = [], []
        for line in lines:
            embedding_X = [(i, j) for i, j in enumerate(zip(emds_dict[line.split()[1]], emds_dict[line.split()[2]]))]
            # X_VEC[i][1] 种类数
            score_full = [(torch.mean(torch.matmul(embedding_X[i][1][0][0], embedding_X[i][1][1][0].T))) for i in range(emb_kind_num)]  # 计算一段
            score_split_utt = [(torch.mean(torch.matmul(embedding_X[i][1][0][1], embedding_X[i][1][1][1].T))) for i in range(emb_kind_num)]  # 计算五段

            for i in range(emb_kind_num):
                score = (torch.tensor(score_full[i]) + torch.tensor(score_split_utt[i])) / 2
                score = score.detach().cpu().numpy()
                scores.append(score)
            labels.append(int(line.split()[0]))
        for i in range(emb_kind_num):
        # Coumpute EER and minDCF
            EER = tuneThresholdfromScore(scores[i::emb_kind_num], labels, [1, 0.1])[1]
            EERS.append(EER)
            fnrs, fprs, thresholds = ComputeErrorRates(scores[i::emb_kind_num], labels)
            minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, 1, 1)
            minDCFS.append(minDCF)
        return EERS, minDCFS, emb_kind_num


     #  test1.wav test2.wav 0 同时读取
     #  1.放入testLoader 2.按照顺序不去重复
    def eval_network2(self, testLoader, emb_kind_num):
        self.eval()
        scores = [[], []]
        labels = []
        EERS = []
        minDCFS = []
        dataset_len = len(testLoader.dataset)
        pc = math.floor(dataset_len/testLoader.batch_size)
        for num, (data_1, data_2, label) in tqdm.tqdm(enumerate(testLoader, start=1), total=pc):
            with torch.no_grad():
                B, G, F = data_1.shape
                # full    and   	split_5
                data_1 = data_1.view(B*G, F).cuda()
                data_2 = data_2.view(B*G, F).cuda()
                label = label.flatten()
                _, _, embedding_1 = self.forward(data_1, aug=False)
                _, _, embedding_2 = self.forward(data_2, aug=False)

                labels.append(label)
                for i in range(emb_kind_num):
                    # 与下面的等价
                    # embedding_1_op = nn.functional.normalize(embedding_1[i], p=2, dim=1)
                    # embedding_2_op = nn.functional.normalize(embedding_2[i], p=2, dim=1)
                    # score_embedding = torch.diag(torch.matmul(embedding_1_op, embedding_2_op.T))
                    # 与上面等价
                    score_embedding = F.cosine_similarity(embedding_1[i], embedding_2[i], dim=1)
                    score_step = 6
                    score_full = 0
                    score_split_utt = 0
                    for j in range(score_step):
                        if j == 0:
                            score_full = score_embedding[j::score_step]
                        else:
                            score_split_utt += score_embedding[j::score_step]
                    del score_embedding
                    score = (score_full + score_split_utt/(score_step-1)) / 2
                    score = score.detach().cpu().numpy()
                    scores[i].append(score)

        labels = np.concatenate([t.flatten() for t in labels], axis=0)
        for i in range(emb_kind_num):
        # Coumpute EER and minDCF
            scores[i] = np.concatenate([t.flatten() for t in scores[i]], axis=0)
            EER = tuneThresholdfromScore(scores[i], labels, [1, 0.1])[1]
            EERS.append(EER)
            fnrs, fprs, thresholds = ComputeErrorRates(scores[i], labels)
            minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, 1, 1)
            minDCFS.append(minDCF)
        return EERS, minDCFS

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path, deviceno=0):
        self_state = self.state_dict()
        loaded_state = torch.load(path, map_location=torch.device(deviceno))
        for name, param in loaded_state.items():

            origname = name
            if name not in self_state:
                # name = name.replace("speaker_AAMsoftmax_loss", "Control_Contrastive_loss")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
            # print("%s 加载" % origname)
    def load_needed_parameters(self, path, deviceno):
        n = 0
        zero_num = math.floor(math.pow(2, n-1))
        self_model_state = self.GPT_IConv_NN.state_dict()
        self_loss_state = self.Control_Contrastive_loss.state_dict()
        loaded_state = torch.load(path, map_location=torch.device(deviceno))
        print("pretrain batch size 400 ,%s" %path)
        for name, param in loaded_state.items():
            origname = name
            name = name.replace("speaker_encoder.", "")
            if 'speaker_loss.weight' in origname:
                self_loss_state["weight"].copy_(param)
                print("Load %s in the model." % origname)
            elif ('conv' in origname and 'attention' not in origname) and 1 == 0 :
                # 稍微改下名字
                name = name.replace("conv1", "upChannel") if "conv1" in origname else name
                # 先处理 卷积核大于1的
                if loaded_state[origname].shape[-1] > 1 and 'weight' in name:
                    name = name.replace("weight", "conv1d.weight")
                    if (self_model_state[name].shape[-1] != param.shape[-1]):
                        zero_num = (self_model_state[name].shape[-1] - 1)//(param.shape[-1] - 1) - 1
                    p_outchannel = self_model_state[name].shape[0] // (zero_num * 2)
                    for i in range(zero_num * 2):
                        self_model_state[name][i * p_outchannel:(i+1) * p_outchannel, :, 0::zero_num+1].copy_(param[i * p_outchannel:(i+1) * p_outchannel, 0:self_model_state[name].shape[1], :])
                    print("Load %s上采样卷积核大于1的!" % origname)
                else:
                    pass
            elif (1 == 0) or ('bn5' in origname) or \
                    ('fc6' in origname) or ('bn6' in origname):
                # self_model_state["layer5"].copy_(param[])
                if 'attention.0' in name:
                    name = name.replace('attention.0', 'attention.0.conv1d')
                if 'attention.4' in name:
                    name = name.replace('attention.4', 'attention.4.conv1d')
                self_model_state[name].copy_(param)
                print("Load attention bn fc 后半部分 %s!" % name)
            else:
                print("----   No needed parameter %s." % origname)
                continue

    def frozen_needed_parameters(self, flag = True):
        # 先处理 卷积核大于1的
        for name, param in self.GPT_IConv_NN.named_parameters():
            if 'conv1d.weight' in name and param.shape[-1] > 1:
                # 主要的赋值
                pass
                # param.requires_grad = False
                # print("conv1d.weight : %s 冻结  param.requires_grad:%s !" % (name, param.requires_grad))
            # elif not (('upChannel' in name) or ('layer5' in name)):
            elif not ('layer5' in name):
                # self_model_state["layer5"].copy_(param[])
                param.requires_grad = flag
                print(" %s: param.requires_grad:%s !" % (name, param.requires_grad))
        # self.Control_Contrastive_loss.weight.requires_grad = flag
        # print("self.GPT_IConv_NN 冻结 结束 ！")

    @torch.no_grad()
    def macs_params(self, input_size=(1, 80, 202), device='cpu', custom_ops=None, verbose=True):
        """
        Count MACs (multiply-and-accumulates) and Params (parameters) using thop.

        Calculate a model/module's MACs/MAdd and parameters in PyTorch.

            Parameter
            ---------
                model : torch.nn.Module
                input_size : tuple or list
                custom_ops : dict
                    Ops as register_ops in `thop`.
                verbose : bool

            Return
            ------
                macs : str
                    the number of MACs/MAdd
                params : str
                    the number of parameters

            Note
            ----
                There always is a gap between the MACs from `profile` and from `count_ops`, in which the 
                    former is lower. The reason is that the operation of `x * y` in SELayer and the 
                    operation of `out += identify` in L3BasicBlock are not considered in `profile`. The 
                    comment 'This is a place that profile can not count' is appended to those operations.
        """
        import thop
        model = self.GPT_IConv_NN
        model.eval()
        # model.to(device)
        with torch.no_grad():
            inp = torch.rand(*input_size, device=device)
            macs, params = thop.profile(model, inputs=(inp,), custom_ops=custom_ops, verbose=verbose)
        if format:
            macs, params = thop.clever_format([macs*1000*1000/1024/1024, params*1000*1000/1024/1024], "%.4f")
        print('{}: MACs {} and Params {} using input of {}'.format(model.__class__.__name__, macs, params,
                                                                       input_size))
        # return macs, params