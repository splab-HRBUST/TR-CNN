'''
Some utilized functions
These functions are all copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/tuneThreshold.py
'''

import os, numpy, torch
from sklearn import metrics
from operator import itemgetter
import torch.nn.functional as F
import torch.nn as nn


def load_parameters(model: nn.Module, path):
    self_state = model.state_dict()
    loaded_state = torch.load(path)
    for name, param in loaded_state.items():
        origname = name
        if name not in self_state:
            name = name.replace("wav2Vec2Model.", "")
            if name not in self_state:
                print("%s is not in the model." % origname)
                continue
        if self_state[name].size() != loaded_state[origname].size():
            print("Wrong parameter length: %s, model: %s, loaded: %s" % (
            origname, self_state[name].size(), loaded_state[origname].size()))
            continue
        self_state[name].copy_(param)
        model.load_state_dict(self_state)
    return model

def init_args(args):
    # nohup  python yzx_train.py > ex_ecapa_aam_v2.log 2>&1 &
    os.makedirs(args.save_path, exist_ok=True)

# 新的尝试 是否泰勒还没启动

    # args.score_save_path = os.path.join(args.save_path, 'score_demo.txt')
    # args.model_save_path = os.path.join(args.save_path, 'model_demo')

    args.score_save_path = os.path.join(args.save_path, 'score_pretrain1.txt')
    args.model_save_path = os.path.join(args.save_path, 'model_pretrain1')

    # 其他的

    # args.score_save_path = os.path.join(args.save_path, 'score_New_ECAPA_C512_v2_fC_b128_1.txt')
    # args.err_save_path = os.path.join(args.save_path, 'error_New_ECAPA_C512_v2_fC_b128_1.txt')
    # args.model_save_path = os.path.join(args.save_path, 'model_New_ECAPA_C512_v2_fC_b128_1')

    # args.score_save_path = os.path.join(args.save_path, 'score_upchannel_b1_m3_ft_res.txt')
    # args.err_save_path = os.path.join(args.save_path, 'error_upchannel_b1_m3_ft_res.txt')
    # args.model_save_path = os.path.join(args.save_path, 'model_upchannel_b1_m3_ft_res')

    # args.score_save_path = os.path.join(args.save_path, 'score_New_ECAPA_C1024_v2_fCn2_b128.txt')
    # args.err_save_path = os.path.join(args.save_path, 'error_New_ECAPA_C1024_v2_fCn2_b128.txt')
    # args.model_save_path = os.path.join(args.save_path, 'model_New_ECAPA_C1024_v2_fCn2_b128')

    # args.score_save_path = os.path.join(args.save_path, 'score_New_V1_ECAPA_noatt_C512_catn0.txt')
    # args.err_save_path = os.path.join(args.save_path, 'error_New_V1_ECAPA_noatt_C512_catn0.txt')
    # args.model_save_path = os.path.join(args.save_path, 'model_New_V1_ECAPA_noatt_C512_catn0')

    # only kernel:  {m} 一个 m :  kernel_size=m  剩下的 m :  kernel_size=m-1
    #      独立部分
    # args.score_save_path = os.path.join(args.save_path, 'score_New_V1_ECAPA_C512_m_simple_isolate.txt')
    # args.score_save_path = os.path.join(args.save_path, 'score_New_V1_ECAPA_C512_m_Concat_simple_isolate_lr001.txt')
    # args.score_save_path = os.path.join(args.save_path, 'score_New_V1_ECAPA_C512_m_att_isolate.txt')
    # args.score_save_path = os.path.join(args.save_path, 'score_New_V1_ECAPA_C512_m_Concat_att_isolate_lr001.txt')

    # args.err_save_path = os.path.join(args.save_path, 'error_New_V1_ECAPA_C512_m_simple_isolate.txt')
    # args.err_save_path = os.path.join(args.save_path, 'error_New_V1_ECAPA_C512_m_Concat_simple_isolate.txt')
    # args.err_save_path = os.path.join(args.save_path, 'error_New_V1_ECAPA_C512_m_att_isolate.txt')
    # args.err_save_path = os.path.join(args.save_path, 'error_New_V1_ECAPA_C512_m_Concat_att_isolate.txt')

    # args.model_save_path = os.path.join(args.save_path, 'model_New_V1_ECAPA_C512_m_simple_isolate')
    # args.model_save_path = os.path.join(args.save_path, 'model_New_V1_ECAPA_C512_m_Concat_simple_isolate')
    # args.model_save_path = os.path.join(args.save_path, 'model_New_V1_ECAPA_C512_m_att_isolate')
    # args.model_save_path = os.path.join(args.save_path, 'model_New_V1_ECAPA_C512_m_Concat_att_isolate')


    #     不独立部分
    # args.score_save_path = os.path.join(args.save_path, 'score_New_V1_ECAPA_C512_m_simple_n1_lr001.txt')
    # args.score_save_path = os.path.join(args.save_path, 'score_New_V1_ECAPA_C512_m_Concat_simple_n1.txt')
    # args.score_save_path = os.path.join(args.save_path, 'score_New_V1_ECAPA_C512_m_att_n1.txt')
    # args.score_save_path = os.path.join(args.save_path, 'score_New_V1_ECAPA_C512_m_Concat_att_n1.txt')
    #
    # args.err_save_path = os.path.join(args.save_path, 'error_New_V1_ECAPA_C512_m_simple_n1.txt')
    # args.err_save_path = os.path.join(args.save_path, 'error_New_V1_ECAPA_C512_m_Concat_simple_n1.txt')
    # args.err_save_path = os.path.join(args.save_path, 'error_New_V1_ECAPA_C512_m_att_n1.txt')
    # args.err_save_path = os.path.join(args.save_path, 'error_New_V1_ECAPA_C512_m_Concat_att_n1.txt')
    #
    # args.model_save_path = os.path.join(args.save_path, 'model_New_V1_ECAPA_C512_m_simple_n1')
    # args.model_save_path = os.path.join(args.save_path, 'model_New_V1_ECAPA_C512_m_Concat_simple_n1')
    # args.model_save_path = os.path.join(args.save_path, 'model_New_V1_ECAPA_C512_m_att_n1')
    # args.model_save_path = os.path.join(args.save_path, 'model_New_V1_ECAPA_C512_m_Concat_att_n1')



    # kernel:  {m, m - 1} 一个m :  kernel_size=m   剩下的 m-1 :  kernel_size=m-1
    #     独立部分
    # args.score_save_path = os.path.join(args.save_path, 'score_New_V1_ECAPA_C512_mm1_simple_isolate.txt')
    # args.score_save_path = os.path.join(args.save_path, 'score_New_V1_ECAPA_C512_mm1_Concat_simple_isolate.txt')
    # args.score_save_path = os.path.join(args.save_path, 'score_New_V1_ECAPA_C512_mm1_att_isolate.txt')
    # args.score_save_path = os.path.join(args.save_path, 'score_New_V1_ECAPA_C512_mm1_Concat_att_isolate.txt')
    #
    # args.err_save_path = os.path.join(args.save_path, 'error_New_V1_ECAPA_C512_mm1_simple_isolate.txt')
    # args.err_save_path = os.path.join(args.save_path, 'error_New_V1_ECAPA_C512_mm1_Concat_simple_isolate.txt')
    # args.err_save_path = os.path.join(args.save_path, 'error_New_V1_ECAPA_C512_mm1_att_isolate.txt')
    # args.err_save_path = os.path.join(args.save_path, 'error_New_V1_ECAPA_C512_mm1_Concat_att_isolate.txt')
    #
    # args.model_save_path = os.path.join(args.save_path, 'model_New_V1_ECAPA_C512_mm1_simple_isolate')
    # args.model_save_path = os.path.join(args.save_path, 'model_New_V1_ECAPA_C512_mm1_Concat_simple_isolate')
    # args.model_save_path = os.path.join(args.save_path, 'model_New_V1_ECAPA_C512_mm1_att_isolate')
    # args.model_save_path = os.path.join(args.save_path, 'model_New_V1_ECAPA_C512_mm1_Concat_att_isolate')
    #
    # #     不独立部分
    # args.score_save_path = os.path.join(args.save_path, 'score_New_V1_ECAPA_C512_mm1_simple_n1.txt')
    # args.score_save_path = os.path.join(args.save_path, 'score_New_V1_ECAPA_C512_mm1_Concat_simple_n1.txt')
    # args.score_save_path = os.path.join(args.save_path, 'score_New_V1_ECAPA_C512_mm1_att_n1.txt')
    # args.score_save_path = os.path.join(args.save_path, 'score_New_V1_ECAPA_C512_mm1_Concat_att_n1.txt')
    #
    # args.err_save_path = os.path.join(args.save_path, 'error_New_V1_ECAPA_C512_mm1_simple_n1.txt')
    # args.err_save_path = os.path.join(args.save_path, 'error_New_V1_ECAPA_C512_mm1_Concat_simple_n1.txt')
    # args.err_save_path = os.path.join(args.save_path, 'error_New_V1_ECAPA_C512_mm1_att_n1.txt')
    # args.err_save_path = os.path.join(args.save_path, 'error_New_V1_ECAPA_C512_mm1_Concat_att_n1.txt')
    #
    # args.model_save_path = os.path.join(args.save_path, 'model_New_V1_ECAPA_C512_mm1_simple_n1')
    # args.model_save_path = os.path.join(args.save_path, 'model_New_V1_ECAPA_C512_mm1_Concat_simple_n1')
    # args.model_save_path = os.path.join(args.save_path, 'model_New_V1_ECAPA_C512_mm1_att_n1')
    # args.model_save_path = os.path.join(args.save_path, 'model_New_V1_ECAPA_C512_mm1_Concat_att_n1')




    os.makedirs(args.model_save_path, exist_ok = True)
    return args

def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):

	fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
	fnr = 1 - tpr
	tunedThreshold = [];
	if target_fr:
		for tfr in target_fr:
			idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
			tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
	for tfa in target_fa:
		idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
		tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
	idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
	eer  = max(fpr[idxE],fnr[idxE])*100
	
	return tunedThreshold, eer, fpr, fnr

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):

      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      labels = [labels[i] for i in sorted_indexes]
      fnrs = []
      fprs = []

      # At the end of this loop, fnrs[i] is the number of errors made by
      # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
      # is the total number of times that we have correctly accepted scores
      # greater than thresholds[i].
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i-1] + labels[i])
              fprs.append(fprs[i-1] + 1 - labels[i])
      fnrs_norm = sum(labels)
      fprs_norm = len(labels) - fnrs_norm

      # Now divide by the total number of false negative errors to
      # obtain the false positive rates across all thresholds
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      # Divide by the total number of corret positives to get the
      # true positive rate.  Subtract these quantities from 1 to
      # get the false positive rates.
      fprs = [1 - x / float(fprs_norm) for x in fprs]
      return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res