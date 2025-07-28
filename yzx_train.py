'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''

import argparse, glob, os, torch, warnings, time, torch.nn as nn
from tools import *
from dataLoader import deal_train_loader, origin_train_loader, test_loader
from ModelMaster import ModelMaster
from model import Ecapa_Tdnn_Model
import random, math
# from transformers import Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2FeatureExtractor, WavLMForXVector

parser = argparse.ArgumentParser(description="New_Ecapa_trainer")
parser.add_argument('--n', type=int, default=1, help='Number of order/head? num')

parser.add_argument('--batch_size', type=int, default=2, help='Batch size  300  try 64 128')
parser.add_argument('--n_cpu', type=int, default=1, help='Number of train threads 15')
parser.add_argument('--deviceno', type=int, default=0, help='device NUM')
parser.add_argument('--C', type=int, default=512, help='新网络专用的通道数 512')
# parser.add_argument('--C', type=int, default=1024, help='Channel size for the speaker encoder')
parser.add_argument('--m', type=float, default=0.2, help='Loss margin in AAM softmax 0.2(0.9801) try 0.35(0.9393)')
parser.add_argument('--s', type=float, default=30, help='Loss scale in AAM softmax  30 try 60')

# Training and evaluation path/lists, save path
parser.add_argument('--n_class', type=int, default=5994, help='Number of speakers 1211 | 5994')

# 超算服务器

parser.add_argument('--train_list', type=str, default="exps/vox2_train.txt",
					help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
parser.add_argument('--train_path', type=str, default="/root/private_data/wmnt/voxceleb/vox2_dev_wav",
					help='The path of the training DAE_data, eg:"/data08/VoxCeleb2/train/wav" in my case')
# parser.add_argument('--train_list', type=str, default="exps/vox1_train_v2.txt",
#                     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
# parser.add_argument('--train_path', type=str, default="/root/private_data/wmnt/voxceleb/vox1_dev_wav",
#                     help='The path of the training DAE_data, eg:"/data08/VoxCeleb2/train/wav" in my case')
parser.add_argument('--eval_path', type=list, default=["/root/private_data/wmnt/voxceleb/vox1_test_wav"],
                    help='The path of the evaluation DAE_data, eg:"/data08/VoxCeleb1/test/wav" in my case')
parser.add_argument('--eval_list', type=list, default=["exps/voxceleb1_o.txt"],
                    help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')

parser.add_argument('--musan_path', type=str, default="/root/private_data/wmnt/voxceleb/musan",
                    help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
parser.add_argument('--rir_path', type=str, default="/root/private_data/wmnt/voxceleb/RIRS_NOISES/simulated_rirs",
                    help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES" in my case')

## Training Settings
parser.add_argument('--model_mode', type=str, default='Taylor_DGCNN', help='YZX | Taylor_DGCNN')
parser.add_argument('--num_frames', type=int, default=200,
                    help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch', type=int, default=1000, help='Maximum number of epochs')
parser.add_argument('--truncate_epoch', type=int, default=39, help='黄金分割率')
parser.add_argument('--test_n_cpu', type=int, default=10, help='Number of test loader threads')
parser.add_argument('--dim', type=int, default=192, help='Number of embedding dim')
parser.add_argument('--test_step', type=int, default=1, help='Test and save every [test_step] epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate 0.001')
parser.add_argument("--lr_decay", type=float, default=0.97, help='0.97 Learning rate decay every [test_step] epochs')

parser.add_argument('--save_path', type=str, default="exps/exp1", help='Path to save the score_ex_EcapaC1024_aam_v2_b400txt and models/ and error.txt')
# parser.add_argument('--initial_model', type=str, default="exps/exp1/model/model_0088.model", help='Path of the initial_model')
parser.add_argument('--initial_model', type=str, default="", help='Path of the initial_model')
parser.add_argument('--teacher_model', type=str, default="exps/pretrain.model", help='Path of the teacher model')

## Model and Loss settings
# parser.add_argument('--eval', dest='eval', action='store_true', help='Only do evaluation')
parser.add_argument('--eval', type=bool, default=False, help='Only do evaluation')
parser.add_argument('--eval_split_num', type=int, default=5, help='score every [eval_split_num] num')

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)

# torch.manual_seed(0)
# random.seed(0)  # 默认100
# cuda 设置
torch.cuda.set_device(args.deviceno)  # 0 / 1 目前两个

## Define the train_data loader
trainloader = origin_train_loader(**vars(args))
trainLoader = torch.utils.data.DataLoader(trainloader, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu,
                                          drop_last=True, pin_memory=True)

test_loader = test_loader(**vars(args))
#  服务器超过20 就会重置
testLoader = torch.utils.data.DataLoader(test_loader, batch_size=4, shuffle=False, num_workers=args.test_n_cpu,
                                         drop_last=False, pin_memory=True)
## Search for the exist models
modelfiles = glob.glob('%s/model_0*.model' % args.model_save_path)
# modelfiles = glob.glob('%s/model_0*.model' % 'exps/exp1/model_NPSV_v2_high_B128')
modelfiles.sort()

# 主要的测试地方
if args.model_mode == 'Taylor_DGCNN':
    if args.eval == True:
    # if True:
        load_path = modelfiles[-1]

        # # 冻结/不冻结 需要实验
        # teacher = Ecapa_Tdnn(C=args.C, dim=args.dim)
        # teacher.load_teacher_parameters(args.initial_model)
        # args.teacher = teacher

        # 冻结/不冻结 需要实验
        # W2V2Config = Wav2Vec2Config()
        # args.W2V2Config = W2V2Config
        # Wav2Vec2Model = Wav2Vec2Model(config=W2V2Config)
        # Wav2Vec2Model = load_parameters(Wav2Vec2Model, load_path, device=args.deviceno)
        # args.Wav2Vec2Model = Wav2Vec2Model


        # 多个大模型的测试
        # 1.
        # feature_extractor = Wav2Vec2FeatureExtractor(sampling_rate=16000).from_pretrained('exps/wavlm-base-sv')
        # model = WavLMForXVector.from_pretrained('exps/wavlm-base-sv')
        # args.feature_extractor = feature_extractor
        # args.WavLMForXVector = model
        # load
        # 2.
        # teacher = torch.jit.load('exps/ecapa2.pt', map_location="cuda:0")
        # args.teacher = teacher
        #     3. 论文模型
        s = ModelMaster(**vars(args))
        # s.load_parameters(load_path, deviceno=args.deviceno)
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "Model %s loaded from previous state!" % (load_path))
        s.train_flag = False
        #  emb_kind_num 这里需要设置一下
        eersss, minDCFsss, emb_kind_num, as_normal_flag = s.eval_network(testLoader=testLoader)
        for idx_asn in range(len(eersss)):  # 针对 as noramal
            for ll in range(len(testLoader.dataset.test_list_path)):
                name = testLoader.dataset.test_list_path[ll].split('/')[-1].split('.')[0]
                tmp_as_normal_name = 'no_asnormal'
                eers, minDCFs = eersss[idx_asn][name], minDCFsss[idx_asn][name]
                if idx_asn == 1:
                    tmp_as_normal_name = 'as_normal'
                # EER, minDCF, ekn, as_normal_flag = s.eval_network1(eval_list = args.eval_list, eval_path = args.eval_path)
                # print("EER %2.2f%%, minDCF %.4f%%"%(EER, minDCF))
                for i in range(emb_kind_num):
                    print("End Test Time: ", time.strftime("%Y-%m-%d %H:%M:%S"),
                          " %s, Test dataset: %s,  the tasks %d: Test EER %2.3f%%,  Test minDCF %2.5f" % (tmp_as_normal_name, name,
                          i + 1, eers[i::emb_kind_num][-1], minDCFs[i::emb_kind_num][-1]))
        quit()


    ## If initial_model is exist, system will train from the initial_model
    if args.initial_model != "":
        print("Model %s loaded from previous state!" % args.initial_model)
        s = ModelMaster(**vars(args))
        s.load_parameters(path=args.initial_model, deviceno=args.deviceno)
        epoch = 1

    ## Otherwise, system will try to start from the saved model&epoch
    elif len(modelfiles) >= 1:
        print("Model %s loaded from previous state!" % modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        # teacher = torch.jit.load('exps/ecapa2.pt', map_location="cuda:0")
        # args.teacher = teacher
        s = ModelMaster(**vars(args))
        s.load_parameters(modelfiles[-1], deviceno=args.deviceno)
    ## Otherwise, system will train from scratch
    else:
        epoch = 1
        s = ModelMaster(**vars(args))
        # s.load_needed_parameters('exps/pretrain.model', 0)
        print("从头训练!!!!")
elif args.model_mode == 'YZX':
    # teacher = Ecapa_Tdnn(C=args.C, dim=args.dim)

    # # 冻结/不冻结 需要实验
    # teacher.load_teacher_parameters(args.teacher_model)
    # args.teacher = teacher
    #
    # W2V2Config = Wav2Vec2Config()
    # args.W2V2Config = W2V2Config
    # # 关闭梯度
    # for param in teacher.parameters():
    #     param.requires_grad = False

    # feature_extractor = Wav2Vec2FeatureExtractor(sampling_rate=16000).from_pretrained('exps/wavlm-base-sv')
    # model = WavLMForXVector.from_pretrained('exps/wavlm-base-sv')
    # args.feature_extractor = feature_extractor
    # args.WavLMForXVector = model
    # 先注释掉  2024年9月10日21:50:35  为了random一个scale 和 list
    s = ModelMaster(**vars(args))
    s.train_flag = True
    epoch = 1

EERs = []
minDCFs = []

for i in range(len(args.eval_list)):
    EERs.append([])
    minDCFs.append([])

score_file = open(args.score_save_path, "a+")

while (1):
    ## Training for one epoch
    # torch.manual_seed(epoch)
    # random.seed(epoch)
    # 每次都新建立一个模型
    # s = ModelMaster(**vars(args))
    s.train_flag = True
    # acc_List 手动对应吧
    # loss, lr, acc_List = s.train_network(epoch=epoch, loader=trainLoader, emb_kind_num=args.emb_kind_num, err_save_path=args.err_save_path)
    # 再次随机取样
    # trainloader = deal_train_loader(**vars(args))
    # trainLoader = torch.utils.data.DataLoader(trainloader, batch_size=args.batch_size, shuffle=True,
    #                                           num_workers=args.n_cpu,
    #                                           drop_last=True, pin_memory=True)

    loss, lr, acc_List, emb_kind_num = s.train_network(epoch=epoch, loader=trainLoader)

    ## Evaluation every [test_step] epochs
    if epoch % args.test_step == 0:
        s.save_parameters(args.model_save_path + "/model_%04d.model" % epoch)
        s.train_flag = False
        print(" Start Test: ", time.strftime("%Y-%m-%d %H:%M:%S"), "epoch: [%d]" % (epoch))
        eersss, minDCFsss, ekn, as_normal_flag = s.eval_network(testLoader=testLoader)
        for idx_asn in range(len(eersss)):  # 针对 as noramal
            for ll in range(len(args.eval_list)):
                name = args.eval_list[ll].split('/')[-1].split('.')[0]
                tmp_as_normal_name = 'no_asnormal'
                eer, minDCF = eersss[idx_asn][name], minDCFsss[idx_asn][name]
                # eer, minDCF, ekn, as_normal_flag = s.eval_network1(eval_list = args.eval_list, eval_path = args.eval_path)
                if idx_asn == 1:
                    tmp_as_normal_name = 'as_normal'
                score_file.write(
                    " [%s] [%s] epoch:  [%d], LR %f, LOSS %f" % (
                        tmp_as_normal_name, name, epoch, lr, loss))
                for i in range(ekn):
                    EERs[ll].append(eer[i])
                    minDCFs[ll].append(minDCF[i])
                    print(
                         " ---The trail %d: ACC %2.2f%%, EER %2.3f%%, bestEER %2.3f%% at epoch [%d], minDCF %2.5f, bestminDCF %2.5f" % (
                           i + 1, acc_List[0], EERs[ll][i::ekn][-1], min(EERs[ll][i::ekn]), EERs[ll][i::ekn].index(min(EERs[ll][i::ekn]))+1,
                           minDCFs[ll][i::ekn][-1], min(minDCFs[ll][i::ekn])))
                    score_file.write(
                        " ---The trail %d: ACC %2.2f%%, EER %2.3f%%, bestEER %2.3f%% at epoch [%d], minDCF %2.5f, bestminDCF %2.5f" % (
                            i + 1, acc_List[0], EERs[ll][i::ekn][-1], min(EERs[ll][i::ekn]),
                            EERs[ll][i::ekn].index(min(EERs[ll][i::ekn])) + 1,
                            minDCFs[ll][i::ekn][-1], min(minDCFs[ll][i::ekn])))
                    if i == ekn-1:
                        score_file.write("\n")
            score_file.flush()
            print(" End Test: ", time.strftime("%Y-%m-%d %H:%M:%S"))
    if epoch >= args.max_epoch:
        quit()

    epoch += 1
