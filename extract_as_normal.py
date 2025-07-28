import argparse, glob, os, torch, warnings, time, torch.nn as nn
import sys

import tqdm

from ModelMaster import ModelMaster
from dataLoader import as_train_loader
import numpy as np
import torch.nn.functional as F

def get_args(num_frames=None):
    parser = argparse.ArgumentParser(description="New_Ecapa_trainer")
    parser.add_argument('--n', type=int, default=1, help='Number of order/head? num')

    parser.add_argument('--batch_size', type=int, default=40, help='Batch size  300  try 64 128')
    parser.add_argument('--n_cpu', type=int, default=10, help='Number of train threads 15')
    parser.add_argument('--deviceno', type=int, default=0, help='device NUM')
    parser.add_argument('--C', type=int, default=512, help='新网络专用的通道数 512')

    #学校服务器

    parser.add_argument('--train_list', type=str, default="exps/vox2_train.txt",
                        help='The path of the training list, https://www.robots.ox.ac.cuk/~vgg/data/voxceleb/meta/train_list.txt')
    parser.add_argument('--train_path', type=str, default="/g813_u1/wmnt/voxceleb/vox2_dev_wav/wav",
                        help='The path of the training DAE_data, eg:"/data08/VoxCeleb2/train/wav" in my case')
    # parser.add_argument('--train_list', type=str, default="exps/vox1_train_v2.txt",
    #                     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
    # parser.add_argument('--train_path', type=str, default="/g813_u1/wmnt/voxceleb/vox1_dev_wav/wav",
    #                     help='The path of the training DAE_data, eg:"/data08/VoxCeleb2/train/wav" in my case')
    parser.add_argument('--eval_path', type=list, default=["/g813_u1/wmnt/voxceleb/vox1_test_wav"],
                        help='The path of the evaluation DAE_data, eg:"/data08/VoxCeleb1/test/wav" in my case')
    parser.add_argument('--eval_list', type=list, default=["exps/voxceleb1_o.txt"],
                        help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')

    # parser.add_argument('--eval_path', type=list, default=["/g813_u1/wmnt/voxceleb/vox1_test_wav", "/g813_u1/wmnt/voxceleb/vox1_dev_wav/wav"],
    #                     help='The path of the evaluation DAE_data, eg:"/data08/VoxCeleb1/test/wav" in my case')
    # parser.add_argument('--eval_list', type=list, default=["exps/voxceleb1_o.txt", "exps/voxceleb1_e.txt", "exps/voxceleb1_h.txt"],
    #                     help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')

    parser.add_argument('--musan_path', type=str, default="/g813_u1/wmnt/g813_u3/musan",
                        help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
    parser.add_argument('--rir_path', type=str, default="/g813_u1/wmnt/g813_u3/RIRS_NOISES/simulated_rirs",
                        help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES" in my case')


    ## Training Settings
    parser.add_argument('--model_mode', type=str, default='Taylor_DGCNN', help='YZX | Taylor_DGCNN')
    parser.add_argument('--num_frames', type=int, default=500,
                        help='Duration of the input segments, eg: 200 for 2 second')
    parser.add_argument('--dim', type=int, default=192, help='Number of embedding dim')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate 0.001')
    parser.add_argument("--lr_decay", type=float, default=0.97, help='0.97 Learning rate decay every [test_step] epochs')
    parser.add_argument('--initial_model', type=str, default="exps/exp1/model_pretrain_1/model_0023.model", help='Path of the initial_model')


    # 没有任何其他用处 除了初始化
    parser.add_argument('--m', type=float, default=0.2, help='')
    parser.add_argument('--s', type=float, default=30, help='')

    parser.add_argument('--n_class', type=int, default=5994, help='')

    warnings.simplefilter("ignore")
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parser.parse_args()
    if num_frames is not None:
        args.num_frames = num_frames
    torch.cuda.set_device(args.deviceno)


    trainloader = as_train_loader(**vars(args))
    loader = torch.utils.data.DataLoader(trainloader, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.n_cpu,
                                              drop_last=True, pin_memory=True)
    # model 1
    model = ModelMaster(**vars(args))
    sys.stderr.write("加载模型：{}".format(args.initial_model))
    model.load_parameters(path=args.initial_model, deviceno=args.deviceno)
    model = nn.Sequential(model.wav_pre_process, model.GPT_IConv_NN)

    return model, loader, args.n_class, args.batch_size, args.num_frames


def extract_network(model, loader, n_class, batch_size, num_frames):
    print(time.strftime("%m-%d %H:%M:%S"))
    model.eval()
    # emds_dict = {}
    emds_tensor = torch.zeros(n_class, 192)
    count_label = torch.zeros(n_class, 1).cuda()
    pc = loader.dataset.train_length//batch_size
    for num, (data, labels) in tqdm.tqdm(enumerate(loader, start=1), miniters=20, total=pc):
        labels = labels.cuda()
        data = data.cuda()
        count_label = count_label + torch.bincount(labels, minlength=5994).view(-1, 1)
        # model 1
        # emds_batch = model(data)[0]
        # model 2
        emds_batch = model(data)
        emds_batch = F.normalize(emds_batch, dim=1, p=2)
        emds_tensor[labels] = emds_batch.detach().cpu() + emds_tensor[labels.detach().cpu()]
    emds_tensor = emds_tensor / torch.where(count_label == 0, 1, count_label).to(emds_tensor.device)
    # for i in range(args.n_class):
    #     emds_dict[i] = emds_tensor[i].view(1, -1)
    os.makedirs('exps/as_normal', exist_ok=True)
    # np.save('exps/as_normal/emds_v2_0023.npy', emds_tensor, allow_pickle=True)
    np.save('exps/as_normal/emds_v2_ecapa2_{}s.npy'.format(num_frames/100), emds_tensor, allow_pickle=True)
    print("提取成功！")
    print(time.strftime("%m-%d %H:%M:%S"))


if __name__ == "__main__":
    model, loader, n_class, batch_size, num_frames = get_args(num_frames=200)
    extract_network(model, loader, n_class, batch_size, num_frames)

    model, loader, n_class, batch_size, num_frames = get_args(num_frames=500)
    extract_network(model, loader, n_class, batch_size, num_frames)



