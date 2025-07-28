import argparse, os, torch, warnings, time, torch.nn as nn
import tqdm
from model import wav_pre_process_fbank_extract
import numpy as np
import torch.nn.functional as F
from dataLoader import wav_loader_npy


def get_args(num_frames=None):
    parser = argparse.ArgumentParser(description="New_Ecapa_trainer")
    parser.add_argument('--n', type=int, default=1, help='Number of order/head? num')

    parser.add_argument('--batch_size', type=int, default=50, help='Batch size  300  try 64 128')
    parser.add_argument('--n_cpu', type=int, default=10, help='Number of train threads 15')
    parser.add_argument('--deviceno', type=int, default=0, help='device NUM')
    parser.add_argument('--C', type=int, default=512, help='新网络专用的通道数 512')
    parser.add_argument('--train_list', type=str, default="exps/vox2_train.txt",
                        help='The path of the training list, https://www.robots.ox.ac.cuk/~vgg/data/voxceleb/meta/train_list.txt')
    # parser.add_argument('--train_path', type=str, default="../vox2_dev_wav",
    parser.add_argument('--train_path', type=str, default="/g813_u1/wmnt/voxceleb/vox2_dev_wav/wav",
                        help='The path of the training DAE_data, eg:"/data08/VoxCeleb2/train/wav" in my case')
    parser.add_argument('--num_frames', type=int, default=200,
                        help='Duration of the input segments, eg: 200 for 2 second')
    parser.add_argument('--n_class', type=int, default=5994, help='')
    parser.add_argument('--feature_path', type=str, default="../yzx_mask_Bfeature_2s",
                        help='')

    warnings.simplefilter("ignore")
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parser.parse_args()
    if num_frames is not None:
        args.num_frames = num_frames
    torch.cuda.set_device(args.deviceno)


    wavloader = wav_loader_npy(**vars(args))
    loader = torch.utils.data.DataLoader(wavloader, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.n_cpu,
                                              drop_last=True, pin_memory=True)
    # model 1
    # model = ModelMaster(**vars(args))
    # sys.stderr.write("加载模型：{}".format(args.initial_model))
    # model.load_parameters(path=args.initial_model, deviceno=args.deviceno)
    # model = nn.Sequential(model.wav_pre_process, model.GPT_IConv_NN)

    # model ecapa2
    model = nn.Sequential(wav_pre_process_fbank_extract(),).cuda()
    return model, loader, args.n_class, args.batch_size, args.num_frames, args.feature_path


def create_yzx(model=None, loader=None, feature_path=None, num_frames=None):
    print(time.strftime("%m-%d %H:%M:%S"))
    # model.eval()
    feature_path = feature_path.replace('2', str(num_frames//100))
    os.makedirs(feature_path, exist_ok=True)
    pc = loader.dataset.train_length // batch_size
    for num, (data, labels, name) in tqdm.tqdm(enumerate(loader, start=1), miniters=20, total=pc):
        emb = model(data.cuda())
        for i in range(len(labels)):
            np.save(os.path.join(feature_path, name[i].replace('/', '') + '.npy'), emb[i].cpu(), allow_pickle=True)
    print("提取语音数据成功！")
    print(time.strftime("%m-%d %H:%M:%S"))


if __name__ == "__main__":
    model, loader, n_class, batch_size, num_frames, feature_path = get_args(num_frames=200)
    create_yzx(model=model, loader=loader,  feature_path=feature_path, num_frames=num_frames)




