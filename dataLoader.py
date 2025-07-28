'''
DataLoader for training
'''

import glob, numpy as np, os, random, torch, torchaudio
from scipy import signal
import pandas as pd

# 加快时间 保存特征 从而减少计算、io 带来时间截取的固定 类似于seed
class deal_train_loader(object):
	def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
		self.train_path = train_path
		self.num_frames = num_frames
		# Load and configure augmentation files
		self.pertur = True
		self.train_length = 0
		self.pertur1 = torchaudio.transforms.SpeedPerturbation(16000, [0.9])
		self.pertur2 = torchaudio.transforms.SpeedPerturbation(16000, [1.1])
		self.noisetypes = ['noise', 'speech', 'music']
		self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
		self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
		self.noiselist = {}
		augment_files   = glob.glob(os.path.join(musan_path, '*/*/*.wav'))
		for file in augment_files:
			if file.split('/')[-3] not in self.noiselist:
				self.noiselist[file.split('/')[-3]] = []
			self.noiselist[file.split('/')[-3]].append(file)
		self.rir_files  = glob.glob(os.path.join(rir_path, '*/*/*.wav'))
		# Load DAE_data & labels
		self.data_list = []
		self.data_label = []

		# lines = open(train_list).read().splitlines()
		lines = pd.read_csv(train_list, delimiter=' ')
		lines.columns = ['spk', 'utter']
		# group_line = lines.groupby('spk').groups
		# todo 随机取min_group_len
		# 21  500
		min_group_len = min(lines.groupby('spk').transform('size'))
		# max_group_len = max(lines.groupby('spk').transform('size'))

		# 有放回
		group_line = lines.groupby('spk').sample(n=int(min_group_len * 1), replace=False)
		# group_line = lines.groupby('spk').sample(n=int(min_group_len * 1), replace=False)
		dictkeys = list(set(group_line['spk']))
		dictkeys.sort()
		# num111 = 0
		dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
		for idx, row in group_line.iterrows():
			file_name = os.path.join(train_path, row['utter'])
			# if not os.path.exists(file_name):
			# 	num111 += 1
			# 	print("{} : {}".format(num111, len(self.data_label)))
			# 	continue
			self.data_label.append(dictkeys[row['spk']])
			self.data_list.append(file_name)
			if self.pertur:
				self.data_label.append(dictkeys[row['spk']])
				self.data_label.append(dictkeys[row['spk']])
				self.data_list.append("Pertur09" + file_name)
				self.data_list.append("Pertur11" + file_name)
		self.train_length = len(self.data_label)
		print("数据集： {}, 说话人: {}, 语音: {}".format(train_list, len(set(self.data_label)), len(self.data_list)))
	def __getitem__(self, index):
		# Read the utterance and randomly select the segment
		# audio, sr = soundfile.read(self.data_list[index])
		# 做2s特征：

		# audio = np.load(feature_path, allow_pickle=True)
		audio = np.load(self.data_list[index].replace('/', '-'), allow_pickle=True)

		# 原始语音
		# audio = self.add_perturb(self.data_list[index])
		# audio.squeeze_(0)
		# length = self.num_frames * 160 + 240
		# if audio.shape[0] <= length:
		# 	shortage = length - audio.shape[0]
		# 	audio = np.pad(audio, (0, shortage), 'wrap')
		# start_frame = np.int64(random.random()*(audio.shape[0]-length))
		# audio = audio[start_frame:start_frame + length]
		# audio = np.stack([audio], axis=0)
		# # Data Augmentation
		# augtype = random.randint(0, 5)
		# if augtype == 0:   # Original
		# 	audio = audio
		# elif augtype == 1: # Reverberation
		# 	audio = self.add_rev(audio)
		# elif augtype == 2: # Babble
		# 	audio = self.add_noise(audio, 'speech')
		# elif augtype == 3: # Music
		# 	audio = self.add_noise(audio, 'music')
		# elif augtype == 4: # Noise
		# 	audio = self.add_noise(audio, 'noise')
		# elif augtype == 5: # Television noise
		# 	audio = self.add_noise(audio, 'speech')
		# 	audio = self.add_noise(audio, 'music')
		# # elif augtype == 5: # sox t
		# # 	audio = self.add_noise(audio, 'speech')
		# # 	audio = self.add_noise(audio, 'music')
		return torch.FloatTensor(audio[0]), self.data_label[index]

	def __len__(self):
		return len(self.data_list)

	def add_rev(self, audio):
		rir_file    = random.choice(self.rir_files)
		# rir, sr     = soundfile.read(rir_file)
		rir, sr     = torchaudio.load(rir_file)
		rir.squeeze_(0)
		# rir         = np.expand_dims(rir.astype(float), 0)
		rir         = np.expand_dims(rir, 0)
		rir         = rir / np.sqrt(np.sum(rir**2))
		return signal.convolve(audio, rir, mode='full')[:, :self.num_frames * 160 + 240 ]

	def add_noise(self, audio, noisecat):
		clean_db    = 10 * np.log10(np.mean(audio ** 2)+1e-4)
		numnoise    = self.numnoise[noisecat]
		noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
		noises = []
		for noise in noiselist:
			# noiseaudio, sr = soundfile.read(noise)
			noiseaudio, sr = torchaudio.load(noise)
			noiseaudio.squeeze_(0)
			length = self.num_frames * 160 + 240
			if noiseaudio.shape[0] <= length:
				shortage = length - noiseaudio.shape[0]
				noiseaudio = np.pad(noiseaudio, (0, shortage), 'wrap')
			start_frame = np.int64(random.random()*(noiseaudio.shape[0]-length))
			noiseaudio = noiseaudio[start_frame:start_frame + length]
			noiseaudio = np.stack([noiseaudio],axis=0)
			noise_db = 10 * np.log10(np.mean(noiseaudio ** 2)+1e-4)
			noisesnr   = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
			noises.append(np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
		return noise + audio

	def add_perturb(self, audioPath):
		if self.pertur and audioPath.startswith("Pertur09"):
			audio, _ = torchaudio.load(audioPath.split("Pertur09")[-1])
			audio = self.pertur1(audio)[0]

		elif self.pertur and audioPath.startswith("Pertur11"):
			audio, _ = torchaudio.load(audioPath.split("Pertur11")[-1])
			audio = self.pertur2(audio)[0]
		else:
			audio, _ = torchaudio.load(audioPath)
		return audio

class origin_train_loader(object):
	def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
		self.train_path = train_path
		self.num_frames = num_frames
		# Load and configure augmentation files
		self.pertur = False
		self.train_length = 0
		self.pertur1 = torchaudio.transforms.SpeedPerturbation(16000, [0.9])
		self.pertur2 = torchaudio.transforms.SpeedPerturbation(16000, [1.1])
		self.noisetypes = ['noise', 'speech', 'music']
		self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
		self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
		self.noiselist = {}
		augment_files   = glob.glob(os.path.join(musan_path, '*/*/*.wav'))
		for file in augment_files:
			if file.split('/')[-3] not in self.noiselist:
				self.noiselist[file.split('/')[-3]] = []
			self.noiselist[file.split('/')[-3]].append(file)
		self.rir_files  = glob.glob(os.path.join(rir_path, '*/*/*.wav'))
		# Load DAE_data & labels
		self.data_list = []
		self.data_label = []

		# lines = open(train_list).read().splitlines()
		lines = pd.read_csv(train_list, delimiter=' ')
		lines.columns = ['spk', 'utter']
		# group_line = lines.groupby('spk').groups
		# todo 随机取min_group_len
		# 21  500
		min_group_len = min(lines.groupby('spk').transform('size'))
		# max_group_len = max(lines.groupby('spk').transform('size'))

		# 有放回
		group_line = lines.groupby('spk').sample(n=int(min_group_len * 0.1), replace=False)
		dictkeys = list(set(group_line['spk']))
		dictkeys.sort()
		# num111 = 0
		dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
		for idx, row in lines.iterrows():
			file_name = os.path.join(train_path, row['utter'])
			# if not os.path.exists(file_name):
			# 	num111 += 1
			# 	print("{} : {}".format(num111, len(self.data_label)))
			# 	continue
			self.data_label.append(dictkeys[row['spk']])
			self.data_list.append(file_name)
			if self.pertur:
				self.data_label.append(dictkeys[row['spk']])
				self.data_label.append(dictkeys[row['spk']])
				self.data_list.append("Pertur09" + file_name)
				self.data_list.append("Pertur11" + file_name)
		self.train_length = len(self.data_label)
		print("数据集： {}, 说话人: {}, 语音: {}".format(train_list, len(set(self.data_label)), len(self.data_list)))
	def __getitem__(self, index):
		audio = self.add_perturb(self.data_list[index])
		audio.squeeze_(0)
		length = self.num_frames * 160 + 240
		if audio.shape[0] <= length:
			shortage = length - audio.shape[0]
			audio = np.pad(audio, (0, shortage), 'wrap')
		start_frame = np.int64(random.random()*(audio.shape[0]-length))
		audio = audio[start_frame:start_frame + length]
		audio = np.stack([audio], axis=0)
		# Data Augmentation
		augtype = random.randint(0, 5)

		if augtype == 0:   # Original
			audio = audio
		elif augtype == 1: # Reverberation
			audio = self.add_rev(audio)
		elif augtype == 2: # Babble
			audio = self.add_noise(audio, 'speech')
		elif augtype == 3: # Music
			audio = self.add_noise(audio, 'music')
		elif augtype == 4: # Noise
			audio = self.add_noise(audio, 'noise')
		elif augtype == 5: # Television noise
			audio = self.add_noise(audio, 'speech')
			audio = self.add_noise(audio, 'music')
		# elif augtype == 5: # sox t
		# 	audio = self.add_noise(audio, 'speech')
		# 	audio = self.add_noise(audio, 'music')
		return torch.FloatTensor(audio[0]), self.data_label[index]

	def __len__(self):
		return len(self.data_list)

	def add_rev(self, audio):
		rir_file    = random.choice(self.rir_files)
		# rir, sr     = soundfile.read(rir_file)
		rir, sr     = torchaudio.load(rir_file)
		rir.squeeze_(0)
		# rir         = np.expand_dims(rir.astype(float), 0)
		rir         = np.expand_dims(rir, 0)
		rir         = rir / np.sqrt(np.sum(rir**2))
		return signal.convolve(audio, rir, mode='full')[:, :self.num_frames * 160 + 240 ]

	def add_noise(self, audio, noisecat):
		clean_db    = 10 * np.log10(np.mean(audio ** 2)+1e-4)
		numnoise    = self.numnoise[noisecat]
		noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
		noises = []
		for noise in noiselist:
			# noiseaudio, sr = soundfile.read(noise)
			noiseaudio, sr = torchaudio.load(noise)
			noiseaudio.squeeze_(0)
			length = self.num_frames * 160 + 240
			if noiseaudio.shape[0] <= length:
				shortage = length - noiseaudio.shape[0]
				noiseaudio = np.pad(noiseaudio, (0, shortage), 'wrap')
			start_frame = np.int64(random.random()*(noiseaudio.shape[0]-length))
			noiseaudio = noiseaudio[start_frame:start_frame + length]
			noiseaudio = np.stack([noiseaudio],axis=0)
			noise_db = 10 * np.log10(np.mean(noiseaudio ** 2)+1e-4)
			noisesnr   = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
			noises.append(np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
		return noise + audio

	def add_perturb(self, audioPath):
		if self.pertur and audioPath.startswith("Pertur09"):
			audio, _ = torchaudio.load(audioPath.split("Pertur09")[-1])
			audio = self.pertur1(audio)[0]

		elif self.pertur and audioPath.startswith("Pertur11"):
			audio, _ = torchaudio.load(audioPath.split("Pertur11")[-1])
			audio = self.pertur2(audio)[0]
		else:
			audio, _ = torchaudio.load(audioPath)
		return audio
class test_loader(object):
	def __init__(self, eval_list, eval_path, eval_split_num=5, **kwargs):
		self.test_list_path = eval_list
		self.test_path = eval_path
		# 去重复
		self.test_list = self.getData(eval_list)
		self.full_max = 1104641
		# self.full_max = 600000
		# self.full_max = 32240 * 4
		# self.greater_full_max_num = 0
		self.eval_split_num = eval_split_num

#  https://pytorch.org/audio/2.2.0/_modules/torchaudio/datasets/voxceleb1.html
	def __getitem__(self, index):
		# Read the utterance and randomly select the segment
		assert (len(self.test_path) > 0), "没有一个测试集 -- err from yzx"
		file_path_wav = self.test_list[index]
		# audio, _ = soundfile.read(os.path.join(self.test_path, file_path_wav))
		# 多个测试集的测试
		if os.path.exists(os.path.join(self.test_path[0], file_path_wav)):
			audio, _ = torchaudio.load(os.path.join(self.test_path[0], file_path_wav))
		else:
			raise ValueError("不支持数据！")
		audio.squeeze_(0)
		data = []
		fullutter_start_list = []
		full_num = 1
		# 第一种方式  full utter
		if audio.shape[0] > self.full_max:
			tmp_full_max = audio.shape[0]
			# print("before max:", self.full_max)
			# self.greater_full_max_num = self.greater_full_max_num + 1
			# print("now max: {:d}, greater_full_max:{:d}".format(self.tmp_full_max, self.greater_full_max_num))
			start = tmp_full_max - self.full_max
			start = np.int64(random.random() * start)
			data.append(audio[start:start+self.full_max])
			fullutter_start_list.append(0)
			# data.append(audio[0:self.full_max])
		elif audio.shape[0] <= self.full_max:
			start_frame = self.full_max - audio.shape[0]
			fullutter_start_list.append(start_frame)
			data.append(np.pad(audio, (start_frame, 0)))
		else:
			raise TypeError("暂时不支持！！！ -- err from yzx")

		# 第二种方式   seperate 5 vector 定长
		max_audio = 300 * 160 + 240
		if audio.shape[0] <= max_audio * self.eval_split_num:
			shortage = max_audio * self.eval_split_num - audio.shape[0]
			audio = torch.tensor(np.pad(audio, (0, shortage), 'wrap'))
			startframe = np.linspace(0,  4 * max_audio, num=self.eval_split_num)
			for asf in startframe:
				data.append(audio[int(asf):int(asf) + max_audio])
		elif audio.shape[0] > max_audio * self.eval_split_num:
			start = np.int64(random.random() * (audio.shape[0] - self.eval_split_num * max_audio))
			startframe = np.linspace(start, start + 4 * max_audio, num=self.eval_split_num)
			for asf in startframe:
				data.append(audio[int(asf):int(asf) + max_audio])
		# return np.asarray(data[0]), np.asarray(data[1]), [int(label)]*6
		return np.stack(data[0:full_num], axis=0), np.stack(data[full_num:], axis=0), self.test_list[index], np.stack(fullutter_start_list)

	def __len__(self):
		return len(self.test_list)

	def getData(self, eval_list):
		data_list = []
		for i in range(len(eval_list)):
			files = []
			lines = open(eval_list[i]).read().splitlines()
			for line in lines:
				files.append(line.split()[1])
				files.append(line.split()[2])
			tmp_data_list = list(set(files))
			if i > 0:
				data_list = data_list + tmp_data_list
				data_list = list(set(data_list))
			elif i == 0:
				data_list = tmp_data_list
		return data_list


# test1.wav test2.wav 0
# 1.放入testLoader 2.按照顺序不去重复
class test_loader2(object):
	def __init__(self, eval_list, eval_path, **kwargs):
		self.test_path = eval_path
		# Load and configure augmentation files
		self.test_list = open(eval_list).read().splitlines()

#  https://pytorch.org/audio/2.2.0/_modules/torchaudio/datasets/voxceleb1.html
	def __getitem__(self, index):
		# Read the utterance and randomly select the segment
		label, file_path_spk1, file_path_spk2 = self.test_list[index].split()
		# audio1, _ = soundfile.read(os.path.join(self.test_path, file_path_spk1))
		# audio2, _ = soundfile.read(os.path.join(self.test_path, file_path_spk2))
		audio1, _ = torchaudio.load(os.path.join(self.test_path, file_path_spk1))
		audio2, _ = torchaudio.load(os.path.join(self.test_path, file_path_spk2))
		audio1.squeeze_(0)
		audio2.squeeze_(0)

		# audio1 : Full utterance  and Spliited utterance matrix [[0],[1...5]]
		audio = [audio1, audio2]
		data = [[], []]
		#
		for i in range(len(audio)):
		# Spliited utterance matrix
		# 	max_audio = 300 * 160 + 240
			max_audio = 200 * 160 + 240
			if audio[i].shape[0] <= max_audio:
				shortage = max_audio - audio[i].shape[0]
				data[i].append(np.pad(audio[i], (0, shortage), 'wrap'))
			elif audio[i].shape[0] > max_audio:
				start = np.int64(random.random() * (audio[i].shape[0] - max_audio))
				data[i].append(audio[i][int(start):int(start) + max_audio])
			# seperate 5 vector

			if audio[i].shape[0] <= max_audio * 5:
				shortage = max_audio * 5 - audio[i].shape[0]
				audio[i] = np.pad(audio[i], (0, shortage), 'wrap')
				startframe = np.linspace(0,  4 * max_audio, num=5)
				for asf in startframe:
					data[i].append(audio[i][int(asf):int(asf) + max_audio])
			elif audio[i].shape[0] > max_audio * 5:
				start = np.int64(random.random() * (audio[i].shape[0] - 5 * max_audio))
				startframe = np.linspace(start, start + 4 * max_audio, num=5)
				for asf in startframe:
					data[i].append(audio[i][int(asf):int(asf) + max_audio])
		# return np.asarray(data[0]), np.asarray(data[1]), [int(label)]*6
		return torch.FloatTensor(data[0]), torch.FloatTensor(data[1]), torch.IntTensor([int(label)])

	def __len__(self):
		return len(self.test_list)


class wav_loader_npy(object):
	def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
		self.train_path = train_path
		self.num_frames = num_frames
		self.train_length = 0
		self.pertur1 = torchaudio.transforms.SpeedPerturbation(16000, [0.9])
		self.pertur2 = torchaudio.transforms.SpeedPerturbation(16000, [1.1])
		self.noisetypes = ['noise', 'speech', 'music']
		self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
		self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
		self.noiselist = {}
		augment_files   = glob.glob(os.path.join(musan_path, '*/*/*.wav'))
		for file in augment_files:
			if file.split('/')[-3] not in self.noiselist:
				self.noiselist[file.split('/')[-3]] = []
			self.noiselist[file.split('/')[-3]].append(file)
		self.rir_files  = glob.glob(os.path.join(rir_path, '*/*/*.wav'))
		# Load DAE_data & labels
		self.data_list = []
		self.data_label = []

		lines = pd.read_csv(train_list, delimiter=' ')
		lines.columns = ['spk', 'utter']
		min_group_len = min(lines.groupby('spk').transform('size'))

		group_line = lines.groupby('spk').sample(n=int(min_group_len * 0.05), replace=False)
		dictkeys = list(set(group_line['spk']))
		dictkeys.sort()
		dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
		for idx, row in lines.iterrows():
			# if not os.path.exists(os.path.join(train_path, row['utter'])):
			# 	continue
			file_name = os.path.join(train_path, row['utter'])
			self.data_label.append(dictkeys[row['spk']])
			self.data_list.append(file_name)
		self.train_length = len(self.data_label)
		print("数据集： {}, 说话人: {}, 语音: {}".format(train_list, len(set(self.data_label)), len(self.data_list)))

	def __getitem__(self, index):
		audio, _ = torchaudio.load(self.data_list[index])
		audio.squeeze_(0)
		length = self.num_frames * 160 + 240
		if audio.shape[0] <= length:
			shortage = length - audio.shape[0]
			audio = np.pad(audio, (0, shortage), 'wrap')
		start_frame = np.int64(random.random() * (audio.shape[0] - length))
		audio = audio[start_frame:start_frame + length]
		audio = np.stack([audio], axis=0)

		augtype = 0
		if augtype == 0:  # Original
			audio = audio
		elif augtype == 1:  # Reverberation
			audio = self.add_rev(audio)
		elif augtype == 2:  # Babble
			audio = self.add_noise(audio, 'speech')
		elif augtype == 3:  # Music
			audio = self.add_noise(audio, 'music')
		elif augtype == 4:  # Noise
			audio = self.add_noise(audio, 'noise')
		elif augtype == 5:  # Television noise
			audio = self.add_noise(audio, 'speech')
			audio = self.add_noise(audio, 'music')
		# elif augtype == 5: # sox t
		# 	audio = self.add_noise(audio, 'speech')
		# 	audio = self.add_noise(audio, 'music')
		# return torch.FloatTensor(audio[0]), self.data_label[index]

		return torch.FloatTensor(audio[0]), self.data_label[index], self.data_list[index].split('/', 6)[-1]

	def __len__(self):
		return len(self.data_list)

	def add_rev(self, audio):
		rir_file = random.choice(self.rir_files)
		# rir, sr     = soundfile.read(rir_file)
		rir, sr = torchaudio.load(rir_file)
		rir.squeeze_(0)
		# rir         = np.expand_dims(rir.astype(float), 0)
		rir = np.expand_dims(rir, 0)
		rir = rir / np.sqrt(np.sum(rir ** 2))
		return signal.convolve(audio, rir, mode='full')[:, :self.num_frames * 160 + 240]

	def add_noise(self, audio, noisecat):
		clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)
		numnoise = self.numnoise[noisecat]
		noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
		noises = []
		for noise in noiselist:
			# noiseaudio, sr = soundfile.read(noise)
			noiseaudio, sr = torchaudio.load(noise)
			noiseaudio.squeeze_(0)
			length = self.num_frames * 160 + 240
			if noiseaudio.shape[0] <= length:
				shortage = length - noiseaudio.shape[0]
				noiseaudio = np.pad(noiseaudio, (0, shortage), 'wrap')
			start_frame = np.int64(random.random() * (noiseaudio.shape[0] - length))
			noiseaudio = noiseaudio[start_frame:start_frame + length]
			noiseaudio = np.stack([noiseaudio], axis=0)
			noise_db = 10 * np.log10(np.mean(noiseaudio ** 2) + 1e-4)
			noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
			noises.append(np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
		return noise + audio

	def add_perturb(self, audioPath):
		if self.pertur and audioPath.startswith("Pertur09"):
			audio, _ = torchaudio.load(audioPath.split("Pertur09")[-1])
			audio = self.pertur1(audio)[0]

		elif self.pertur and audioPath.startswith("Pertur11"):
			audio, _ = torchaudio.load(audioPath.split("Pertur11")[-1])
			audio = self.pertur2(audio)[0]
		else:
			audio, _ = torchaudio.load(audioPath)
		return audio