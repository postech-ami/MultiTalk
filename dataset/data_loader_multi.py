import os
import pdb

import torch
import numpy as np
import pickle
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa
from collections import defaultdict
from torch.utils import data


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, subjects_dict, data_type="train", read_audio=False):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        #self.one_hot_labels = np.eye(1)
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        self.read_audio = read_audio

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        if self.data_type == "train":
            if len(self.one_hot_labels)==1:
                one_hot = self.one_hot_labels[0]
            else:
            #subject = "_".join(file_name.split("_")[:-1])
                subject = file_name.split("_")[0]
                one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject.capitalize())]

        else:
            #one_hot = self.one_hot_labels
            if len(self.one_hot_labels)==1:
                one_hot = self.one_hot_labels[0]
            else:
                subject = file_name.split("_")[0]
                one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject.capitalize())]

        if self.read_audio:
            return torch.FloatTensor(audio), torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name
        else:
            return torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

    def __len__(self):
        return self.len

def read_data(args, test_config=False):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(args.data_root, args.wav_path)
    vertices_path = os.path.join(args.data_root, args.vertices_path)
    if args.read_audio:  # read_audio==False when training vq to save time
        processor = Wav2Vec2Processor.from_pretrained(args.wav2vec2model_path)

    template_file = os.path.join(args.data_root, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    cnt=0

    ####spliting train, val, test
    train_txt = open(os.path.join(args.data_root,"train.txt"), "r")
    test_txt = open(os.path.join(args.data_root,"test.txt"), "r")
    train_lines, test_lines, train_list, test_list = train_txt.readlines(), test_txt.readlines(), [], []
    for tt in train_lines:
        train_list.append(tt.split("\n")[0])
    for tt in test_lines:
        test_list.append(tt.split("\n")[0])

    for r, ds, fs in os.walk(audio_path):

        for f in tqdm(fs):
            ###Activate when testing the model
            if test_config and f not in test_list:
                continue

            if f.endswith("wav"):
                if args.read_audio:
                    wav_path = os.path.join(r, f)
                    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                    input_values = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
                key = f.replace("wav", "npy")
                data[key]["audio"] = input_values if args.read_audio else None
                subject_id = "_".join(key.split("_")[:-1])
                #temp = templates[subject_id]
                temp = templates["id"]

                data[key]["name"] = f
                data[key]["template"] = temp.reshape((-1))

                vertice_path = os.path.join(vertices_path, f.replace("wav", "npz"))

                if not os.path.exists(vertice_path):
                    del data[key]
                else:
                    if args.dataset == "vocaset":
                        data[key]["vertice"] = np.load(vertice_path, allow_pickle=True)[::2,
                                               :]  # due to the memory limit
                    elif args.dataset == "BIWI":
                        data[key]["vertice"] = np.load(vertice_path, allow_pickle=True)
                    elif args.dataset=="multi":
                        flame_param = np.load(vertice_path, allow_pickle=True)
                        data[key]["vertice"] = flame_param["verts"].reshape((flame_param["verts"].shape[0], -1))

    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]

    # train vq and pred
    train_cnt = 0
    for k, v in data.items():
        k_wav = k.replace("npy", "wav")
        if k_wav in train_list:
            if train_cnt<int(len(train_list)*0.9):
                train_data.append(v)
            else:
                valid_data.append(v)
            train_cnt+=1
        elif k_wav in test_list:
            test_data.append(v)

    print('Loaded data: Train-{}, Val-{}, Test-{}'.format(len(train_data), len(valid_data), len(test_data)))
    return train_data, valid_data, test_data, subjects_dict


def get_dataloaders(args, test_config=False):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args, test_config)

    if not test_config:
        train_data = Dataset(train_data, subjects_dict, "train", args.read_audio)
        dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers)
        valid_data = Dataset(valid_data, subjects_dict, "val", args.read_audio)
        dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False, num_workers=args.workers)
    test_data = Dataset(test_data, subjects_dict, "test", args.read_audio)
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=args.workers)
    return dataset


if __name__ == "__main__":
    get_dataloaders()