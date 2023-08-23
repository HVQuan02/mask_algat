import os
import sys
import numpy as np
import csv
from torch.utils.data import Dataset


class FCVID(Dataset):
    NUM_CLASS = 239
    NUM_FRAMES = 9
    NUM_BOXES = 50

    def __init__(self, root_dir, is_train, ext_method):
        self.root_dir = root_dir
        self.phase = 'train' if is_train else 'test'
        if ext_method == 'VIT':
            self.local_folder = 'vit_local'
            self.global_folder = 'vit_global'
            self.NUM_FEATS = 768
        elif ext_method == 'RESNET':
            self.local_folder = 'R152_local'
            self.global_folder = 'R152_global'
            self.NUM_FEATS = 2048
        else:
            sys.exit("Unknown Extractor")

        split_path = os.path.join(root_dir, 'materials', 'FCVID_VideoName_TrainTestSplit.txt')
        data_split = np.genfromtxt(split_path, dtype='str')

        label_path = os.path.join(root_dir, 'materials', 'FCVID_Label.txt')
        labels = np.genfromtxt(label_path, dtype=np.float32)

        self.num_missing = 0
        mask = np.zeros(data_split.shape[0], dtype=bool)
        for i, row in enumerate(data_split):
            if row[1] == self.phase:
                base, _ = os.path.splitext(os.path.normpath(row[0]))
                feats_path = os.path.join(root_dir, self.local_folder, base + '.npy')
                if os.path.exists(feats_path):
                    mask[i] = 1
                else:
                    self.num_missing += 1

        self.labels = labels[mask, :]
        self.videos = data_split[mask, 0]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        name = self.videos[idx]
        name, _ = os.path.splitext(name)

        feats_path = os.path.join(self.root_dir, self.local_folder, name + '.npy')
        global_path = os.path.join(self.root_dir, self.global_folder, name + '.npy')
        feats = np.load(feats_path)
        feat_global = np.load(global_path)
        label = self.labels[idx, :]

        return feats, feat_global, label, name


class ACTNET(Dataset):
    NUM_CLASS = 200
    NUM_FRAMES = 120
    NUM_BOXES = 50

    def __init__(self, root_dir, is_train, ext_method):
        self.root_dir = root_dir
        self.phase = 'train' if is_train else 'test'
        if ext_method == 'CLIP':
            self.local_folder = 'feats_clip/clip_local'
            self.global_folder = 'feats_clip/clip_global'
            self.NUM_FEATS = 1024
        elif ext_method == 'VIT':
            self.local_folder = 'feats/vit_local'
            self.global_folder = 'feats/vit_global'
            self.NUM_FEATS = 768
        else:
            sys.exit("Unknown Extractor")

        if self.phase == 'train':
            split_path = os.path.join(root_dir, 'actnet_train_split.txt')
        else:
            split_path = os.path.join(root_dir, 'actnet_val_split.txt')
        self.num_missing = 0
        vidname_list = []
        labels_list = []
        with open(split_path) as f:
            for line in f:
                row = line.strip().split(',')
                feats_path = os.path.join(self.root_dir, self.local_folder, row[0] + '.npy')
                if os.path.exists(feats_path):
                    vidname_list.append(row[0])
                    labels_list.append(list(map(int, row[2:])))
                else:
                    self.num_missing += 1

        length = len(vidname_list)
        labels_np = np.zeros((length, self.NUM_CLASS), dtype=np.float32)
        for i, lbllst in enumerate(labels_list):
            for lbl in lbllst:
                labels_np[i, lbl] = 1.

        self.labels = labels_np
        self.videos = vidname_list


    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        name = self.videos[idx]
        # name, _ = os.path.splitext(name)

        feats_path = os.path.join(self.root_dir, self.local_folder, name + '.npy')  #
        global_path = os.path.join(self.root_dir, self.global_folder, name + '.npy')  #
        feats = np.load(feats_path)
        feat_global = np.load(global_path)
        label = self.labels[idx, :]

        return feats, feat_global, label, name


class miniKINETICS(Dataset):
    NUM_CLASS = 200
    NUM_FRAMES = 30
    NUM_BOXES = 50

    def __init__(self, root_dir, is_train, ext_method):
        self.root_dir = root_dir
        self.phase = 'train' if is_train else 'test'
        if ext_method == 'CLIP':
            self.local_folder = 'feats_val/clip_local'
            self.global_folder = 'feats_val/clip_global'
            self.NUM_FEATS = 1024
        elif ext_method == 'VIT':
            self.local_folder = 'feats/vit_local'
            self.global_folder = 'feats/vit_global'
            self.NUM_FEATS = 768
        else:
            sys.exit("Unknown Extractor")

        if self.phase == 'train':
            split_path = os.path.join(root_dir, 'annotations', 'miniKinetics130trainv2.csv')
        else:
            split_path = os.path.join(root_dir, 'annotations', 'miniKinetics130valv2.csv')

        vidname_list = []
        labels_list = []
        self.num_missing = 0

        with open(split_path) as f:
            file = csv.reader(f)
            header = []
            header = next(file)
            if self.phase == 'train':
                mask = np.zeros(121215, dtype=bool)
            else:
                mask = np.zeros(9867, dtype=bool)
            for i, row in enumerate(file):
                base = row[1] + '_' + row[2].zfill(6) + '_' + row[3].zfill(6) + '_frames'
                vidname_list.append(base)
                labels_list.append(list(map(int, [row[0]])))
                feats_path = os.path.join(root_dir, self.local_folder, base + '.npy')
                if os.path.exists(feats_path):
                    mask[i] = 1
                else:
                    self.num_missing += 1
        self.labels = np.array(labels_list, dtype=np.int64).squeeze()[mask]   # , :]
        self.videos = np.array(vidname_list)[mask]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        name = self.videos[idx]
        # name, _ = os.path.splitext(name)

        feats_path = os.path.join(self.root_dir, self.local_folder, name + '.npy')  #
        global_path = os.path.join(self.root_dir, self.global_folder, name + '.npy')  #
        feats = np.load(feats_path)
        feat_global = np.load(global_path)
        label = self.labels[idx]

        return feats, feat_global, label, name


class YLIMED(Dataset):
    NUM_CLASS = 10
    NUM_FRAMES = 9
    NUM_BOXES = 50

    def __init__(self, root_dir, is_train, ext_method):
        self.root_dir = root_dir
        self.phase = 'Training' if is_train else 'Test'
        if ext_method == 'VIT':
            self.local_folder = 'feats_vit/vit_local'
            self.global_folder = 'feats_vit/vit_global'
            self.NUM_FEATS = 768
        elif ext_method == 'CLIP':
            self.local_folder = 'feats_clip/clip_local'
            self.global_folder = 'feats_clip/clip_global'
            self.NUM_FEATS = 1024
        else:
            sys.exit("Unknown Extractor")

        split_path = os.path.join(root_dir, 'YLI-MED_Corpus_v.1.4.txt')
        data_split = np.genfromtxt(split_path, dtype='str', skip_header=1)

        self.num_missing = 0
        mask = np.zeros(data_split.shape[0], dtype=bool)
        for i, row in enumerate(data_split):
            if row[7] == 'Ev100':
                continue

            if row[13] == self.phase:
                feats_path = os.path.join(root_dir, self.local_folder, row[0] + '.npy')
                if os.path.exists(feats_path):
                    mask[i] = 1
                else:
                    self.num_missing += 1

        self.videos = data_split[mask, 0]
        labels = [int(x[3:]) - 1 for x in data_split[mask, 7]]
        self.labels = np.array(labels, dtype=np.int32)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        name = self.videos[idx]
        name, _ = os.path.splitext(name)

        feats_path = os.path.join(self.root_dir, self.local_folder, name + '.npy')
        global_path = os.path.join(self.root_dir, self.global_folder, name + '.npy')

        feats = np.load(feats_path)
        feat_global = np.load(global_path)
        label = np.int64(self.labels[idx])

        return feats, feat_global, label, name


class YLIMED_tokens(Dataset):
    NUM_CLASS = 10
    NUM_FRAMES = 9
    NUM_BOXES = 50

    def __init__(self, root_dir, is_train, ext_method):
        self.root_dir = root_dir
        self.phase = 'Training' if is_train else 'Test'
        self.clip_folder = 'feats_clip/vit_local'
        self.token_folder = 'tokens_new/local'
        self.NUM_FEATS = 1024

        split_path = os.path.join(root_dir, 'YLI-MED_Corpus_v.1.4.txt')
        data_split = np.genfromtxt(split_path, dtype='str', skip_header=1)

        self.num_missing = 0
        mask = np.zeros(data_split.shape[0], dtype=bool)
        for i, row in enumerate(data_split):
            if row[7] == 'Ev100':
                continue

            if row[13] == self.phase:
                feats_path = os.path.join(root_dir, self.clip_folder, row[0] + '.npy')
                if os.path.exists(feats_path):
                    mask[i] = 1
                else:
                    self.num_missing += 1

        self.videos = data_split[mask, 0]
        labels = [int(x[3:]) - 1 for x in data_split[mask, 7]]
        self.labels = np.array(labels, dtype=np.int32)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        name = self.videos[idx]
        name, _ = os.path.splitext(name)

        feats_path = os.path.join(self.root_dir, self.clip_folder, name + '.npy')
        token_path = os.path.join(self.root_dir, self.token_folder, name + '.npy')

        feats = np.load(feats_path)
        tokens = np.load(token_path)

        # label = np.int64(self.labels[idx])

        return feats, tokens, name


class ACTNET_tokens(Dataset):
    NUM_CLASS = 200
    NUM_FRAMES = 120
    NUM_BOXES = 50

    def __init__(self, root_dir, is_train, ext_method):
        self.root_dir = root_dir
        self.phase = 'train' if is_train else 'test'

        self.clip_folder = 'clip_feats/clip_local'
        self.token_folder = 'tokens/local'
        self.NUM_FEATS = 1024

        if self.phase == 'train':
            split_path = os.path.join(root_dir, 'actnet_train_split.txt')
        else:
            split_path = os.path.join(root_dir, 'actnet_val_split.txt')
        self.num_missing = 0

        vidname_list = []
        labels_list = []
        with open(split_path) as f:
            for line in f:
                row = line.strip().split(',')
                feats_path = os.path.join(self.root_dir, self.clip_folder, row[0] + '.npy')
                if os.path.exists(feats_path):
                    vidname_list.append(row[0])
                    labels_list.append(list(map(int, row[2:])))
                else:
                    self.num_missing += 1

        length = len(vidname_list)
        labels_np = np.zeros((length, self.NUM_CLASS), dtype=np.float32)
        for i, lbllst in enumerate(labels_list):
            for lbl in lbllst:
                labels_np[i, lbl] = 1.

        self.labels = labels_np
        self.videos = vidname_list


    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        name = self.videos[idx]
        # name, _ = os.path.splitext(name)

        feats_path = os.path.join(self.root_dir, self.clip_folder, name + '.npy')
        token_path = os.path.join(self.root_dir, self.token_folder, name + '.npz')

        feats = np.load(feats_path)
        tokens = np.load(token_path)['arr_0']

        return feats, tokens, name

class miniKinetics_tokens(Dataset):
    NUM_CLASS = 200
    NUM_FRAMES = 30
    NUM_BOXES = 50

    def __init__(self, root_dir, is_train, ext_method):
        self.root_dir = root_dir
        self.phase = 'train' if is_train else 'test'

        self.clip_folder = 'feats/clip_local'
        self.token_folder = 'tokens/local'
        self.NUM_FEATS = 1024

        if self.phase == 'train':
            split_path = os.path.join(root_dir, 'annotations', 'miniKinetics130trainv2.csv')
        else:
            split_path = os.path.join(root_dir, 'annotations', 'miniKinetics130valv2.csv')

        vidname_list = []
        labels_list = []
        self.num_missing = 0

        with open(split_path) as f:
            file = csv.reader(f)
            header = []
            header = next(file)
            if self.phase == 'train':
                mask = np.zeros(121215, dtype=bool)
            else:
                mask = np.zeros(9867, dtype=bool)
            for i, row in enumerate(file):
                base = row[1] + '_' + row[2].zfill(6) + '_' + row[3].zfill(6) + '_frames'
                vidname_list.append(base)
                labels_list.append(list(map(int, [row[0]])))
                feats_path = os.path.join(root_dir, self.token_folder, base + '.npz')
                if os.path.exists(feats_path):
                    mask[i] = 1
                else:
                    self.num_missing += 1
        self.labels = np.array(labels_list, dtype=np.int64).squeeze()[mask]  # , :]
        self.videos = np.array(vidname_list)[mask]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        name = self.videos[idx]
        # name, _ = os.path.splitext(name)

        feats_path = os.path.join(self.root_dir, self.clip_folder, name + '.npy')
        token_path = os.path.join(self.root_dir, self.token_folder, name + '.npz')

        feats = np.load(feats_path)
        tokens = np.load(token_path)['arr_0']
        if feats.shape[1] != 50:
            print()
        return feats, tokens, name