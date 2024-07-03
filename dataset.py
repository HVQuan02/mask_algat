import os
import json
import numpy as np
from torch.utils.data import Dataset


class CUFED(Dataset):
    NUM_CLASS = 23
    NUM_FRAMES = 30
    NUM_BOXES = 50
    NUM_FEATS = 1024
    event_labels = ['Architecture', 'BeachTrip', 'Birthday', 'BusinessActivity',
                    'CasualFamilyGather', 'Christmas', 'Cruise', 'Graduation',
                    'GroupActivity', 'Halloween', 'Museum', 'NatureTrip',
                    'PersonalArtActivity', 'PersonalMusicActivity', 'PersonalSports',
                    'Protest', 'ReligiousActivity', 'Show', 'Sports', 'ThemePark',
                    'UrbanTrip', 'Wedding', 'Zoo']

    def get_album_importance(self, album_imgs, album_importance):
        img_to_score = {}
        for _, image, score in album_importance:
            img_to_score[image.split('/')[1]] = score
        importance = np.zeros(len(album_imgs))
        for i, image in enumerate(album_imgs):
            importance[i] = img_to_score[image[:-4]]
        return importance

    def __init__(self, root_dir, feats_dir, split_dir, is_train=True):
        self.root_dir = root_dir
        self.feats_dir = feats_dir

        self.local_folder = 'clip_local'
        self.global_folder = 'clip_global'
        
        if is_train:
            self.phase = 'train'
        else:
            self.phase = 'test'

        if self.phase == 'train':
            split_path = os.path.join(split_dir, 'train_split.txt')
        else:
            split_path = os.path.join(split_dir, 'test_split.txt')

        with open(split_path, 'r') as f:
            album_names = f.readlines()
        vidname_list = [name.strip() for name in album_names]

        label_path = os.path.join(root_dir, "event_type.json")
        with open(label_path, 'r') as f:
          album_data = json.load(f)

        labels_np = np.zeros((len(vidname_list), self.NUM_CLASS), dtype=np.float32)
        for i, vidname in enumerate(vidname_list):
            for lbl in album_data[vidname]:
                idx = self.event_labels.index(lbl)
                labels_np[i, idx] = 1

        self.videos = vidname_list
        self.labels = labels_np

        if self.phase == 'test':
            importance_path = os.path.join(root_dir, "image_importance.json")
            with open(importance_path, 'r') as f:
                album_importance = json.load(f)

            album_imgs_path = os.path.join(split_dir, "album_imgs_mask.json")
            with open(album_imgs_path, 'r') as f:
                album_imgs = json.load(f)
                
            self.importance = album_importance
            self.album_imgs = album_imgs

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        name = self.videos[idx]

        local_path = os.path.join(self.feats_dir, self.local_folder, name + '.npy')
        global_path = os.path.join(self.feats_dir, self.global_folder, name + '.npy')
        feat_local = np.load(local_path)
        feat_global = np.load(global_path)
        label = self.labels[idx, :]

        if self.phase == 'test':
            album_imgs = self.album_imgs[name]
            album_importance = self.importance[name]
            importance = self.get_album_importance(album_imgs, album_importance)
            return feat_local, feat_global, label, importance
        
        return feat_local, feat_global, label


class PEC(Dataset):
    NUM_CLASS = 14
    NUM_FRAMES = 30
    NUM_BOXES = 50
    NUM_FEATS = 1024
    event_labels = ['birthday', 'children_birthday', 'christmas', 'concert', 'cruise', 'easter', 'exhibition', 'graduation', 'halloween', 'hiking', 'road_trip', 'saint_patricks_day', 'skiing', 'wedding']
    lbl_to_idx = {'birthday': 0, 'children_birthday': 1, 'christmas': 2, 'concert': 3, 'cruise': 4, 'easter': 5, 'exhibition': 6, 'graduation': 7, 'halloween': 8, 'hiking': 9, 'road_trip': 10, 'saint_patricks_day': 11, 'skiing': 12, 'wedding': 13}

    def __init__(self, root_dir, feats_dir, split_dir, is_train=True):
        self.root_dir = root_dir
        self.feats_dir = feats_dir
        self.split_dir = split_dir

        self.local_dir = 'clip_local'
        self.global_dir = 'clip_global'

        if is_train:
            self.phase = 'train'
        else:
            self.phase = 'test'

        if self.phase == 'train':
            split_path = os.path.join(self.split_dir, 'train.txt')
        else:
            split_path = os.path.join(self.split_dir, 'test.txt')
        
        with open(split_path, 'r') as f:
            lines = f.readlines()
        label_albums = [line.strip() for line in lines]
        
        albums = []
        lbl_oh = np.zeros((len(label_albums), self.NUM_CLASS), dtype=np.float32)
        
        for i, label_album in enumerate(label_albums):
            label, album = label_album.split('/')
            lbl_oh[i][self.lbl_to_idx[label]] = 1
            albums.append(album)
            
        self.albums = albums
        self.labels = lbl_oh
        
    def __len__(self):
        return len(self.albums)
    
    def __getitem__(self, idx):
        album = self.albums[idx]
        
        feat_local = np.load(os.path.join(self.feats_dir, self.local_dir, album + '.npy'))
        feat_global = np.load(os.path.join(self.feats_dir, self.global_dir, album + '.npy'))
        label = self.labels[idx]

        return feat_local, feat_global, label
    
    
class CUFED_Tokens(Dataset):
    NUM_CLASS = 23
    NUM_FRAMES = 30
    NUM_BOXES = 50
    NUM_FEATS = 1024
    TOKEN_SIZE = 8192
    event_labels = ['birthday', 'children_birthday', 'christmas', 'concert', 'cruise', 'easter', 'exhibition', 'graduation', 'halloween', 'hiking', 'road_trip', 'saint_patricks_day', 'skiing', 'wedding']
    lbl_to_idx = {'birthday': 0, 'children_birthday': 1, 'christmas': 2, 'concert': 3, 'cruise': 4, 'easter': 5, 'exhibition': 6, 'graduation': 7, 'halloween': 8, 'hiking': 9, 'road_trip': 10, 'saint_patricks_day': 11, 'skiing': 12, 'wedding': 13}

    def __init__(self, root_dir, feats_dir, split_dir, is_train=True):
        self.root_dir = root_dir
        self.feats_dir = feats_dir

        self.local_dir = 'clip_local'
        self.global_dir = 'clip_global'
        self.token_dir = 'token'
        
        if is_train:
            self.phase = 'train'
        else:
            self.phase = 'test'
            
        if self.phase == 'train':
            split_path = os.path.join(split_dir, 'train_split.txt')
        else:
            split_path = os.path.join(split_dir, 'test_split.txt')

        with open(split_path, 'r') as f:
            album_names = f.readlines()
        vidname_list = [name.strip() for name in album_names]
        
        self.videos = vidname_list

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        name = self.videos[idx]

        local_path = os.path.join(self.feats_dir, self.local_dir, name + '.npy')
        global_path = os.path.join(self.feats_dir, self.global_dir, name + '.npy')
        token_path = os.path.join(self.feats_dir, self.token_dir, name + '.npy')

        local_feat = np.load(local_path)
        global_feat = np.load(global_path)
        token = np.load(token_path)

        return local_feat, global_feat, token
    

class PEC_Tokens(Dataset):
    NUM_CLASS = 14
    NUM_FRAMES = 30
    NUM_BOXES = 50
    NUM_FEATS = 1024
    TOKEN_SIZE = 8192
    event_labels = ['Birthday', 'Children Birthday', 'Christmas', 'Concert', 'Cruise', 'Easter', 'Exhibition', 'Graduation', 'Halloween', 'Hiking', 'Road Trip', 'Saint Patrick Day', 'Skiing', 'Wedding']

    def __init__(self, root_dir, feats_dir, split_dir, is_train=True):
        self.root_dir = root_dir
        self.feats_dir = feats_dir
        self.split_dir = split_dir

        self.local_dir = 'clip_local'
        self.global_dir = 'clip_global'
        self.token_dir = 'token'

        if is_train:
            self.phase = 'train'
        else:
            self.phase = 'test'

        if self.phase == 'train':
            split_path = os.path.join(self.split_dir, 'train.txt')
        else:
            split_path = os.path.join(self.split_dir, 'test.txt')
        
        with open(split_path, 'r') as f:
            lines = f.readlines()

        albums = [line.strip().split('/')[-1] for line in lines]

        self.albums = albums

    def __len__(self):
        return len(self.albums)
    
    def __getitem__(self, idx):
        album = self.albums[idx]
        
        feat_local = np.load(os.path.join(self.feats_dir, self.local_dir, album + '.npy'))
        feat_global = np.load(os.path.join(self.feats_dir, self.global_dir, album + '.npy'))
        token = np.load(os.path.join(self.feats_dir, self.token_dir, album + '.npy'))

        return feat_local, feat_global, token