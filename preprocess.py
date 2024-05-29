import torch
import numpy as np
import sys
import os
import json
from json.decoder import JSONDecodeError
from PIL import Image
import cv2
from torchvision.transforms.functional import crop
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader, Dataset
import argparse

parser = argparse.ArgumentParser(description='ViGAT global feature processing')
parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'], help='device')
parser.add_argument('--preprocess_dir', type=str, default='/kaggle/input/cufed-full-split', help='preprocess directory')
parser.add_argument('--save_dir', type=str, default='/kaggle/working/preprocess/global_feat', help='save directory for global feature')
parser.add_argument('--num_workers', type=int, default=4, help='num of workers of data loader')
parser.add_argument('--sample_size', type=int, default=30, help='sampling number of images in an album')
args = parser.parse_args()

# import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

# Detic libraries
sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test

# Build the detector and download our pretrained weights
cfg = get_cfg()
add_centernet_config(cfg)
add_detic_config(cfg)
cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
cfg.MODEL.DEVICE=args.device # uncomment this to use cpu-only mode.
predictor = DefaultPredictor(cfg)

# Setup the model's vocabulary using build-in datasets
BUILDIN_CLASSIFIER = {
    'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
    'coco': 'datasets/metadata/coco_clip_a+cname.npy',
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}

vocabulary = 'lvis' # change to 'lvis', 'objects365', 'openimages', or 'coco'
metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
classifier = BUILDIN_CLASSIFIER[vocabulary]
num_classes = len(metadata.thing_classes)
reset_cls_test(predictor.model, classifier, num_classes)

event_labels = ['Architecture', 'BeachTrip', 'Birthday', 'BusinessActivity',
                  'CasualFamilyGather', 'Christmas', 'Cruise', 'Graduation',
                  'GroupActivity', 'Halloween', 'Museum', 'NatureTrip',
                  'PersonalArtActivity', 'PersonalMusicActivity', 'PersonalSports',
                  'Protest', 'ReligiousActivity', 'Show', 'Sports', 'ThemePark',
                  'UrbanTrip', 'Wedding', 'Zoo']

event_processor = {'Architecture': 'Architecture', 'BeachTrip': 'Beach Trip', 'Birthday': 'Birthday', 'BusinessActivity': 'Business Activity', 'CasualFamilyGather': 'Casual Family Gather', 'Christmas': 'Christmas', 'Cruise': 'Cruise', 'Graduation': 'Graduation', 'GroupActivity': 'Group Activity', 'Halloween': 'Halloween', 'Museum': 'Museum', 'NatureTrip': 'Nature Trip', 'PersonalArtActivity': 'Personal Art Activity', 'PersonalMusicActivity': 'Personal Music Activity', 'PersonalSports': 'Personal Sports', 'Protest': 'Protest', 'ReligiousActivity': 'Religious Activity', 'Show': 'Show', 'Sports': 'Sports', 'ThemePark': 'Theme Park', 'UrbanTrip': 'Urban Trip', 'Wedding': 'Wedding', 'Zoo': 'Zoo'}

class AlbumDataset(Dataset):
  def __init__(self, album_names):
    self.album_names = album_names

  def __len__(self):
    return len(self.album_names)

  def __getitem__(self, idx):
    return self.album_names[idx]

class ImageDataset(Dataset):
  def __init__(self, image_paths, read_image=None, transform=None):
    self.image_paths = image_paths
    self.read_image = read_image
    self.transform = transform

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    image_path = self.image_paths[idx]
    image = self.read_image(image_path)
    if self.transform:
      image = self.transform(image)
    return image

class CropDataset(Dataset):
  def __init__(self, cropped_paths):
    self.cropped_paths = cropped_paths

  def __len__(self):
    return len(self.cropped_paths)

  def __getitem__(self, idx):
    return self.cropped_paths[idx]
  
def my_collate(batch):
    data = [item for item in batch]
    return data

def masked_preprocess(args):
  datasets_root = "/kaggle/input/thesis-cufed"
  dataset_root = os.path.join(datasets_root, 'CUFED')
  label_path = os.path.join(dataset_root, 'event_type.json')
  dataset_path = os.path.join(dataset_root, 'images')
  album_imgs_path = '/kaggle/working/preprocess/album_imgs.json'

  with open(label_path, 'r') as f:
    album_types = json.load(f)

  preprocess_path = os.path.join(args.preprocess_dir, 'full_split.txt')

  processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
  model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

  album_batch_size = 1
  detic_batch_size = 1
  crop_batch_size = 100
  objects_size = 50
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  with open(preprocess_path, 'r') as f:
    album_names = [name.strip() for name in f]

  album_dataset = AlbumDataset(album_names)
  album_loader = DataLoader(album_dataset, batch_size=album_batch_size, num_workers=args.num_workers, collate_fn=my_collate)
  image_dataset = ImageDataset(image_paths, read_image=cv2.imread)
  image_loader = DataLoader(image_dataset, batch_size=detic_batch_size, num_workers=args.num_workers, collate_fn=my_collate)

  with open(album_imgs_path, 'r') as json_file:
    try:
        album_imgs_dict = json.load(json_file)
    except JSONDecodeError:
        album_imgs_dict = {}

  for album_idx, albums in enumerate(album_loader):
    album = albums[0]
    
    # skip pre-existing npy feats
    global_path = os.path.join(args.save_dir, f"{album}.npy")
    local_path = os.path.join(args.save_dir, f"{album}.npy")
    if os.path.exists(global_path) and os.path.exists(local_path) and album in album_imgs_dict:
      continue

    print(f"------Album {album_idx + 1}: {album}------")

    album_dir = os.path.join(dataset_path, album)
    image_names = os.listdir(album_dir)
    image_paths = [os.path.join(album_dir, image_name) for image_name in image_names]

    # vit_global_processor
    with torch.no_grad():
      inputs = processor(text=[event_processor[album_type] for album_type in album_types[album]], 
                        images=[Image.open(image_path) for image_path in image_paths], 
                        return_tensors="pt", padding=True)
      outputs = model(**inputs.to(device))
    sims = outputs.logits_per_image.cpu().numpy().mean(axis=1)
    feat = outputs.image_embeds.cpu().numpy()
    selected_feat = np.zeros((args.sample_size, feat.shape[1]), dtype=np.float32)
    top_sims = np.argsort(sims)[-args.sample_size:][::-1]

    for i, idx in enumerate(top_sims):
      selected_feat[i] = feat[idx]
    np.save(os.path.join(args.save_dir, f"{album}.npy"), selected_feat)

    # crop objects
    cropped_image_tensor_list = []
    pad_list = []
    gidx = 0
    for imgs in image_loader:
      img = imgs[0]
      outputs = predictor(img)
      boxes = outputs.pred_boxes
      pad_size = objects_size
      loop_cnt = min(len(boxes), objects_size)
      for box_idx in range(loop_cnt):
        x1, y1, x2, y2 = [round(cord) for cord in boxes[box_idx].tolist()]
        cropped_image_tensor = crop(img, y1, x1, y2 - y1, x2 - x1)
        cropped_image_tensor = torch.clamp(cropped_image_tensor, min=0.0, max=1.0)
        cropped_image_tensor_list.append(cropped_image_tensor)
      gidx += loop_cnt
      pad_size -= loop_cnt
      pad_list.append((gidx, pad_size))

    # vit_local_processor
    crop_dataset = CropDataset(cropped_image_tensor_list)
    crop_loader = DataLoader(crop_dataset, batch_size=crop_batch_size, num_workers=args.num_workers, collate_fn=my_collate)
    vit_local_list = []
    with torch.no_grad():
      for crops in crop_loader:
        cropped_inputs = processor(text=[event_processor[album_type] for album_type in album_types[album]], 
                                   images=crops, return_tensors="pt", padding=True)
        cropped_outputs = model(**cropped_inputs.to(device))
        cropped_features = cropped_outputs.image_embeds.cpu().numpy()
        vit_local_list.append(cropped_features)
    vit_local_np = np.vstack(vit_local_list)
    if len(cropped_image_tensor_list) < args.sample_size * objects_size:
      for pad in reversed(pad_list):
        vit_local_np = np.vstack([vit_local_np[:pad[0]], np.zeros((pad[1], 768)), vit_local_np[pad[0]:]])
    vit_local_np = vit_local_np.reshape(args.sample_size, objects_size, -1).astype(np.float32)
    np.save(os.path.join(args.save_dir, f"{album}.npy"), vit_local_np)

    # get selected images of albums
    image_names_without_extension = np.char.replace(image_names, '.jpg', '')
    album_imgs_dict[album] = image_names_without_extension[top_sims].tolist()
  with open(album_imgs_path, 'w') as json_file:
    json.dump(album_imgs_dict, json_file)

def main():
  masked_preprocess(args)


if __name__ == '__main__':
  main()