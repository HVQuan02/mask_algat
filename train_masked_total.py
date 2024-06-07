import argparse
import time
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import CUFED
from model import tokengraph_with_global_part_sharing as Model
from utils import AP_partial


parser = argparse.ArgumentParser(description='GCN Video Classification')
parser.add_argument('model', nargs=2, help='trained model')
parser.add_argument('--seed', type=int, default=2024, help='seed for randomness')
parser.add_argument('--gcn_layers', type=int, default=2, help='number of gcn layers')
parser.add_argument('--dataset', default='cufed', choices=['holidays', 'pec', 'cufed'])
parser.add_argument('--dataset_root', default='/kaggle/input/thesis-cufed/CUFED', help='dataset root directory')
parser.add_argument('--feats_dir', default='/kaggle/input/mask-cufed-feats', help='global and local features directory')
parser.add_argument('--split_dir', default='/kaggle/input/cufed-full-split', help='train split and val split')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--milestones', nargs="+", type=int, default=[110, 160], help='milestones of learning decay')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
parser.add_argument('--save_scores', action='store_true', help='save the output scores')
parser.add_argument('--save_path', default='scores.txt', help='output path')
parser.add_argument('--resume', default=None, help='checkpoint to resume training')
parser.add_argument('--save_folder', default='weights', help='directory to save checkpoints')
parser.add_argument('--patience', type=int, default=20, help='patience of early stopping')
parser.add_argument('--min_delta', type=float, default=0.1, help='min delta of early stopping')
parser.add_argument('--threshold', type=float, default=95, help='val mAP threshold of early stopping')
parser.add_argument('-v', '--verbose', action='store_true', help='show details')
args = parser.parse_args()


class EarlyStopper:
    def __init__(self, patience, min_delta, threshold):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_mAP = -float('inf')
        self.threshold = threshold

    def early_stop(self, validation_mAP):
        if validation_mAP >= self.threshold:
            return True, True
        if validation_mAP > self.max_validation_mAP:
            self.max_validation_mAP = validation_mAP
            self.counter = 0
            return False, True
        if validation_mAP < (self.max_validation_mAP - self.min_delta):
            self.counter += 1
            if self.counter > self.patience:
                return True, False
        return False, False


def train_omega(model, loader,  crit, opt, sched, device):
    epoch_loss = 0
    model.train()
    for feats, feats_global, label in loader:
        feats = feats.to(device)
        feats_global = feats_global.to(device)
        label = label.to(device)
        opt.zero_grad()
        out_data = model(feats, feats_global)
        loss = crit(out_data, label)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    sched.step()
    return epoch_loss / len(loader)


def validate_omega(model, dataset, loader, device):
    scores = np.zeros((len(dataset), dataset.NUM_CLASS), dtype=np.float32)
    gidx = 0
    model.eval()
    with torch.no_grad():
        for feats, feat_global, _ in loader:
            feats = feats.to(device)
            feat_global = feat_global.to(device)
            out_data = model(feats, feat_global)
            shape = out_data.shape[0]
            scores[gidx:gidx+shape, :] = out_data.cpu()
            gidx += shape
        map_macro = AP_partial(dataset.labels, scores)[2]
        return map_macro


def main():
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    if args.dataset == 'cufed':
        dataset = CUFED(root_dir=args.dataset_root, feats_dir=args.feats_dir, split_dir=args.split_dir, is_train=True)
        val_dataset = CUFED(root_dir=args.dataset_root, feats_dir=args.feats_dir, split_dir=args.split_dir, is_train=True, is_val=True)
    else:
        sys.exit("Unknown dataset!")

    device = torch.device('cuda:0')
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    if args.verbose:
        print("running on {}".format(device))
        print("num of train set = {}".format(len(dataset)))
        print("num of val set = {}".format(len(val_dataset)))

    start_epoch = 0

    # Load the saved model
    local_checkpoint = torch.load(args.model[0])
    global_checkpoint = torch.load(args.model[1])

    local_graph_state_dict = local_checkpoint['graph_state_dict']
    global_graph_state_dict = global_checkpoint['graph_state_dict']

    print('load local graph model from epoch {}'.format(local_checkpoint['epoch']))
    print('load global graph model from epoch {}'.format(global_checkpoint['epoch']))

    # Create an instance of the omega4_video model and load the pretrained GraphModule
    model = Model(args.gcn_layers, dataset.NUM_FEATS,  dataset.NUM_CLASS).to(device)
    model.graph.load_state_dict(local_graph_state_dict)
    model.graph_omega.load_state_dict(global_graph_state_dict)
    model.graph.eval()
    model.graph_omega.eval()
    crit = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)
    # Different LR
    # Separate parameter groups for graph and graph_omega3 modules
    #parameters = [
    #    {"params": model.graph.parameters(), "lr": 1e-5},  # Set desired learning rate for graph
    #    {"params": model.graph_omega.parameters(), "lr": args.lr},  # Set desired learning rate for graph_omega3
    #    {"params": model.cls.parameters()}  # Use default learning rate for cls module
    #]
    #opt = optim.Adam(parameters, lr=opt.param_groups[0]['lr'])
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=args.milestones)
    early_stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta, threshold=args.threshold)

    if args.resume:
        data = torch.load(args.resume)
        start_epoch = data['epoch']
        model.load_state_dict(data['model_state_dict'])
        opt.load_state_dict(data['opt_state_dict'])
        sched.load_state_dict(data['sched_state_dict'])
        if args.verbose:
            print("resuming from epoch {}".format(start_epoch))

    for epoch in range(start_epoch, args.num_epochs):
        epoch_cnt = epoch + 1
        
        t0 = time.perf_counter()
        train_loss = train_omega(model, loader, crit, opt, sched, device)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        val_map = validate_omega(model, val_dataset, val_loader, device)
        t3 = time.perf_counter()

        is_early_stopping, is_save_ckpt = early_stopper.early_stop(val_map)

        model_config = {
            'epoch': epoch_cnt,
            'loss': train_loss,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict(),
            'sched_state_dict': sched.state_dict()
        }

        torch.save(model_config, os.path.join(args.save_folder, 'last-total-maskedViGAT-{}.pt'.format(args.dataset)))

        if is_save_ckpt:
            torch.save(model_config, os.path.join(args.save_folder, 'best-total-maskedViGAT-{}.pt'.format(args.dataset)))

        if is_early_stopping:
            print('Stop at epoch {}'.format(epoch_cnt)) 
            break

        if args.verbose:
            print("[epoch {}] train_loss={} val_map={} dt_train={:.2f}sec dt_val={:.2f}sec dt={:.2f}sec".format(epoch_cnt, train_loss, val_map, t1 - t0, t3 - t2, t1 - t0 + t3 - t2))


if __name__ == '__main__':
    main()