import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataset import CUFED_Tokens, PEC_Tokens
from model import MaskedGCN as Model
from torch.utils.data import DataLoader
from options.train_local_options import TrainLocalOptions

args = TrainLocalOptions().parse()


class EarlyStopper:
    def __init__(self, patience, min_delta, stopping_threshold):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = float('inf')
        self.stopping_threshold = stopping_threshold

    def early_stop(self, min_val_loss):
        if min_val_loss <= self.stopping_threshold:
            return True, True
        if min_val_loss < self.min_val_loss:
            self.min_val_loss = min_val_loss
            self.counter = 0
            return False, True
        if min_val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter > self.patience:
                return True, False
        return False, False


def train(model, loader, crit, opt, sched, device):
    model.train()
    epoch_loss = 0
    for batch in loader:
        local_feats, _, tokens = batch
        local_feats = local_feats.to(device)
        tokens = tokens.to(device)
        opt.zero_grad()
        out_data = model(local_feats)
        loss = crit(out_data, tokens)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    sched.step()
    return epoch_loss / len(loader)


def validate(model, loader, crit, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in loader:
            local_feats, _, tokens = batch
            local_feats = local_feats.to(device)
            tokens = tokens.to(device)
            out_data = model(local_feats)
            loss = crit(out_data, tokens)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if args.dataset == 'cufed':
        train_dataset = CUFED_Tokens(root_dir=args.dataset_root, feats_dir=args.feats_dir, split_dir=args.split_dir)
        val_dataset = CUFED_Tokens(root_dir=args.dataset_root, feats_dir=args.feats_dir, split_dir=args.split_dir, is_train=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset == 'pec':
        train_dataset = PEC_Tokens(root_dir=args.dataset_root, feats_dir=args.feats_dir, split_dir=args.split_dir)
        val_dataset = PEC_Tokens(root_dir=args.dataset_root, feats_dir=args.feats_dir, split_dir=args.split_dir, is_train=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    else:
        sys.exit("Unknown dataset!")

    if args.verbose:
        print("running on {}".format(device))
        print("train_set = {}".format(len(train_dataset)))
        print("val_set = {}".format(len(val_dataset)))

    model = Model(args.gcn_layers, train_dataset.NUM_FEATS, train_dataset.TOKEN_SIZE, args.mask_percentage)
    crit = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=args.milestones)
    early_stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta, stopping_threshold=args.stopping_threshold)

    start_epoch = 0
    if args.resume:
        data = torch.load(args.resume, map_location=device)
        start_epoch = data['epoch']
        model.load_state_dict(data['model_state_dict'], strict=True)
        opt.load_state_dict(data['opt_state_dict'])
        sched.load_state_dict(data['sched_state_dict'])
        if args.verbose:
            print("resuming from epoch {}".format(start_epoch))

    for epoch in range(start_epoch, args.num_epochs):
        epoch_cnt = epoch + 1
        model = model.to(device)

        t0 = time.perf_counter()
        train_loss = train(model, train_loader, crit, opt, sched, device)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        val_loss = validate(model, val_loader, crit, device)
        t3 = time.perf_counter()

        is_early_stopping, is_save_ckpt = early_stopper.early_stop(val_loss)

        model_config = {
            'epoch': epoch_cnt,
            'loss': train_loss,
            'model_state_dict': model.state_dict(),
            'graph_state_dict': model.graph.state_dict(),
            'opt_state_dict': opt.state_dict(),
            'sched_state_dict': sched.state_dict()
        }

        torch.save(model_config, os.path.join(args.save_dir, 'last_local_mask_algat_{}.pt'.format(args.dataset)))

        if is_save_ckpt:
            torch.save(model_config, os.path.join(args.save_dir, 'best_local_mask_algat_{}.pt'.format(args.dataset)))

        if is_early_stopping:
            print('Early stop at epoch {}'.format(epoch_cnt)) 
            break

        if args.verbose:
            print("[epoch {}] train_loss={} val_loss={} dt_train={:.2f}sec dt_val={:.2f}sec dt={:.2f}sec".format(epoch_cnt, train_loss, val_loss, t1 - t0, t3 - t2, t1 - t0 + t3 - t2))


if __name__ == '__main__':
    main()