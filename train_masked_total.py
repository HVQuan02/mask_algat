import argparse
import time
import torch
import sys
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from datasets import FCVID, ACTNET, miniKINETICS, YLIMED
from model import tokens_with_global_part_sharing as Model


parser = argparse.ArgumentParser(description='GCN Video Classification')
parser.add_argument('model', nargs=1, help='trained model')
parser.add_argument('--gcn_layers', type=int, default=2, help='number of gcn layers')
parser.add_argument('--dataset', default='actnet', choices=['fcvid', 'minikinetics', 'actnet', 'ylimed'])
parser.add_argument('--dataset_root', default='/ActivityNet', help='dataset root directory')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--milestones', nargs="+", type=int, default=[110, 160], help='milestones of learning decay')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_objects', type=int, default=50, help='number of objects with best DoC')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
parser.add_argument('--ext_method', default='CLIP', choices=['VIT', 'CLIP'], help='Extraction method for features')
parser.add_argument('--save_scores', action='store_true', help='save the output scores')
parser.add_argument('--save_path', default='scores.txt', help='output path')
parser.add_argument('--resume', default=None, help='checkpoint to resume training')
parser.add_argument('--save_interval', type=int, default=10, help='interval for saving models (epochs)')
parser.add_argument('--save_folder', default='weights', help='directory to save checkpoints')
parser.add_argument('-v', '--verbose', action='store_true', help='show details')
args = parser.parse_args()


def train_omega(model, loader,  crit, opt, sched, device):
    epoch_loss = 0
    # model.eval()
    for i, batch in enumerate(loader):
        feats, feats_global, label, _ = batch

        # Run model with all frames
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


def main():
    if args.dataset == 'fcvid':
        dataset = FCVID(args.dataset_root, is_train=True, ext_method=args.ext_method)
        crit = nn.BCEWithLogitsLoss()
    elif args.dataset == 'actnet':
        dataset = ACTNET(args.dataset_root, is_train=True, ext_method=args.ext_method)
        crit = nn.BCEWithLogitsLoss()
    elif args.dataset == 'minikinetics':
        dataset = miniKINETICS(args.dataset_root, is_train=True, ext_method=args.ext_method)
        crit = nn.CrossEntropyLoss()
    elif args.dataset == 'ylimed':
        dataset = YLIMED(args.dataset_root, is_train=True, ext_method=args.ext_method)
        crit = nn.CrossEntropyLoss()
    else:
        sys.exit("Unknown dataset!")
    device = torch.device('cuda:0')
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    if args.verbose:
        print("running on {}".format(device))
        print("num samples={}".format(len(dataset)))
        print("missing videos={}".format(dataset.num_missing))

    start_epoch = 0

    # Load the saved model
    checkpoint = torch.load(args.model[0])

    # Extract the state dictionaries of the model and optimizer
    model_state_dict = checkpoint['model_state_dict']

    # Create an instance of the omega4_video model and load the pretrained GraphModule
    model = Model(args.gcn_layers, dataset.NUM_FEATS,  dataset.NUM_CLASS).to(device)
    graph_state_dict = {}
    for k, v in model_state_dict.items():
        if k.startswith('graph.'):
            graph_state_dict[k[6:]] = v
    model.graph.load_state_dict(graph_state_dict)

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
    if args.resume:
        data = torch.load(args.resume)
        start_epoch = data['epoch']
        model.load_state_dict(data['model_state_dict'])
        opt.load_state_dict(data['opt_state_dict'])
        sched.load_state_dict(data['sched_state_dict'])
        if args.verbose:
            print("resuming from epoch {}".format(start_epoch))

    model.train()
    for epoch in range(start_epoch, args.num_epochs):
        t0 = time.perf_counter()
        loss = train_omega(model, loader,  crit, opt, sched, device)
        t1 = time.perf_counter()

        if args.verbose:
            print("[epoch {}] loss={} dt={:.2f}sec".format(epoch + 1, loss, t1 - t0))
        if (epoch + 1) % args.save_interval == 0:
            sfnametmpl = 'model-{}-tgraph_actnet_finetune_and_graph-{:03d}.pt'
            sfname = sfnametmpl.format(args.dataset, epoch + 1)
            spth = os.path.join(args.save_folder, sfname)
            torch.save({
                'epoch': epoch + 1,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'opt_state_dict': opt.state_dict(),
                'sched_state_dict': sched.state_dict()
            }, spth)


if __name__ == '__main__':
    main()
