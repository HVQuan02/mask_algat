import sys
import time
import torch
import numpy as np
import torch.nn as nn
from dataset import CUFED, PEC
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from options.test_options import TestOptions
from utils import AP_partial, spearman_correlation, showCM
from model import tokengraph_with_global_part_sharing as Model
from sklearn.metrics import multilabel_confusion_matrix, classification_report

args = TestOptions().parse()


def evaluate(model, dataset, loader, out_file, device):
    model.eval()
    gidx = 0
    frame_wid_list = []
    importance_list = []
    is_cufed = isinstance(dataset, CUFED)
    scores = torch.zeros((len(dataset), dataset.NUM_CLASS), dtype=torch.float32)

    with torch.no_grad():
        if is_cufed:
            for batch in loader:
                feats_local, feats_global, _, importances = batch
                # Run model with all frames
                feats_local = feats_local.to(device)
                feats_global = feats_global.to(device)
                out_data, _, wids_frame_local, wids_frame_global = model(feats_local, feats_global, get_adj=True)
                shape = out_data.shape[0]

                if out_file:
                    for j in range(shape):
                        video_name = dataset.videos[gidx + j]
                        out_file.write("{} ".format(video_name))
                        out_file.write(' '.join([str(x.item()) for x in out_data[j, :]]))
                        out_file.write('\n')

                scores[gidx:gidx+shape, :] = out_data.cpu()
                gidx += shape
                importance_list.append(importances)
                avg_frame_wid = (wids_frame_local + wids_frame_global) / 2
                frame_wid_list.append(torch.from_numpy(avg_frame_wid))
        else:
            for batch in loader:
                feats_local, feats_global, _ = batch
                # Run model with all frames
                feats_local = feats_local.to(device)
                feats_global = feats_global.to(device)
                out_data = model(feats_local, feats_global)
                shape = out_data.shape[0]

                if out_file:
                    for j in range(shape):
                        video_name = dataset.videos[gidx + j]
                        out_file.write("{} ".format(video_name))
                        out_file.write(' '.join([str(x.item()) for x in out_data[j, :]]))
                        out_file.write('\n')

                scores[gidx:gidx+shape, :] = out_data.cpu()
                gidx += shape
                
    m = nn.Sigmoid()
    preds = m(scores)
    preds[preds >= args.threshold] = 1
    preds[preds < args.threshold] = 0
    scores, preds = scores.numpy(), preds.numpy()
    
    # Ensure no row has all zeros
    for i in range(preds.shape[0]):
        if np.sum(preds[i]) == 0:
            preds[i][np.argmax(scores[i])] = 1

    map_micro, map_macro = AP_partial(dataset.labels, scores)[1:3]
    acc = accuracy_score(dataset.labels, preds)
    cms = multilabel_confusion_matrix(dataset.labels, preds)
    cr = classification_report(dataset.labels, preds)
    
    if is_cufed:
        importance_matrix = torch.cat(importance_list).to(device)
        wid_frame_matrix = torch.cat(frame_wid_list).to(device)
        spearman = spearman_correlation(wid_frame_matrix, importance_matrix)

        return map_micro, map_macro, acc, spearman, cms, cr
    
    return map_micro, map_macro, acc, cms, cr


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'cufed':
        dataset = CUFED(root_dir=args.dataset_root, feats_dir=args.feats_dir, split_dir=args.split_dir, is_train=False)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset == 'pec':
        dataset = PEC(root_dir=args.dataset_root, feats_dir=args.feats_dir, split_dir=args.split_dir, is_train=False)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    else:
        sys.exit("Unknown dataset!")

    model = Model(args.gcn_layers, dataset.NUM_FEATS, dataset.NUM_CLASS)
    state = torch.load(args.model[0], map_location=device)
    model.load_state_dict(state['model_state_dict'], strict=True)
    model = model.to(device)

    if args.verbose:
        print("running on {}".format(device))
        print("test_set = {}".format(len(dataset)))
        print("load model from epoch {}".format(state['epoch']))

    out_file = None
    if args.save_scores:
        out_file = open(args.save_path, 'w')

    t0 = time.perf_counter()
    if args.dataset == 'cufed':
        map_micro, map_macro, acc, spearman, cms, cr = evaluate(model, dataset, loader, out_file, device)
    else:
        map_micro, map_macro, acc, cms, cr = evaluate(model, dataset, loader, out_file, device)
    t1 = time.perf_counter()

    if args.save_scores:
        out_file.close()

    if args.dataset == 'cufed':
        print('map_micro={:.2f} map_macro={:.2f} accuracy={:.2f} spearman={:.3f} dt={:.2f}sec'.format(map_micro, map_macro, acc * 100, spearman, t1 - t0))
    else:
        print('map_micro={:.2f} map_macro={:.2f} accuracy={:.2f} dt={:.2f}sec'.format(map_micro, map_macro, acc * 100, t1 - t0))

    print(cr)
    showCM(cms)


if __name__ == '__main__':
    main()