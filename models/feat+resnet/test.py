import logging
import os
import torch
from torch.utils.model_zoo import tqdm
import random
import numpy as np
from dataset import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
import eval_metrics as em
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from utils import setup_seed
import argparse
from utils import str2bool

## Adapted from https://github.com/pytorch/audio/tree/master/torchaudio
## https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/blob/newfunctions/

def init():
    parser = argparse.ArgumentParser("load model scores")
    parser.add_argument('--seed', type=int, help="random number seed", default=1000)
    parser.add_argument("-d", "--path_to_database", type=str, help="dataset path",
                        default='/data1/neil/DS_10283_3336/')
    parser.add_argument("-f", "--path_to_features", type=str, help="features path",
                        default='/data2/neil/ASVspoof2019LA/')
    parser.add_argument('-m', '--model_dir', type=str, help="directory for pretrained model", required=True,
                        default='/data3/neil/chan/adv1010')
    parser.add_argument("-t", "--task", type=str, help="which dataset you would like to test on",
                        required=True, default='ASVspoof2019LA',
                        choices=["ASVspoof2019LA", "ASVspoof2015", "VCC2020", "ASVspoof2019LASim", "ASVspoof2021LA", "singfake"])
    parser.add_argument('-l', '--loss', type=str, default="ocsoftmax",
                        choices=["softmax", "amsoftmax", "ocsoftmax", "isolate", "scl", "angulariso"],
                        help="loss for scoring")
    parser.add_argument('--weight_loss', type=float, default=0.5, help="weight for other loss")
    parser.add_argument("--feat", type=str, help="which feature to use", default='LFCC',
                        choices=["CQCC", "LFCC", "Raw", "MFCC"])
    parser.add_argument("--feat_len", type=int, help="features length", default=500)
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size for training")
    parser.add_argument('--is_mixture', type=str2bool, nargs='?', const=True, default=False,
                        help="whether use mixture or vocals in training")
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(args.seed)      # Set seeds
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

def test_model_on_ASVspoof2019LA(feat_model_path, loss_model_path, part, add_loss):
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
    else:
        dir_path = dirname(feat_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(feat_model_path)
    loss_model = torch.load(loss_model_path) if add_loss is not None else None
    test_set = ASVspoof2019LASpec(args.path_to_database, args.path_to_features, part,
                            args.feat, feat_len=args.feat_len)
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model.eval()
    score_loader, idx_loader = [], []

    with open(os.path.join(dir_path, 'checkpoint_cm_score_ASVspoof2019LA.txt'), 'w') as cm_score_file:
        for i, (feat, audio_fn, tags, labels, _) in enumerate(tqdm(testDataLoader)):
            if args.feat == "Raw":
                feat = feat.to(args.device)
            else:
                feat = feat.transpose(2, 3).to(args.device)
            # print(feat.shape)
            tags = tags.to(device)
            labels = labels.to(device)

            feats, feat_outputs = model(feat)

            if add_loss == "softmax":
                score = F.softmax(feat_outputs)[:, 0]
            elif add_loss == "ocsoftmax":
                ang_isoloss, score = loss_model(feats, labels)
            elif add_loss == "isolate":
                _, score = loss_model(feats, labels)
            elif add_loss == "scl":
                score_softmax = F.softmax(feat_outputs)[:, 0]
                _, score_scl = loss_model(feats, labels)
                score = score_softmax + args.weight_loss * score_scl
            elif add_loss == "amsoftmax":
                outputs, moutputs = loss_model(feats, labels)
                score = F.softmax(outputs, dim=1)[:, 0]
            elif add_loss == "angulariso":
                angularisoloss, score = loss_model(feats, labels)
            else:
                raise ValueError("what is the loss?")

            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))

            # score_loader.append(score.detach().cpu())
            # idx_loader.append(labels.detach().cpu())

    # scores = torch.cat(score_loader, 0).data.cpu().numpy()
    # labels = torch.cat(idx_loader, 0).data.cpu().numpy()
    # eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

    eer, min_tDCF = compute_eer_and_tdcf(os.path.join(dir_path, 'checkpoint_cm_score_ASVspoof2019LA.txt'),
                                            "/data1/neil/DS_10283_3336/")

    return eer, min_tDCF

def test_model_on_singfake(feat_model_path, loss_model_path, dataset_path, add_loss, args):
    model = torch.load(feat_model_path)
    loss_model = torch.load(loss_model_path) if add_loss is not None else None
    test_set = Dataset_SingFake(dataset_path, is_mixture=args.is_mixture)
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model.eval()
    with torch.no_grad():
        idx_loader, score_loader = [], []
        for i, (feat, labels) in enumerate(tqdm(testDataLoader)):
            if args.feat == "Raw":
                feat = feat.to(args.device)
            else:
                feat = feat.transpose(2, 3).to(args.device)

            labels = labels.to(args.device)
            feats, feat_outputs = model(feat)

            if add_loss == "softmax":
                score = F.softmax(feat_outputs)[:, 0]
            elif add_loss == "ocsoftmax":
                ang_isoloss, score = loss_model(feats, labels)
            elif add_loss == "isolate":
                _, score = loss_model(feats, labels)
            elif add_loss == "scl":
                score_softmax = F.softmax(feat_outputs)[:, 0]
                _, score_scl = loss_model(feats, labels)
                score = score_softmax + args.weight_loss * score_scl
            elif add_loss == "amsoftmax":
                outputs, moutputs = loss_model(feats, labels)
                score = F.softmax(outputs, dim=1)[:, 0]
            elif add_loss == "angulariso":
                angularisoloss, score = loss_model(feats, labels)
            else:
                raise ValueError("what is the loss?")
            

            idx_loader.append((labels))
            score_loader.append(score)

        scores = torch.cat(score_loader, 0).data.cpu().numpy()
        labels = torch.cat(idx_loader, 0).data.cpu().numpy()
        eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
        # print eer in percentage
        print(eer * 100)

    return eer


def test_on_VCC(feat_model_path, loss_model_path, part, add_loss):
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
    else:
        dir_path = dirname(feat_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(feat_model_path)
    loss_model = torch.load(loss_model_path) if add_loss is not None else None
    test_set_VCC = VCC2020("/data2/neil/VCC2020/", "LFCC", feat_len=args.feat_len)
    testDataLoader = DataLoader(test_set_VCC, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model.eval()
    score_loader, idx_loader = [], []

    with open(os.path.join(dir_path, 'checkpoint_cm_score_VCC.txt'), 'w') as cm_score_file:
        for i, (feat, _, tags, labels, _) in enumerate(tqdm(testDataLoader)):
            if args.feat == "Raw":
                feat = feat.to(args.device)
            else:
                feat = feat.transpose(2, 3).to(args.device)

            tags = tags.to(device)
            labels = labels.to(device)

            feats, feat_outputs = model(feat)

            if add_loss == "softmax":
                score = F.softmax(feat_outputs)[:, 0]
            elif add_loss == "ocsoftmax":
                ang_isoloss, score = loss_model(feats, labels)
            elif add_loss == "isolate":
                _, score = loss_model(feats, labels)
            elif add_loss == "scl":
                score_softmax = F.softmax(feat_outputs)[:, 0]
                _, score_scl = loss_model(feats, labels)
                score = score_softmax + args.weight_loss * score_scl
            elif add_loss == "amsoftmax":
                outputs, moutputs = loss_model(feats, labels)
                score = F.softmax(outputs, dim=1)[:, 0]
            elif add_loss == "angulariso":
                angularisoloss, score = loss_model(feats, labels)
            else:
                raise ValueError("what is the loss?")

            for j in range(labels.size(0)):
                cm_score_file.write(
                    'A%02d %s %s\n' % (tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))

            score_loader.append(score.detach().cpu())
            idx_loader.append(labels.detach().cpu())

    scores = torch.cat(score_loader, 0).data.cpu().numpy()
    labels = torch.cat(idx_loader, 0).data.cpu().numpy()
    eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

    return eer

def test_on_ASVspoof2015(feat_model_path, loss_model_path, part, add_loss):
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
    else:
        dir_path = dirname(feat_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(feat_model_path)
    loss_model = torch.load(loss_model_path) if add_loss is not None else None
    test_set_2015 = ASVspoof2015("/data2/neil/ASVspoof2015/", part="eval", feature="LFCC", feat_len=args.feat_len)
    testDataLoader = DataLoader(test_set_2015, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model.eval()
    score_loader, idx_loader = [], []

    with open(os.path.join(dir_path, 'checkpoint_cm_score_ASVspoof2015.txt'), 'w') as cm_score_file:
        for i, (feat, audio_fn, tags, labels, _) in enumerate(tqdm(testDataLoader)):
            if args.feat == "Raw":
                feat = feat.to(args.device)
            else:
                feat = feat.transpose(2, 3).to(args.device)
            tags = tags.to(device)
            labels = labels.to(device)

            feats, feat_outputs = model(feat)

            if add_loss == "softmax":
                score = F.softmax(feat_outputs)[:, 0]
            elif add_loss == "ocsoftmax":
                ang_isoloss, score = loss_model(feats, labels)
            elif add_loss == "isolate":
                _, score = loss_model(feats, labels)
            elif add_loss == "scl":
                score_softmax = F.softmax(feat_outputs)[:, 0]
                _, score_scl = loss_model(feats, labels)
                score = score_softmax + args.weight_loss * score_scl
            elif add_loss == "amsoftmax":
                outputs, moutputs = loss_model(feats, labels)
                score = F.softmax(outputs, dim=1)[:, 0]
            elif add_loss == "angulariso":
                angularisoloss, score = loss_model(feats, labels)
            else:
                raise ValueError("what is the loss?")

            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))

            score_loader.append(score.detach().cpu())
            idx_loader.append(labels.detach().cpu())

    scores = torch.cat(score_loader, 0).data.cpu().numpy()
    labels = torch.cat(idx_loader, 0).data.cpu().numpy()
    eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

    return eer

def test_individual_attacks(cm_score_file):
    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(np.float)

    eer_cm_lst, min_tDCF_lst = [], []
    for attack_idx in range(0, 55):
        # Extract target, nontarget, and spoof scores from the ASV scores

        # Extract bona fide (real human) and spoof scores from the CM scores
        bona_cm = cm_scores[cm_keys == 'bonafide']
        spoof_cm = cm_scores[cm_sources == 'A%02d' % attack_idx]

        # EERs of the standalone systems and fix ASV operating point to EER threshold
        eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
        eer_cm_lst.append(eer_cm)

    return eer_cm_lst

def test_on_ASVspoof2019LASim(feat_model_path, loss_model_path, part, add_loss):
    dirname = os.path.dirname
    # basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
    else:
        dir_path = dirname(feat_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(feat_model_path)
    loss_model = torch.load(loss_model_path) if add_loss is not None else None
    test_set = ASVspoof2019LASim(path_to_features="/data2/neil/ASVspoof2019LA/",
                                                path_to_deviced="/dataNVME/neil/ASVspoof2019LADevice",
                                                part="eval",
                                                feature=args.feat, feat_len=args.feat_len)
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model.eval()
    # score_loader, idx_loader = [], []

    with open(os.path.join(dir_path, 'checkpoint_cm_score_ASVspoof2019LASim.txt'), 'w') as cm_score_file:
        for i, (feat, audio_fn, tags, labels, _) in enumerate(tqdm(testDataLoader)):
            if i > int(len(test_set) / args.batch_size / (len(test_set.devices) + 1)): break
            if args.feat == "Raw":
                feat = feat.to(args.device)
            else:
                feat = feat.transpose(2, 3).to(args.device)
            # print(feat.shape)
            tags = tags.to(device)
            labels = labels.to(device)

            feats, feat_outputs = model(feat)

            if add_loss == "softmax":
                score = F.softmax(feat_outputs)[:, 0]
            elif add_loss == "ocsoftmax":
                ang_isoloss, score = loss_model(feats, labels)
            elif add_loss == "isolate":
                _, score = loss_model(feats, labels)
            elif add_loss == "scl":
                score_softmax = F.softmax(feat_outputs)[:, 0]
                _, score_scl = loss_model(feats, labels)
                score = score_softmax + args.weight_loss * score_scl
            elif add_loss == "amsoftmax":
                outputs, moutputs = loss_model(feats, labels)
                score = F.softmax(outputs, dim=1)[:, 0]
            elif add_loss == "angulariso":
                angularisoloss, score = loss_model(feats, labels)
            else:
                raise ValueError("what is the loss?")

            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))

    #         score_loader.append(score.detach().cpu())
    #         idx_loader.append(labels.detach().cpu())
    #
    # scores = torch.cat(score_loader, 0).data.cpu().numpy()
    # labels = torch.cat(idx_loader, 0).data.cpu().numpy()
    # eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

    eer, min_tDCF = compute_eer_and_tdcf(os.path.join(dir_path, 'checkpoint_cm_score_ASVspoof2019LASim.txt'),
                                            "/data/neil/DS_10283_3336/")

    return eer, min_tDCF


def test_on_ASVspoof2021LA(feat_model_path, loss_model_path, part, add_loss):
    dirname = os.path.dirname
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
    else:
        dir_path = dirname(feat_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(feat_model_path)
    loss_model = torch.load(loss_model_path) if add_loss is not None else None

    ### use this line to generate score for LA 2021 Challenge
    test_set = ASVspoof2021LAeval(feature=args.feat, feat_len=args.feat_len)
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model.eval()

    txt_file_name = os.path.join(dir_path, 'score.txt')

    with open(txt_file_name, 'w') as cm_score_file:
        for i, data_slice in enumerate(tqdm(testDataLoader)):
            feat, audio_fn = data_slice
            if args.feat == "Raw":
                feat = feat.to(args.device)
            else:
                feat = feat.transpose(2, 3).to(args.device)

            labels = torch.zeros((feat.shape[0]))
            labels = labels.to(device)

            feats, feat_outputs = model(feat)

            if add_loss == "softmax":
                score = F.softmax(feat_outputs)[:, 0]
            elif add_loss == "ocsoftmax":
                ang_isoloss, score = loss_model(feats, labels)
            elif add_loss == "isolate":
                _, score = loss_model(feats, labels)
            elif add_loss == "scl":
                score_softmax = F.softmax(feat_outputs)[:, 0]
                _, score_scl = loss_model(feats, labels)
                score = score_softmax + args.weight_loss * score_scl
            elif add_loss == "amsoftmax":
                outputs, moutputs = loss_model(feats, labels)
                score = F.softmax(outputs, dim=1)[:, 0]
            elif add_loss == "angulariso":
                angularisoloss, score = loss_model(feats, labels)
            else:
                raise ValueError("what is the loss?")

            for j in range(labels.size(0)):
                cm_score_file.write('%s %s\n' % (audio_fn[j], score[j].item()))


if __name__ == "__main__":
    device = torch.device("cuda")

    args = init()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model_path = os.path.join(args.model_dir, "anti-spoofing_feat_model.pt")
    loss_model_path = os.path.join(args.model_dir, "anti-spoofing_loss_model.pt")

    if args.task == "ASVspoof2019LA":
        eer = test_model_on_ASVspoof2019LA(model_path, loss_model_path, "eval", args.loss)
    elif args.task == "ASVspoof2015":
        eer = test_on_ASVspoof2015(model_path, loss_model_path, "eval", args.loss)
        print(eer)
    elif args.task =="VCC2020":
        eer = test_on_VCC(model_path, loss_model_path, "eval", args.loss)
        print(eer)
    elif args.task =="ASVspoof2019LASim":
        eer = test_on_ASVspoof2019LASim(model_path, loss_model_path, "eval", args.loss)
    elif args.task == "ASVspoof2021LA":
        eer = test_on_ASVspoof2021LA(model_path, loss_model_path, "eval", args.loss)
    elif args.task == "singfake":
        sets = ["train", "dev", "test", "additional_test", "codec_test/adts_64k", "codec_test/mp3_128k", "codec_test/ogg_64k", "codec_test/opus_64k", "persian_test"]
        for set in sets:
            print(set)
            eer = test_model_on_singfake(model_path, loss_model_path, os.path.join(args.path_to_database, set), args.loss, args)
    else:
        raise ValueError("Evaluation task unknown!")


