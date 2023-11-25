import torch
import argparse
import os
import json
import shutil
import numpy as np
from model import *
from dataset import *
from torch.utils.data import DataLoader
from loss import *
from collections import defaultdict
from tqdm import tqdm
from utils import str2bool, setup_seed
import eval_metrics as em
import yaml

torch.set_default_tensor_type(torch.FloatTensor)

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed', type=int, help="random number seed", default=1000)

    # Data folder prepare
    parser.add_argument("-d", "--path_to_database", type=str, help="dataset path",
                        default='/data1/neil/DS_10283_3336/')
    parser.add_argument("-f", "--path_to_features", type=str, help="features path",
                        default='/data2/neil/ASVspoof2019LA/')
    parser.add_argument("-p", "--path_to_protocol", type=str, help="protocol path",
                        default='/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=True, default='./models/try/')

    # Dataset prepare
    parser.add_argument("--feat", type=str, help="which feature to use", default='LFCC',
                        choices=["CQCC", "LFCC", "Raw", "MFCC"])
    parser.add_argument("--feat_len", type=int, help="features length", default=64000)
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=256)

    parser.add_argument('-m', '--model', help='Model arch', default='resnet',
                        choices=['resnet', 'lcnn', 'rawnet'])

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=128, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=20, help="interval to decay lr")

    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="1")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers")

    parser.add_argument('-l', '--loss', type=str, default="ocsoftmax",
                        choices=["softmax", "amsoftmax", "ocsoftmax", "isolate", "scl", "angulariso"], help="loss for training")
    parser.add_argument('--weight_loss', type=float, default=0.5, help="weight for other loss")
    parser.add_argument('--m_real', type=float, default=0.9, help="m_real for ocsoftmax loss")
    parser.add_argument('--m_fake', type=float, default=0.2, help="m_fake for ocsoftmax loss")
    parser.add_argument('--r_real', type=float, default=25.0, help="r_real for isolate loss")
    parser.add_argument('--r_fake', type=float, default=75.0, help="r_fake for isolate loss")
    parser.add_argument('--alpha', type=float, default=20, help="scale factor for amsoftmax and ocsoftmax loss")
    parser.add_argument('--scale_factor', type=float, default=0.5, help="scale factor for single center loss")

    parser.add_argument('--continue_training', action='store_true', help="continue training with trained model")

    parser.add_argument('--test_on_eval', action='store_true',
                        help="whether to run EER on the evaluation set")
    parser.add_argument('--test_interval', type=int, default=5, help="test on eval for every how many epochs")
    parser.add_argument('--save_interval', type=int, default=5, help="save checkpoint model for every how many epochs")

    args = parser.parse_args()

    # Change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds
    setup_seed(args.seed)

    if args.continue_training:
        pass
    else:
        # Path for output data
        if not os.path.exists(args.out_fold):
            os.makedirs(args.out_fold)
        else:
            shutil.rmtree(args.out_fold)
            os.mkdir(args.out_fold)

        # Folder for intermediate results
        if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
            os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
        else:
            shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
            os.mkdir(os.path.join(args.out_fold, 'checkpoint'))

        # Path for input data
        # assert os.path.exists(args.path_to_database)
        assert os.path.exists(args.path_to_features)

        # Save training arguments
        with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
            file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

        with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
            file.write("Start recording training loss ...\n")
        with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
            file.write("Start recording validation loss ...\n")
        with open(os.path.join(args.out_fold, 'test_loss.log'), 'w') as file:
            file.write("Start recording test loss ...\n")

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

def adjust_learning_rate(args, lr, optimizer, epoch_num):
    lr = lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_lambda_(args, epoch_num):
    args.lambda_ = 2 / (1 + np.exp(- 0.001 * epoch_num)) - 1 + 1e-9

def shuffle(feat, tags, labels):
    shuffle_index = torch.randperm(labels.shape[0])
    feat = feat[shuffle_index]
    tags = tags[shuffle_index]
    labels = labels[shuffle_index]
    # this_len = this_len[shuffle_index]
    return feat, tags, labels

def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)

    # initialize model
    if args.model == 'resnet':
        feat_model = ResNet(11, args.enc_dim, resnet_type='18', nclasses=2).to(args.device)
    elif args.model == 'lcnn':
        feat_model = LCNN(4, args, nclasses=2).to(args.device)
    elif args.model == 'rawnet':
        assert args.feat == "Raw"
        with open("./model_config_RawNet.yml", 'r') as f_yaml:
            parser1 = yaml.safe_load(f_yaml)
            feat_model = RawNet(parser1["model"], args).to(args.device)

    if args.continue_training:
        feat_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt')).to(args.device)
    feat_optimizer = torch.optim.Adam(feat_model.parameters(), lr=args.lr,
                                      betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)

    training_set = ASVspoof2019LASpec(args.path_to_database, args.path_to_features, 'train',
                                    args.feat, feat_len=args.feat_len)
    validation_set = ASVspoof2019LASpec(args.path_to_database, args.path_to_features, 'dev',
                                    args.feat, feat_len=args.feat_len)

    trainDataLoader = DataLoader(training_set, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers, collate_fn=training_set.collate_fn)
    valDataLoader = DataLoader(validation_set, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers, collate_fn=validation_set.collate_fn)

    test_set = ASVspoof2019LASpec(args.path_to_database, args.path_to_features, "eval", args.feat,
                            feat_len=args.feat_len)
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=test_set.collate_fn)

    feat, _, _, _, _ = training_set[23]
    print("Feature shape", feat.shape)

    criterion = nn.CrossEntropyLoss().to(args.device)

    if args.loss == "ocsoftmax":
        ocsoftmax = OCSoftmax(args.enc_dim, m_real=args.m_real, m_fake=args.m_fake, alpha=args.alpha).to(args.device)
        ocsoftmax.train()
        ocsoftmax_optimizer = torch.optim.SGD(ocsoftmax.parameters(), lr=args.lr)
    elif args.loss == "isolate":
        iso_loss = IsolateLoss(2, args.enc_dim, r_real=args.r_real, r_fake=args.r_fake).to(args.device)
        iso_loss.train()
        iso_optimizer = torch.optim.SGD(iso_loss.parameters(), lr=args.lr)
    elif args.loss == "scl":
        scl_loss = SingleCenterLoss(2, args.enc_dim, m=args.scale_factor).to(args.device)
        scl_loss.train()
        scl_optimizer = torch.optim.SGD(scl_loss.parameters(), lr=args.lr)
    elif args.loss == "amsoftmax":
        amsoftmax_loss = AMSoftmax(2, args.enc_dim, s=args.alpha, m=args.m_real).to(args.device)
        amsoftmax_loss.train()
        amsoftmax_optimizer = torch.optim.SGD(amsoftmax_loss.parameters(), lr=0.01)
    elif args.loss == "angulariso":
        angulariso = AngularIsoLoss(args.enc_dim, m_real=args.m_real, m_fake=args.m_fake, alpha=args.alpha).to(args.device)
        angulariso.train()
        angulariso_optimizer = torch.optim.SGD(angulariso.parameters(), lr=args.lr)

    early_stop_cnt = 0
    prev_loss = 1e8

    monitor_loss = args.loss

    for epoch_num in tqdm(range(args.num_epochs)):
        genuine_feats, ip1_loader, tag_loader, idx_loader = [], [], [], []
        feat_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)
        testlossDict = defaultdict(list)
        adjust_learning_rate(args, args.lr, feat_optimizer, epoch_num)
        if args.loss == "ocsoftmax":
            adjust_learning_rate(args, args.lr, ocsoftmax_optimizer, epoch_num)
        elif args.loss == "isolate":
            adjust_learning_rate(args, args.lr, iso_optimizer, epoch_num)
        elif args.loss == "scl":
            adjust_learning_rate(args, args.lr, scl_optimizer, epoch_num)
        elif args.loss == "amsoftmax":
            adjust_learning_rate(args, args.lr, amsoftmax_optimizer, epoch_num)
        elif args.loss == "angulariso":
            adjust_learning_rate(args, args.lr, angulariso_optimizer, epoch_num)
        adjust_lambda_(args, epoch_num)
        print('\nEpoch: %d ' % (epoch_num + 1))
        correct_m, total_m, correct_c, total_c, correct_v, total_v = 0, 0, 0, 0, 0, 0

        for i, (feat, audio_fn, tags, labels, channel) in enumerate(tqdm(trainDataLoader)):
            ## data prep
            if args.feat == "Raw":
                feat = feat.to(args.device)
            else:
                feat = feat.transpose(2,3).to(args.device)
            tags = tags.to(args.device)
            labels = labels.to(args.device)

            # Train the embedding network
            ## forward
            feats, feat_outputs = feat_model(feat)

            ## loss calculate
            if args.loss == "softmax":
                feat_loss = criterion(feat_outputs, labels)
            elif args.loss == "ocsoftmax":
                ocsoftmaxloss, _ = ocsoftmax(feats, labels)
                feat_loss = ocsoftmaxloss
            elif args.loss == "isolate":
                isoloss, _ = iso_loss(feats, labels)
                feat_loss = isoloss
            elif args.loss == "scl":
                sclloss, _ = scl_loss(feats, labels)
                feat_loss = criterion(feat_outputs, labels) + sclloss * args.weight_loss
            elif args.loss == "amsoftmax":
                outputs, moutputs = amsoftmax_loss(feats, labels)
                feat_loss = criterion(moutputs, labels)
            elif args.loss == "angulariso":
                angularisoloss, _ = angulariso(feats, labels)
                feat_loss = angularisoloss

            ## backward
            if args.loss == "softmax":
                trainlossDict[args.loss].append(feat_loss.item())
                feat_optimizer.zero_grad()
                feat_loss.backward()
                feat_optimizer.step()
            elif args.loss == "ocsoftmax":
                ocsoftmax_optimizer.zero_grad()
                trainlossDict[args.loss].append(ocsoftmaxloss.item())
                feat_optimizer.zero_grad()
                feat_loss.backward()
                feat_optimizer.step()
                ocsoftmax_optimizer.step()
            elif args.loss == "isolate":
                iso_optimizer.zero_grad()
                trainlossDict[args.loss].append(isoloss.item())
                feat_optimizer.zero_grad()
                feat_loss.backward()
                feat_optimizer.step()
                iso_optimizer.step()
            elif args.loss == "scl":
                scl_optimizer.zero_grad()
                trainlossDict[args.loss].append(sclloss.item())
                feat_optimizer.zero_grad()
                feat_loss.backward()
                feat_optimizer.step()
                scl_optimizer.step()
            elif args.loss == "amsoftmax":
                trainlossDict[args.loss].append(feat_loss.item())
                feat_optimizer.zero_grad()
                amsoftmax_optimizer.zero_grad()
                feat_loss.backward()
                feat_optimizer.step()
                amsoftmax_optimizer.step()
            elif args.loss == "angulariso":
                angulariso_optimizer.zero_grad()
                trainlossDict[args.loss].append(angularisoloss.item())
                feat_optimizer.zero_grad()
                feat_loss.backward()
                feat_optimizer.step()
                angulariso_optimizer.step()

            ## record
            ip1_loader.append(feats)
            idx_loader.append((labels))
            tag_loader.append((tags))


            with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(i) + "\t" +
                            str(trainlossDict[monitor_loss][-1]) + "\n")


        # Val the model
        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        feat_model.eval()
        with torch.no_grad():
            ip1_loader, tag_loader, idx_loader, score_loader = [], [], [], []
            for i, (feat, audio_fn, tags, labels, channel) in enumerate(tqdm(valDataLoader)):
                if args.feat == "Raw":
                    feat = feat.to(args.device)
                else:
                    feat = feat.transpose(2, 3).to(args.device)

                tags = tags.to(args.device)
                labels = labels.to(args.device)
                feat, tags, labels = shuffle(feat, tags, labels)
                feats, feat_outputs = feat_model(feat)

                if args.loss == "softmax":
                    feat_loss = criterion(feat_outputs, labels)
                    score = F.softmax(feat_outputs, dim=1)[:, 0]
                    devlossDict[args.loss].append(feat_loss.item())
                elif args.loss == "ocsoftmax":
                    ocsoftmaxloss, score = ocsoftmax(feats, labels)
                    devlossDict[args.loss].append(ocsoftmaxloss.item())
                elif args.loss == "isolate":
                    isoloss, score = iso_loss(feats, labels)
                    devlossDict[args.loss].append(isoloss.item())
                elif args.loss == "scl":
                    sclloss, score = scl_loss(feats, labels)
                    sclloss = criterion(feat_outputs, labels) + sclloss * args.weight_loss
                    devlossDict[args.loss].append(sclloss.item())
                elif args.loss == "amsoftmax":
                    outputs, moutputs = amsoftmax_loss(feats, labels)
                    feat_loss = criterion(moutputs, labels)
                    score = F.softmax(outputs, dim=1)[:, 0]
                    devlossDict[args.loss].append(feat_loss.item())
                elif args.loss == "angulariso":
                    angularisoloss, score = angulariso(feats, labels)
                    devlossDict[args.loss].append(angularisoloss.item())

                ip1_loader.append(feats)
                idx_loader.append((labels))
                tag_loader.append((tags))

                score_loader.append(score)

            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()
            eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

            with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" +
                            str(np.nanmean(devlossDict[monitor_loss])) + "\t" +
                            str(eer) +"\n")
            print("Val EER: {}".format(eer))


        if args.test_on_eval:
            if (epoch_num + 1) % args.test_interval == 0:
                with torch.no_grad():
                    ip1_loader, tag_loader, idx_loader, score_loader = [], [], [], []
                    for i, (feat, audio_fn, tags, labels, channel) in enumerate(tqdm(testDataLoader)):
                        if args.feat == "Raw":
                            feat = feat.to(args.device)
                        else:
                            feat = feat.transpose(2, 3).to(args.device)
                        tags = tags.to(args.device)
                        labels = labels.to(args.device)
                        feats, feat_outputs = feat_model(feat)
                        if args.loss == "softmax":
                            feat_loss = criterion(feat_outputs, labels)
                            score = F.softmax(feat_outputs, dim=1)[:, 0]
                            testlossDict[args.loss].append(feat_loss.item())
                        elif args.loss == "ocsoftmax":
                            ocsoftmaxloss, score = ocsoftmax(feats, labels)
                            testlossDict[args.loss].append(ocsoftmaxloss.item())
                        elif args.loss == "isolate":
                            isoloss, score = iso_loss(feats, labels)
                            testlossDict[args.loss].append(isoloss.item())
                        elif args.loss == "scl":
                            sclloss, score = scl_loss(feats, labels)
                            testlossDict[args.loss].append(sclloss.item())
                        elif args.loss == "amsoftmax":
                            outputs, moutputs = amsoftmax_loss(feats, labels)
                            feat_loss = criterion(moutputs, labels)
                            score = F.softmax(outputs, dim=1)[:, 0]
                            testlossDict[args.loss].append(feat_loss.item())
                        elif args.loss == "angulariso":
                            angularisoloss, score = angulariso(feats, labels)
                            testlossDict[args.loss].append(angularisoloss.item())

                        ip1_loader.append(feats)
                        idx_loader.append((labels))
                        tag_loader.append((tags))

                        score_loader.append(score)

                    scores = torch.cat(score_loader, 0).data.cpu().numpy()
                    labels = torch.cat(idx_loader, 0).data.cpu().numpy()
                    eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

                    with open(os.path.join(args.out_fold, "test_loss.log"), "a") as log:
                        log.write(str(epoch_num) + "\t" + str(np.nanmean(testlossDict[monitor_loss])) + "\t" + str(eer) + "\n")
                    print("Test EER: {}".format(eer))


        valLoss = np.nanmean(devlossDict[monitor_loss])
        if (epoch_num + 1) % args.save_interval == 0:
            # Save the model checkpoint
            torch.save(feat_model, os.path.join(args.out_fold, 'checkpoint',
                                                'anti-spoofing_feat_model_%d.pt' % (epoch_num + 1)))
            if args.loss == "ocsoftmax":
                loss_model = ocsoftmax
            elif args.loss == "isolate":
                loss_model = iso_loss
            elif args.loss == "scl":
                loss_model = scl_loss
            elif args.loss == "softmax":
                loss_model = None
            elif args.loss == "amsoftmax":
                loss_model = amsoftmax_loss
            elif args.loss == "angulariso":
                loss_model = angulariso
            else:
                print("What is your loss? You may encounter error.")
            torch.save(loss_model, os.path.join(args.out_fold, 'checkpoint',
                                                'anti-spoofing_loss_model_%d.pt' % (epoch_num + 1)))

        if valLoss < prev_loss:
            torch.save(feat_model, os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt'))
            if args.loss == "ocsoftmax":
                loss_model = ocsoftmax
            elif args.loss == "isolate":
                loss_model = iso_loss
            elif args.loss == "scl":
                loss_model = scl_loss
            elif args.loss == "softmax":
                loss_model = None
            elif args.loss == "amsoftmax":
                loss_model = amsoftmax_loss
            elif args.loss == "angulariso":
                loss_model = angulariso
            else:
                print("What is your loss? You may encounter error.")
            torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))

            prev_loss = valLoss
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt == 50:
            with open(os.path.join(args.out_fold, 'args.json'), 'a') as res_file:
                res_file.write('\nTrained Epochs: %d\n' % (epoch_num - 49))
            break


if __name__ == "__main__":
    args = initParams()
    train(args)

