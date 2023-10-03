"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA
from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list, Dataset_SingFake)
from evaluation import compute_eer
from utils import create_optimizer, seed_worker, set_seed, str_to_bool
from models.wav2vecAASIST import Wav2Vec2Model
warnings.filterwarnings("ignore", category=FutureWarning)

def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    prefix_2019 = "ASVspoof2019.{}".format(track)
    database_path = Path(config["database_path"])
    dev_trial_path = (database_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    eval_trial_path = (
        database_path /
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2019))

    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(
        track,
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    gpu_id = args.gpu
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    # model = get_model(model_config, device)
    model = get_wav2vec2_model(model_config, device).to(device)

    # define dataloaders
    trn_loader, dev_loader, eval_loader, additional_loader, persian_loader, mp3_loader, ogg_loader, aac_loader, opus_loader = get_singfake_loaders(args.seed, config)

    # evaluates pretrained model and exit script
    if args.eval:
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        print("Model loaded : {}".format(config["model_path"]))
        print("Evaluating on SingFake Eval Set...")
        train_eer = evaluate(trn_loader, model, device)
        test_eer = evaluate(eval_loader, model, device)
        additional_set_eer = evaluate(additional_loader, model, device)
        persian_eer = evaluate(persian_loader, model, device)
        mp3_eer = evaluate(mp3_loader, model, device)
        ogg_eer = evaluate(ogg_loader, model, device)
        aac_eer = evaluate(aac_loader, model, device)
        opus_eer = evaluate(opus_loader, model, device)
        average_codec_eer = (mp3_eer + aac_eer + ogg_eer + opus_eer) / 4.
        print("Done. train_eer: {:.2f} %, test_set_eer: {:.2f} %, additional_test_eer: {:.2f} %, persian_eer: {:.2f} %".format(train_eer * 100, test_eer * 100, additional_set_eer * 100, persian_eer * 100))
        print("Codec testing: Average EER: {:.2f} %, MP3 EER: {:.2f} %, OGG EER: {:.2f} %, AAC EER: {:.2f} %, OPUS EER: {:.2f} %".format(average_codec_eer * 100, mp3_eer * 100, ogg_eer * 100, aac_eer * 100, opus_eer * 100))
        sys.exit(0)
        
    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 1.
    best_eval_eer = 100.
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, config)
        
        train_eer = evaluate(trn_loader, model, device)
        dev_eer = evaluate(dev_loader, model, device)
        additional_eer = evaluate(additional_loader, model, device)
        
        print("DONE.\nLoss:{:.5f}, train_eer: {:2f} %, dev_eer: {:.2f} %, additional_test_eer: {:.2f} %".format(
            running_loss, train_eer * 100, dev_eer * 100, additional_eer * 100))
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("train_eer", train_eer, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("additional_eer", additional_eer, epoch)
        
        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            torch.save(model.state_dict(),
                       model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))

            # do evaluation whenever best model is renewed
            if str_to_bool(config["eval_all_best"]):
                eval_eer = evaluate(eval_loader, model, device)
                log_text = "epoch{:03d}, ".format(epoch)
                if eval_eer < best_eval_eer:
                    log_text += "best eer, {:.4f}%".format(eval_eer)
                    best_eval_eer = eval_eer
                    torch.save(model.state_dict(), model_save_path / "best.pth")
                    
                if len(log_text) > 0:
                    print(log_text)
                    f_log.write(log_text + "\n")

            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)

    print("Start final evaluation")
    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    eval_eer = evaluate(eval_loader, model, device)
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    f_log.write("EER: {:.3f} %".format(eval_eer * 100))
    f_log.close()

    torch.save(model.state_dict(),
               model_save_path / "swa.pth")

    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
        torch.save(model.state_dict(), model_save_path / "best.pth")
        
    print("Exp FIN. EER: {:.3f} %".format(
        best_eval_eer * 100))


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model

def get_wav2vec2_model(model_config: Dict, device: torch.device):
    model = Wav2Vec2Model(model_config, device)
    return model


def get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    track = config["track"]
    prefix_2019 = "ASVspoof2019.{}".format(track)

    trn_database_path = database_path / "ASVspoof2019_{}_train/".format(track)
    dev_database_path = database_path / "ASVspoof2019_{}_dev/".format(track)
    eval_database_path = database_path / "ASVspoof2019_{}_eval/".format(track)

    trn_list_path = (database_path /
                     "ASVspoof2019_{}_cm_protocols/{}.cm.train.trn.txt".format(
                         track, prefix_2019))
    dev_trial_path = (database_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    eval_trial_path = (
        database_path /
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2019))

    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    print("no. training files:", len(file_train))

    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=trn_database_path)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    _, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                is_train=False,
                                is_eval=False)
    print("no. validation files:", len(file_dev))

    dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev,
                                            base_dir=dev_database_path)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    file_eval = genSpoof_list(dir_meta=eval_trial_path,
                              is_train=False,
                              is_eval=True)
    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                             base_dir=eval_database_path)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return trn_loader, dev_loader, eval_loader

def get_singfake_loaders(seed: int, config: dict) -> List[torch.utils.data.DataLoader]:
    """
    Work in progress; no split is provided as of now.
    """
    base_dir = "/home/yongyi/split_0831"
    
    target_sr = float(config["target_sr"])
    vocals_only = str_to_bool(config["vocals_only"])
    
    is_mixture = not vocals_only
    
    train_set = Dataset_SingFake(base_dir=os.path.join(base_dir, "train"), is_mixture=is_mixture, target_sr=target_sr)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            num_workers=4,
                            generator=gen)
    
    dev_set = Dataset_SingFake(base_dir=os.path.join(base_dir, "dev"), is_mixture=is_mixture, target_sr=target_sr)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            num_workers=4,
                            pin_memory=True)
    
    eval_set = Dataset_SingFake(base_dir=os.path.join(base_dir, "test"), is_mixture=is_mixture, target_sr=target_sr)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             num_workers=4,
                             pin_memory=True)
    
    additional_set = Dataset_SingFake(base_dir=os.path.join(base_dir, "additional_test"), is_mixture=is_mixture, target_sr=target_sr)
    additional_loader = DataLoader(additional_set,
                                      batch_size=config["batch_size"],
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=4,
                                      pin_memory=True)
    
    persian_set = Dataset_SingFake(base_dir=os.path.join(base_dir, "persian_test"), is_mixture=is_mixture, target_sr=target_sr)
    persian_loader = DataLoader(persian_set,
                                batch_size=config["batch_size"],
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=4,
                                      pin_memory=True)
    
    mp3_test_set = Dataset_SingFake(base_dir=os.path.join(base_dir, "codec_test", "mp3_128k"), is_mixture=is_mixture, target_sr=target_sr)
    mp3_loader = DataLoader(mp3_test_set,
                            batch_size=config["batch_size"],
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=4,
                                      pin_memory=True)
    ogg_test_set = Dataset_SingFake(base_dir=os.path.join(base_dir, "codec_test", "ogg_64k"), is_mixture=is_mixture, target_sr=target_sr)
    ogg_loader = DataLoader(ogg_test_set,
                            batch_size=config["batch_size"],
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=4,
                                      pin_memory=True)
    aac_test_set = Dataset_SingFake(base_dir=os.path.join(base_dir, "codec_test", "adts_64k"), is_mixture=is_mixture, target_sr=target_sr)
    aac_loader = DataLoader(aac_test_set,
                            batch_size=config["batch_size"],
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=4,
                                      pin_memory=True)
    opus_test_set = Dataset_SingFake(base_dir=os.path.join(base_dir, "codec_test", "opus_64k"), is_mixture=is_mixture, target_sr=target_sr)
    opus_loader = DataLoader(opus_test_set,
                            batch_size=config["batch_size"],
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=4,
                                      pin_memory=True)
    
    return trn_loader, dev_loader, eval_loader, additional_loader, persian_loader, mp3_loader, ogg_loader, aac_loader, opus_loader
    


def evaluate(loader, model, device: torch.device):
    """
    Evaluate the model on the given loader, then return EER.
    """
    model.eval()
    # we save target (1) scores to target_scores, and non target (0) scores to nontarget_scores.
    target_scores = []
    nontarget_scores = []
    debug = False
    count = 0
    with torch.no_grad():
        for batch_x, batch_y in tqdm(loader, total=len(loader)):
            batch_x = batch_x.to(device)
            batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
            batch_y = batch_y.data.cpu().numpy().ravel()
            for i in range(len(batch_y)):
                if batch_y[i] == 1:
                    target_scores.append(batch_score[i])
                else:
                    nontarget_scores.append(batch_score[i])
            count += 1
            if count == 10 and debug:
                break
    
    eer, _ = compute_eer(target_scores, nontarget_scores)
    return eer


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    pbar = tqdm(trn_loader, total=len(trn_loader))
    for batch_x, batch_y in pbar:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        pbar.set_description("loss: {:.5f}, running loss: {:.5f}".format(
            batch_loss.item(), running_loss / num_total))
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="gpu id to use (default: 0)")
    main(parser.parse_args())
