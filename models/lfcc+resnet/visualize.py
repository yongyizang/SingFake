import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from dataset import ASVspoof2019LA
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def visualize_dev_and_eval(dev_feat, dev_labels, dev_tags, eval_feat, eval_labels, eval_tags,
                           center, seed, out_fold):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
    c = ['#7030a0', '#ff0000', '#ffff00']

    torch.manual_seed(888)
    num_centers, enc_dim = center.shape
    ind_dev = torch.randperm(dev_feat.shape[0])[:5000].numpy()
    ind_eval = torch.randperm(eval_feat.shape[0])[:5000].numpy()

    dev_feat_sample = dev_feat[ind_dev]
    eval_feat_sample = eval_feat[ind_eval]
    dev_lab_sam = dev_labels[ind_dev]
    eval_lab_sam = eval_labels[ind_eval]

    X = np.concatenate((center, dev_feat_sample, eval_feat_sample), axis=0)
    os.environ['PYTHONHASHSEED'] = str(668)
    np.random.seed(668)
    X_tsne = TSNE(random_state=seed, perplexity=40, early_exaggeration=40).fit_transform(X)
    center = X_tsne[:num_centers]
    feat_dev = X_tsne[num_centers:num_centers + 5000]
    feat_eval = X_tsne[num_centers + 5000:]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    ex_ratio = pca.explained_variance_ratio_
    center_pca = X_pca[:num_centers]
    feat_pca_dev = X_pca[num_centers:num_centers + 5000]
    feat_pca_eval = X_pca[num_centers + 5000:]

    # t-SNE visualization
    ax1.plot(feat_dev[dev_lab_sam == 0, 0], feat_dev[dev_lab_sam == 0, 1], '.', c=c[0], markersize=1.2)
    ax1.plot(feat_dev[dev_lab_sam == 1, 0], feat_dev[dev_lab_sam == 1, 1], '.', c=c[1], markersize=1.2)
    ax1.axis('off')
    ax1.plot(center[:, 0], center[:, 1], 'x', c=c[2], markersize=5)
    plt.setp((ax2), xlim=ax1.get_xlim(), ylim=ax1.get_ylim())
    ax2.plot(center[:, 0], center[:, 1], 'x', c=c[2], markersize=5)

    ax2.plot(feat_eval[eval_lab_sam == 0, 0], feat_eval[eval_lab_sam == 0, 1], '.', c=c[0], markersize=1.2)
    ax2.plot(feat_eval[eval_lab_sam == 1, 0], feat_eval[eval_lab_sam == 1, 1], '.', c=c[1], markersize=1.2)
    ax2.axis('off')

    # PCA visualization
    ax3.plot(feat_pca_dev[dev_lab_sam == 0, 0], feat_pca_dev[dev_lab_sam == 0, 1], '.', c=c[0], markersize=1.2)
    ax3.plot(feat_pca_dev[dev_lab_sam == 1, 0], feat_pca_dev[dev_lab_sam == 1, 1], '.', c=c[1], markersize=1.2)
    ax3.axis('off')
    plt.setp((ax4), xlim=ax3.get_xlim(), ylim=ax3.get_ylim())
    ax4.plot(feat_pca_eval[eval_lab_sam == 0, 0], feat_pca_eval[eval_lab_sam == 0, 1], '.', c=c[0], markersize=1.2)
    ax4.plot(feat_pca_eval[eval_lab_sam == 1, 0], feat_pca_eval[eval_lab_sam == 1, 1], '.', c=c[1], markersize=1.2)
    ax4.axis('off')
    plt.savefig(os.path.join(out_fold, '_vis_feat.pdf'), dpi=500, bbox_inches="tight")
    plt.show()
    fig.clf()
    plt.close(fig)

def get_features(feat_model_path, part):
    model = torch.load(feat_model_path)
    dataset = ASVspoof2019LA('/data/neil/DS_10283_3336/',
                             '/data2/neil/ASVspoof2019LA/', part,
                             "LFCC", feat_len=500)
    dataLoader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0,
                            collate_fn=dataset.collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    ip1_loader, tag_loader, idx_loader, score_loader = [], [], [], []
    for i, (feat, audio_fn, tags, labels, channel) in enumerate(tqdm(dataLoader)):
        feat = feat.transpose(2, 3).float().to(device)
        tags = tags.to(device)
        labels = labels.to(device)
        feats, _ = model(feat)
        ip1_loader.append(feats.detach().cpu().numpy())
        idx_loader.append((labels.detach().cpu().numpy()))
        tag_loader.append((tags.detach().cpu().numpy()))
    features = np.concatenate(ip1_loader, 0)
    labels = np.concatenate(idx_loader, 0)
    tags = np.concatenate(tag_loader, 0)
    gen_feats = features[labels == 0]
    return features, labels, tags


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device("cuda")

    model_dir = "/data3/neil/hbas/models1106/resnet_ocsoftmax_500"

    feat_model_path = os.path.join(model_dir, "anti-spoofing_feat_model.pt")
    loss_model_path = os.path.join(model_dir, "anti-spoofing_loss_model.pt")

    center = torch.load(loss_model_path).center.detach().cpu().numpy()
    dev_feat, dev_labels, dev_tags = get_features(feat_model_path, "dev")
    eval_feat, eval_labels, eval_tags = get_features(feat_model_path, "eval")

    visualize_dev_and_eval(dev_feat, dev_labels, dev_tags, eval_feat, eval_labels, eval_tags, center, 88, model_dir)

