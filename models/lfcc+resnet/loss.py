import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class OCSoftmax(nn.Module):
    def __init__(self, feat_dim=2, m_real=0.5, m_fake=0.2, alpha=20.0):
        super(OCSoftmax, self).__init__()
        self.feat_dim = feat_dim
        self.m_real = m_real
        self.m_fake = m_fake
        self.alpha = alpha
        self.center = nn.Parameter(torch.randn(1, self.feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        scores = x @ w.transpose(0,1)
        output_scores = scores.clone()

        scores[labels == 0] = self.m_real - scores[labels == 0]
        scores[labels == 1] = scores[labels == 1] - self.m_fake

        loss = self.softplus(self.alpha * scores).mean()

        return loss, output_scores.squeeze(1)


class AMSoftmax(nn.Module):
    def __init__(self, num_classes, enc_dim, s=20, m=0.5):
        super(AMSoftmax, self).__init__()
        self.enc_dim = enc_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.centers = nn.Parameter(torch.randn(num_classes, enc_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)
        margin_logits = self.s * (logits - y_onehot)

        return logits, margin_logits


class IsolateLoss(nn.Module):
    """Isolate loss.
        Reference:
        I. Masi, A. Killekar, R. M. Mascarenhas, S. P. Gurudatt, and W. AbdAlmageed, “Two-branch Recurrent Network for Isolating Deepfakes in Videos,” 2020, [Online]. Available: http://arxiv.org/abs/2008.03412.
        Args:
            num_classes (int): number of classes.
            feat_dim (int): feature dimension.
            r_real (float): small radius to keep real inside
            r_fake (float): large radius to keep fake outside
        """
    def __init__(self, num_classes=2, feat_dim=256, r_real=25, r_fake=75):
        super(IsolateLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake

        self.center = nn.Parameter(torch.randn(1, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        loss = F.relu(torch.norm(x[labels==0]-self.center, p=2, dim=1) - self.r_real).mean() \
               + F.relu(self.r_fake - torch.norm(x[labels==1]-self.center, p=2, dim=1)).mean()
        return loss, -torch.norm(x-self.center, p=2, dim=1)


class SingleCenterLoss(nn.Module):
    """Single-Center loss.
        Reference:
        Li, J., Xie, H., Li, J., Wang, Z., & Zhang, Y. (2021). Frequency-aware Discriminative Feature Learning Supervised by Single-Center Loss for Face Forgery Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 6458-6467).
        Args:
            num_classes (int): number of classes.
            feat_dim (int): feature dimension.
            m (float): scale factor that the margin is proportional to the square root of dimension
        """
    def __init__(self, num_classes=2, feat_dim=256, m=0.3):
        super(SingleCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.m = m

        self.center = nn.Parameter(torch.randn(1, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        m_nat = torch.norm(x[labels==0]-self.center, p=2, dim=1).mean()
        m_man = torch.norm(x[labels==1]-self.center, p=2, dim=1).mean()
        loss = m_nat + F.relu(m_nat - m_man + self.m * math.sqrt(self.feat_dim))
        return loss, -torch.norm(x-self.center, p=2, dim=1)


class AngularIsoLoss(nn.Module):
    def __init__(self, feat_dim=2, m_real=0.5, m_fake=0.2, alpha=20.0):
        super(AngularIsoLoss, self).__init__()
        self.feat_dim = feat_dim
        self.m_real = m_real
        self.m_fake = m_fake
        self.alpha = alpha
        self.center = nn.Parameter(torch.randn(1, self.feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        scores = x @ w.transpose(0,1)
        output_scores = scores.clone()

        scores[labels == 0] = self.m_real - scores[labels == 0]
        scores[labels == 1] = scores[labels == 1] - self.m_fake

        loss = self.softplus(self.alpha * scores[labels == 0]).mean() + \
               self.softplus(self.alpha * scores[labels == 1]).mean()

        return loss, output_scores.squeeze(1)
