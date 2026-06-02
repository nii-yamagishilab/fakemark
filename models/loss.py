#!/usr/bin/env python
"""Loss functions and metrics for watermark training.

References:
    [1] Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Li.
        Energy-Based Out-of-Distribution Detection.
        In Proc. NeurIPS, 33:21464-21475. 2020.
    [2] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna,
        Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan.
        Supervised Contrastive Learning. Proc. NeurIPS. 2020.
    [3] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz.
        Mixup: Beyond Empirical Risk Minimization. In Proc. ICLR. 2018.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class LMCL_loss(nn.Module):
    """Large Margin Cosine Loss (CosFace).

    Computes cosine similarity between L2-normalised features and learnable
    class centres.  The margin ``m`` and scale ``s`` are applied downstream
    by the caller.

    Reference:
        Hao Wang et al., "CosFace: Large Margin Cosine Loss for Deep Face
        Recognition", CVPR 2018.

    Args:
        num_classes: Number of target classes.
        feat_dim: Dimensionality of input feature vectors.
        s: Scaling factor (default 7.0).
        m: Cosine margin (default 0.2).
    """

    def __init__(self, num_classes: int, feat_dim: int, s: float = 7.00, m: float = 0.2):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Compute cosine logits between features and class centres.

        Args:
            feat: Input features of shape ``(batch, feat_dim)``.

        Returns:
            Cosine similarity logits of shape ``(batch, num_classes)``.
        """
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        return logits

class AAMSoftmaxLoss(nn.Module):
    """Additive Angular Margin Softmax loss.

    Args:
        embedding_dim: Dimensionality of input embeddings.
        no_classes: Number of target classes.
        scale: Logit scaling factor (default 30.0).
        margin: Additive angular margin (default 0.4).
    """

    def __init__(self, embedding_dim: int, no_classes: int, scale: float = 30.0, margin: float = 0.4):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.no_classes = no_classes
        self.embedding = nn.Embedding(no_classes, embedding_dim, max_norm=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute AAM-Softmax loss and scaled logits.

        Args:
            x: Input embeddings of shape ``(N, embedding_dim)``.
            labels: Ground-truth class indices of shape ``(N,)``.

        Returns:
            Tuple of (loss scalar, logits of shape ``(N, no_classes)``).
        """
        n, m = x.shape
        assert n == len(labels)
        assert m == self.embedding_dim
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.no_classes

        x = F.normalize(x, dim=1)
        w = self.embedding.weight
        cos_theta = torch.matmul(w, x.T).T
        psi = cos_theta - self.margin

        onehot = F.one_hot(labels, self.no_classes)
        logits = self.scale * torch.where(onehot == 1, psi, cos_theta)
        loss = self.loss(logits, labels)
        return loss, logits


def freq_mag_loss(wav_target: torch.Tensor, wav_estimated: torch.Tensor) -> torch.Tensor:
    """Spectral-shape KL divergence loss in the frequency domain.

    Computes the STFT magnitude of both waveforms, averages across time to
    obtain frequency envelopes, normalises them to unit-sum distributions,
    and returns their KL divergence.

    Args:
        wav_target: Target waveform of shape ``(B, T)`` or ``(B, 1, T)``.
        wav_estimated: Estimated waveform with the same shape.

    Returns:
        Scalar KL-divergence loss averaged over the batch.
    """
    if wav_target.ndim == 3 and wav_target.size(1) == 1:
        wav_target = wav_target.squeeze(1)
    if wav_estimated.ndim == 3 and wav_estimated.size(1) == 1:
        wav_estimated = wav_estimated.squeeze(1)
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    window = torch.hann_window(win_length).to(wav_target.device)
    # target mag
    spec_target = torch.stft(
        input=wav_target, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=True, return_complex=True
    )
    mag_target = torch.abs(spec_target).mean(dim=2)
    # estimated mag
    spec_estimated = torch.stft(
        input=wav_estimated, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=True, return_complex=True
    )
    mag_estimated = torch.abs(spec_estimated).mean(dim=2)
    # normalize them so that only their shape matters, not their amplitude
    eps = 1e-8
    shape_norm_target = mag_target / (mag_target.sum(dim=1, keepdim=True) + eps)
    shape_norm_estimated = mag_estimated / (mag_estimated.sum(dim=1, keepdim=True) + eps)
    # loss
    loss = torch.sum(
        shape_norm_target * torch.log(shape_norm_target / shape_norm_estimated),
        dim=1
    ).mean()
    return loss


def time_mag_loss(
    wav_target: torch.Tensor,
    wav_estimated: torch.Tensor,
    wav_len: torch.Tensor,
) -> torch.Tensor:
    """Spectral-shape KL divergence loss in the time domain with length masking.

    Similar to :func:`freq_mag_loss`, but averages STFT magnitude across
    frequency to obtain a temporal energy envelope and masks out padded
    regions using ``wav_len``.

    Args:
        wav_target: Target waveform of shape ``(B, T)`` or ``(B, 1, T)``.
        wav_estimated: Estimated waveform with the same shape.
        wav_len: Per-sample waveform lengths of shape ``(B,)``.

    Returns:
        Scalar KL-divergence loss averaged over the batch.
    """
    if wav_target.ndim == 3 and wav_target.size(1) == 1:
        wav_target = wav_target.squeeze(1)
    if wav_estimated.ndim == 3 and wav_estimated.size(1) == 1:
        wav_estimated = wav_estimated.squeeze(1)

    n_fft = 1024
    hop_length = 256
    win_length = 1024
    window = torch.hann_window(win_length).to(wav_target.device)

    # STFTs
    spec_target = torch.stft(
        input=wav_target, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=True, return_complex=True
    )   # [B, F, T]
    spec_estimated = torch.stft(
        input=wav_estimated, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=True, return_complex=True
    )

    mag_target = torch.abs(spec_target).mean(dim=1)     # [B, T]
    mag_estimated = torch.abs(spec_estimated).mean(dim=1) # [B, T]

    # ----- build mask from wav_len -----
    B, T = mag_target.shape
    device = wav_target.device
#     frame_lens = torch.div(wav_len, hop_length, rounding_mode="floor")  # frame length per sample
    frame_lens = torch.floor((wav_len - 1) / hop_length).long() + 1
    mask = torch.arange(T).unsqueeze(0) < frame_lens.unsqueeze(1)  # [B, T]
    mask = mask[:, :T].to(device)

    # apply mask
    mag_target = mag_target * mask
    mag_estimated = mag_estimated * mask

    # normalize over *only valid frames*
    eps = 1e-8
    shape_norm_target = mag_target / (mag_target.sum(dim=1, keepdim=True) + eps)
    shape_norm_estimated = mag_estimated / (mag_estimated.sum(dim=1, keepdim=True) + eps)

    # safe ratio
    ratio = (shape_norm_target + eps) / (shape_norm_estimated + eps)

    # KL divergence–like loss
    loss = torch.sum(shape_norm_target * torch.log(ratio), dim=1).mean()

    return loss
    

#####################
# negative energy [1]
#####################

def neg_energy(logits, temperature=1):
    """ neg_eng = neg_energy(logits, temperature=1)

    neg_eng[x] =  -T \log \sum_y \exp (logits[x, y] / T)

    See [1]

    input
    -----
      logits: tensor, shape (batch, dim)
      temperature: float, temperature hyperparameter
    
    output
    ------
      neg_eng: tensor, shape (batch,)
    """
    eng = - temperature * torch.logsumexp(logits / temperature, dim=1)
    return eng


def neg_energy_reg_loss(energy, margin_in, margin_out, flag_in):
    """ loss = neg_energy_reg_loss(energy, margin_in, margin_out, flag_in)

    See [1] eqs.(8-9)
    
    input
    -----
      energy: tensor, any shape is OK
      margin_in: float, margin for the in-dist. data
      margin_out: float, margin for the out-dist. data
      flag_in: bool, if the input data is in-dist. data

    output
    ------
      loss: scalar
    """
    if flag_in:
        loss = torch.pow(F.relu(energy - margin_in), 2).mean()
    else:
        loss = torch.pow(F.relu(margin_out - energy), 2).mean()
    return loss


#####################
# supervised contrastive loss [2]
#####################

def supcon_loss(input_feat, 
               labels = None, mask = None, sim_metric = None, 
               t=0.07, contra_mode='all', length_norm=False):
    """
    loss = SupConLoss(feat, 
                      labels = None, mask = None, sim_metric = None, 
                      t=0.07, contra_mode='all')
    input
    -----
      feat: tensor, feature vectors z [bsz, n_views, ...].
      labels: ground truth of shape [bsz].
      mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
      sim_metric: func, function to measure the similarity between two 
            feature vectors, default, inner product
            when sim_metric=None, length_norm=True, it is cosine
      t: float, temperature
      contra_mode: str, default 'all'
         'all': use all data in class i as anchors
         'one': use 1st data in class i as anchors
      length_norm: bool, default False
          if True, l2 normalize feat along the last dimension

    output
    ------
      A loss scalar.
        
    Based on https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.

    Example:
      feature = torch.rand([16, 2, 1000], dtype=torch.float32)
      feature = F.normalize(feature, dim=-1)
      label = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 1, 1, 1, 1, 1], 
               dtype=torch.long)
      loss = supcon_loss(feature, labels=label)
    """
    if length_norm:
        feat = F.normalize(input_feat, dim=-1)
    else:
        feat = input_feat
        
    # batch size
    bs = feat.shape[0]
    # device
    dc = feat.device
    # dtype
    dt = feat.dtype
    # number of view
    nv = feat.shape[1]
    
    # get the mask
    # mask[i][:] indicates the data that has the same class label as data i
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(bs, dtype=dt, device=dc)
    elif labels is not None:
        labels = labels.view(-1, 1)
        if labels.shape[0] != bs:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).type(dt).to(dc)
    else:
        mask = mask.type(dt).to(dc)
    
    # prepare feature matrix
    # -> (num_view * batch, feature_dim, ...)
    contrast_feature = torch.cat(torch.unbind(feat, dim=1), dim=0)
    
    # 
    if contra_mode == 'one':
        # (batch, feat_dim, ...)
        anchor_feature = feat[:, 0]
        anchor_count = 1
    elif contra_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = nv
    else:
        raise ValueError('Unknown mode: {}'.format(contra_mode))
    
    # compute logits
    # logits_mat is a matrix of size [num_view * batch, num_view * batch]
    # or [batch, num_view * batch]
    if sim_metric is not None:
        logits_mat = torch.div(
            sim_metric(anchor_feature, contrast_feature), t)
    else:
        logits_mat = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), t)
    
    # mask based on the label
    # -> same shape as logits_mat 
    mask_ = mask.repeat(anchor_count, nv)
    # mask on each data itself (
    self_mask = torch.scatter(
        torch.ones_like(mask_), 1, 
        torch.arange(bs * anchor_count).view(-1, 1).to(dc), 
        0)
    
    # 
    mask_ = mask_ * self_mask
    
    # for numerical stability, remove the max from logits
    # see https://en.wikipedia.org/wiki/LogSumExp trick
    # for numerical stability
    logits_max, _ = torch.max(logits_mat * self_mask, dim=1, keepdim=True)
    logits_mat_ = logits_mat - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits_mat_ * self_mask) * self_mask
    log_prob = logits_mat_ - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_ * log_prob).sum(1) / mask_.sum(1)

    # loss
    loss = - mean_log_prob_pos
    loss = loss.view(anchor_count, bs).mean()

    return loss


class MixUpCE(nn.Module):
    """Mixup cross-entropy loss.

    Computes ``gamma * CE(logits, y1) + (1 - gamma) * CE(logits, y2)`` where
    the logits are assumed to come from a mixup-augmented input.  Falls back
    to standard cross-entropy when ``y2`` and ``gammas`` are both ``None``.

    Args:
        weight: Optional class weight tensor forwarded to
            :class:`torch.nn.CrossEntropyLoss`.
    """

    def __init__(self, weight: torch.Tensor | None = None):
        super().__init__()
        self.m_loss1 = nn.CrossEntropyLoss(weight=weight, reduction='none')
        self.m_loss2 = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(
        self,
        logits: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor | None = None,
        gammas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the (optionally mixup-weighted) cross-entropy loss.

        Args:
            logits: Predicted logits of shape ``(batch, dim)``.
            y1: Primary target labels of shape ``(batch,)``.
            y2: Secondary target labels of shape ``(batch,)``, or ``None``.
            gammas: Per-sample mixup coefficients of shape ``(batch,)``, or
                ``None``.

        Returns:
            Scalar loss averaged over the batch.
        """
        if y2 is None and gammas is None:
            loss_val = self.m_loss1(logits, y1)
        else:
            loss_val = gammas * self.m_loss1(logits, y1)
            loss_val += (1 - gammas) * self.m_loss2(logits, y2)
        return loss_val.mean()



#####################
# Distillation related
#####################

def kld_distill(logits_s, logits_t, temp=20):
    """ KLD-based distillation loss

    input
    -----
      logits_s: tensor, (batch, ..., dim), student output logits
                where dim is #. output categories
      logits_t: tensor, (batch, ..., dim), teacher output logits
      temp: float, temperature, default=20

    output
    ------
      loss: scalar
    """
    
    KD_loss = F.kl_div(
        F.log_softmax(logits_s / temp, dim = -1),
        F.log_softmax(logits_t / temp, dim = -1),
        reduction = 'batchmean',
        log_target = True) * temp * temp

    return KD_loss

#####################
# Rank consistency 
#####################
def rank_consistency(x, metric = None, anchor = None, diff_mat = None):
    """loss = rank_consistency(x, metric)
    
    input
    -----
      x: tensor, (batch, D1, D2 ...)
      metric: a function or None
      
              This function must be f(x1, x2) -> scalar
              where x1 and x2 are in shape (D1, D2 ...)
              
              if None, negative cosine similarity for
              x1 and x2 of shape (D1, )
      anchor: tensor, (batch, D1, D2, ...), as anchor
              or None

              If None, one of difference vector in the
              matrix will be selected as anchor
    
      diff_mat: tensor, (batch, batch, D1, D2 ...)
                of None
                if diff_mat is provided, x will be ignored
    output
    ------
      loss: scalar, loss value
      
    Example
    -------
      >> x = torch.randn(4, 10)
      >> x[1] = x[0] + 1.0
      >> x[2] = x[0] + 2.0
      >> x[3] = x[0] + 3.0
      >> rank_consistency(x)
      tensor(-1.)
    """
    
    if diff_mat is None:
        # (batch, batch, dim)
        # diff_mat[i, j]  = x[j] - x[i]
        diff_mat = x - x.unsqueeze(1)
        
    # batch size
    bs = diff_mat.shape[0]

    # loss to be accumulated
    loss = 0.0
    
    # metric
    if metric is None:
        # default use negative cosine_similarity
        metric = lambda x1, x2: -F.cosine_similarity(x1, x2, dim=0)
    
    #
    if bs < 3:
        return loss

    # get anchor
    if anchor is None:
        # choose the diff_mat[1, 0] as the anchor
        anchor_row_idx = 1
        anchor_col_idx = 0
        anchor = diff_mat[anchor_row_idx, anchor_col_idx]
    else:
        # anchor is provided externally
        anchor_row_idx = -1
        anchor_col_idx = -1


    # loop over the matrix, compare the off-diagnoal elements 
    # with the anchor
    count = 0.0
    for col_idx in np.arange(bs-1):
        for row_idx in np.arange(col_idx+1, bs):
            if col_idx == anchor_col_idx and anchor_row_idx == row_idx:
                continue
            loss += metric(anchor, diff_mat[row_idx, col_idx])
            count += 1
            
    loss = loss / count    
    return loss


def rank_consistency_v2(x, metric = None, diff_mat = None):
    """loss = rank_consistency_v2(x, metric)
    
    input
    -----
      x: tensor, (batch, D1, D2 ...)
      metric: a function or None
      
              This function must be f(x1, x2) -> scalar
              where x1 and x2 are in shape (D1, D2 ...)
              
              if None, negative cosine similarity for
              x1 and x2 of shape (D1, )
      
      diff_mat: tensor, (batch, batch, D1, D2 ...)
                of None
                if diff_mat is provided, x will be ignored
    output
    ------
      loss: scalar, loss value
      
    Example
    -------
      >> x = torch.randn(4, 10)
      >> x[1] = x[0] + 1.0
      >> x[2] = x[0] + 2.0
      >> x[3] = x[0] + 3.0
      >> metric = lambda x1, x2: \
         F.margin_ranking_loss(x1, x2, torch.ones_like(x1), 0.1)
      >> rank_consistencyV2(x, metric)
      tensor(.0)
    """
    
    if diff_mat is None:
        # (batch, batch, dim)
        # diff_mat[i, j]  = x[j] - x[i]
        diff_mat = x - x.unsqueeze(1)
    
    # batch size
    bs = diff_mat.shape[0]

    # loss to be accumulated
    loss = 0.0
    
    # metric
    if metric is None:
        # default use margin_ranking_loss
        metric = lambda x1, x2: F.margin_ranking_loss(
            x1, x2, torch.ones_like(x1), 0.1)
    
    #
    if bs < 3:
        return loss

    count = 0.0
    # loop over the matrix, column first
    for col_idx in np.arange(bs-2):
        for row_idx in np.arange(col_idx+2, bs):
            # assume diff[i, j] should be diff[i-1, j]
            loss += metric(diff_mat[row_idx-1, col_idx], 
                           diff_mat[row_idx, col_idx])
            count += 1
            
    # loop over the matrix, column first
    for row_idx in np.arange(2, bs):
        for col_idx in np.arange(1, row_idx):
            # assume diff[i, j] should be diff[i, j-1]
            loss += metric(diff_mat[row_idx, col_idx], 
                           diff_mat[row_idx, col_idx-1])
            count += 1
            
    loss = loss / count    
    return loss


def rank_consistency_v3(x, metric = None):
    """loss = rank_consistency_v3(x, metric)
    
    input
    -----
      x: tensor, (batch, D1, D2 ...)
      metric: a function or None
      
              This function must be f(x1, x2) -> scalar
              where x1 and x2 are in shape (D1, D2 ...)
              
              if None, negative cosine similarity for
              x1 and x2 of shape (D1, )

    output
    ------
      loss: scalar, loss value
      
    Example
    -------
      >> x = torch.randn(4, 10)
      >> x[1] = x[0] + 1.0
      >> x[2] = x[0] + 2.0
      >> x[3] = x[0] + 3.0
      >> metric = lambda x1, x2: \
         F.margin_ranking_loss(x1, x2, torch.ones_like(x1), 0.1)
      >> rank_consistency_v3(x, metric)
      tensor(.0)
    """
    # batch size
    bs = x.shape[0]

    # loss to be accumulated
    loss = 0.0
    
    # metric
    if metric is None:
        # default use margin_ranking_loss
        # x1 should be ranked higher
        metric = lambda x1, x2: F.margin_ranking_loss(
            x1, x2, torch.ones_like(x1), 0.1)
    
    #
    if bs < 2:
        return loss

    count = 0.0
    # loop over the rows
    for row_idx1 in np.arange(1, bs):
        for row_idx2 in np.arange(0, row_idx1):
            loss += metric(x[row_idx1], 
                           x[row_idx2])
            count += 1
                        
    loss = loss / count    
    return loss

