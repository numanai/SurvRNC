import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numba
from tqdm import tqdm

class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        elif self.distance_type == 'l1_vec':
            return (labels[None, :, :] - labels[:, None, :]).sum(dim=-1)
        else:
            raise ValueError(f"Unsupported distance type: {self.distance_type}")

class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            return -(features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        else:
            raise ValueError(f"Unsupported similarity type: {self.similarity_type}")

class RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        # features: [bs, 2, feat_dim]
        # labels: [bs, label_dim]

        features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        labels = labels.repeat(2, 1)  # [2bs, label_dim]

        label_diffs = self.label_diff_fn(labels) # [2bs, 2bs]
        logits = self.feature_sim_fn(features).div(self.t) # [2bs, 2bs]
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0] # n = 2bs

        # remove diagonal
        mask = 1 - torch.eye(n).to(logits.device)
        
        logits = logits.masked_select(mask.bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select(mask.bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select(mask.bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss

class RnCEHRLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCEHRLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        # features: [bs, feat_dim]
        # labels: [bs, label_dim]

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0] # n = 2bs

        # remove diagonal
        mask = 1 - torch.eye(n).to(logits.device)
        
        logits = logits.masked_select(mask.bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select(mask.bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select(mask.bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # bs
            pos_label_diffs = label_diffs[:, k]  # bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [bs, bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss

@numba.njit
def _func(case_ids, neg_mask):
    dict_caseid_to_neg_or_ignore_or_unknown = {
                                        -11: np.array([[11,1], [12,0], [13, 1], [1, 2], [2, 2], [3, 1]]), # case1
                                        11: np.array([[11,1], [12,0], [13, 1], [1, 2], [2, 2], [3, 1]]), # case1
                                        10: np.array([[11,2], [12,0], [13, 2], [1, 2], [2, 2], [3, 2]]), # case2
                                        -10: np.array([[11,2], [12,2], [13, 2], [1, 2], [2, 2], [3, 2]]), # case3
                                        1: np.array([[11,1], [12,2], [13, 2], [1, 2], [2, 2], [3, 2]]), # case4
                                        -1: np.array([[11,1], [12,0], [13, 2], [1, 2], [2, 2], [3, 2]]), # case5
                                        0: np.array([[11,2], [12,2], [13, 2], [1, 2], [2, 2], [3, 2]]),
    }   
    
    for j in range(neg_mask.shape[-1]):
        case_id = case_ids[j]
        old_values = neg_mask[j]
        mask = old_values == dict_caseid_to_neg_or_ignore_or_unknown[int(case_id)][:, :1]
        neg_mask[j] = (1 - mask.sum(axis=0)) * old_values + (mask * dict_caseid_to_neg_or_ignore_or_unknown[int(case_id)][:,1:]).sum(axis=0)
    return neg_mask

_case_ids = np.array([0., -11., 10.], dtype=np.float32)
_neg_mask = np.array([[0., 11., 3.], [0., 0., 3.], [0., 12., 0.]], dtype=np.float32)
_ = _func(_case_ids, _neg_mask)

class ProgRnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1_vec', feature_sim='l2'):
        super(ProgRnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, y_times, y_events):
        # features: [bs, feat_dim]
        # labels: [bs, label_dim]
        if len(y_times.shape) == 1:
            y_times = y_times[..., None]
            
        if len(y_events.shape) > 1:
            y_events = y_events.squeeze()

        # label_diffs = self.label_diff_fn(labels) # [bs, bs]
        logits = self.feature_sim_fn(features).div(self.t) # [bs, bs]
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp() # [bs, bs]
            
        time_diffs = self.label_diff_fn(y_times)
        signs = torch.sign(time_diffs).detach()

        # Total number of valid anchors (which have at least 1 positive pair having negative pairs) 
        Na = 0
        loss = 0.
        
        for k in range(y_events.shape[-1]): 
            pos_label_diffs = time_diffs[k, :]  # bs
            sign = signs[k, :]
            
            time_diff_base = pos_label_diffs.expand([len(pos_label_diffs), len(pos_label_diffs)])
            time_diff = time_diff_base.clone().detach()

            # 0's; anchor
            time_diff[pos_label_diffs == 0] = 0

            # 3's; right
            time_diff[time_diff_base > abs(pos_label_diffs.view(-1, 1))] = 3  

            # 2's; within circle
            time_diff[(abs(time_diff_base) <= abs(pos_label_diffs.view(-1, 1))) & (abs(time_diff_base) > 0)] = 2

            # 1's; left
            time_diff[-time_diff_base > abs(pos_label_diffs.view(-1, 1))] = 1  

            # 0's; positive pairs
            time_diff.fill_diagonal_(0)

            # get position matrix; hacky way
            # the constant should be more than 3 (here we're choosing 10)
            # we do this in order to have a unique value for every position-event combination
            neg_mask = y_events * 10 + time_diff

            # might not need 0-mask if filter out early on (or myb still need for diagonals...)
            zero_mask = time_diff != 0
            neg_mask *= zero_mask
            # remove anchor idx on row & col
            neg_mask = torch.cat((neg_mask[:k], neg_mask[k+1:]), dim=0)
            neg_mask = torch.cat((neg_mask[:,:k], neg_mask[:,k+1:]), dim=1)

            # hacky way to get unique case ids for each of the 6 cases
            case_ids = (y_events + 10 * y_events[k]) * sign
            case_ids = case_ids.detach().cpu().numpy() # PROGRNCLOSS RND (add .cpu())
            case_ids = np.delete(case_ids, k, axis=0)

            neg_mask = neg_mask.detach().cpu().numpy()
            neg_mask = _func(case_ids, neg_mask)
            neg_mask = torch.from_numpy(neg_mask).to(y_events.device)
            
            neg_mask_final1 = (neg_mask == 1)
            neg_mask_final2 = (neg_mask == 2) * 0.5
            neg_mask_final = neg_mask_final1 + neg_mask_final2
            
            # Number of positive pairs which have valid negative pairs
            valid_rows = neg_mask_final.detach().sum(axis=1) > 0
            # Np = (neg_mask_final.detach().sum(axis=1) > 0).sum()
            Np = valid_rows.sum()
            if Np < 1: # If we don't have any negative pair
                continue
            Na += 1
            
            # TODO: check below. copied from original RnC. Haven't tested yet
            pos_logits = logits[:, k]  # bs
            pos_logits = torch.cat((pos_logits[:k], pos_logits[k+1:]), dim=0)
            
            assert pos_logits.shape[0] == y_times.shape[0] - 1
            
            _exp_logits = torch.cat((exp_logits[:k], exp_logits[k+1:]), dim=0)
            _exp_logits = torch.cat((_exp_logits[:,:k], _exp_logits[:,k+1:]), dim=1)
            
            pos_log_probs = pos_logits[valid_rows] - torch.log((neg_mask_final[valid_rows] * _exp_logits[valid_rows]).sum(dim=-1))
            loss += -(pos_log_probs / Np).sum()

        loss /= Na
        return loss