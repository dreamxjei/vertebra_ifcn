import numpy as np
import torch
import torch.nn as nn


class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()
        self.eps = 1e-9

    def forward(self, logits, target):
        # inputs are both shape N x K x H x W (N is batch size, K is # class, 26), 
        N, K, H, W = logits.shape  # should be 10, 26, 128, 128

        # Figure out how many classes there are in the target image for each batch
    #         _, label = torch.max(target, dim=1)
    #         label = label.cpu().numpy()
    #         N_label = np.zeros(N)
    #         for i in range(N):
    #             N_label[i] = np.unique(label[i,:,:]).shape[0]

        l_flat = logits.view(logits.size(0), logits.size(1), -1) # want to be size N x K x HW
        t_flat = target.view(target.size(0), target.size(1), -1) # want to be size N x K x HW
        intersection = torch.sum(torch.mul(l_flat, t_flat), dim=2) # N x K
        num = 2 * intersection # + self.eps

        den = torch.sum(l_flat, dim=2) + torch.sum(t_flat, dim=2) + self.eps # N x K
        dice_score = torch.div(num, den) # N x K

        # First mean over class manually, using N_label
    #         dice_score = torch.sum(dice_score, dim=1) / torch.from_numpy(N_label).to(device)
        # Now mean over batch
        dice_score = torch.mean(dice_score, dim=(0,1))  # scalar now

        loss = 1 +  -1.0*dice_score  # per-class loss
        loss.requires_grad_(True)
        return loss


def dice_score(preds, one_hot_target):  # , num_classes=num_classes):
    N, K, H, W = preds.shape
    num_classes = K
    
    #Not differentiating anything, so detach everything
    target_np = one_hot_target.clone().detach().cpu().numpy()
    preds_torch = preds.clone().detach().cpu()
    
    # one-hot encode preds to calculate denominator
    _, preds_torch = torch.max(preds_torch, dim=1)
    preds_onehot = nn.functional.one_hot(preds_torch, num_classes)
    preds_np = np.transpose(preds_onehot.numpy(), (0, 3, 1, 2)) # N x K x H x W

    num = 2*np.sum(((preds_np == target_np) & (target_np != 0) & (preds_np != 0)), axis=(0, 2, 3)) # sum up per class, so den is size (K,)

    den = np.sum(preds_np, axis=(0, 2, 3)) + np.sum(target_np, axis=(0, 2, 3)) # sum up per class, so den is size (K,)
    den[np.where(den == 0)] = 1e-9
    num[np.where(den == 0)] = 1e-9
    dice_coeff = num / den

    return(dice_coeff)


def seg_loss(pred, target, weight):
    FP = torch.sum(weight * (1 - target) * pred)
    FN = torch.sum(weight * (1 - pred) * target)
    return FP, FN


# use dice_score instead
# def DiceCoeff(pred, gt):
#     pred = pred.to('cpu').numpy()
#     gt = gt.to('cpu').numpy()

#     # if gt is all zero (use inverse to count)
#     if np.count_nonzero(gt) == 0:
#         gt = gt + 1
#         pred = 1 - pred

#     return dc(pred, gt)