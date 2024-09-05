import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvConLoss(nn.Module):
    def __init__(self, device, temperature):
        super().__init__()

        self.device = device
        self.temperature = temperature
    
    def forward(self, x_proj, pos_mask, neg_mask):
        x_proj = F.normalize(x_proj, dim=1)
        batch_size = x_proj.size(0)
        
        logits = torch.exp(torch.matmul(x_proj[:batch_size, :], x_proj.T) / self.temperature)
        pos_logits = logits * pos_mask
        neg_logits = logits * neg_mask

        loss = pos_logits / (pos_logits.sum(dim=1, keepdim=True) + neg_logits.sum(dim=1, keepdim=True))
        loss = -torch.log(torch.where(loss == 0.0, 1.0, loss))
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        
        return loss.mean()


class ModifiedConLoss(nn.Module):
    def __init__(self, device, temperature):
        super().__init__()

        self.device = device
        self.temperature = temperature
    
    def forward(self, x_proj, pos_mask, neg_mask):
        x_proj = F.normalize(x_proj, dim=1)
        batch_size = x_proj.size(0)
        
        logits = torch.exp(torch.matmul(x_proj[:batch_size, :], x_proj.T) / self.temperature)
        pos_logits = logits * pos_mask
        neg_logits = logits * neg_mask

        loss = pos_logits / (pos_logits + neg_logits.sum(dim=1, keepdim=True))
        loss = -torch.log(torch.where(loss == 0.0, 1.0, loss))
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        
        return loss.mean()


class SimCLRLoss(nn.Module):
    '''
    This code is modified based on the impelemtation by
    https://github.com/google-research/simclr
    '''
    def __init__(self, device, temperature, epsilon=-1e9):
        super().__init__()

        self.device = device
        self.temperature = temperature
        self.epsilon = epsilon

    def forward(self, proj_i, proj_j):
        ce = torch.nn.CrossEntropyLoss().to(self.device)

        proj_i = F.normalize(proj_i, dim=1)
        proj_j = F.normalize(proj_j, dim=1)
        batch_size = proj_i.size(0)
        mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)

        logits_ii = torch.matmul(proj_i, proj_i.T) / self.temperature
        logits_ii[mask] = self.epsilon
        logits_jj = torch.matmul(proj_j, proj_j.T) / self.temperature
        logits_jj[mask] = self.epsilon
        logits_ij = torch.matmul(proj_i, proj_j.T) / self.temperature
        logits_ji = torch.matmul(proj_j, proj_i.T) / self.temperature

        cat_i = torch.cat([logits_ij, logits_ii], 1)
        cat_j = torch.cat([logits_ji, logits_jj], 1)
        labels = torch.cat([mask, torch.zeros(size=mask.size(), device=self.device)], 1).requires_grad_(True)
        loss_i = ce(cat_i, torch.argmax(labels, dim=1))
        loss_j = ce(cat_j, torch.argmax(labels, dim=1))

        loss = loss_i + loss_j
        
        return loss


class SupConLoss(nn.Module):
    '''
    This code is modified based on the impelemtation by
    https://github.com/HobbitLong/SupContrast
    '''
    def __init__(self, device, temperature, n_views=2):
        super().__init__()

        self.device = device
        self.temperature = temperature
        self.n_views = n_views

    def forward(self, proj_i, proj_j, labels):
        proj_i = F.normalize(proj_i, dim=1)
        proj_j = F.normalize(proj_j, dim=1)
        batch_size = proj_i.size(0)
        
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)
        proj = torch.cat([proj_i, proj_j], dim=0)

        logits_ = torch.matmul(proj, proj.T) / self.temperature
        
        logits_max, _ = torch.max(logits_, dim=1, keepdim=True)
        logits = logits_ - logits_max.detach()

        mask = mask.repeat(self.n_views, self.n_views)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * self.n_views).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = (-1 * mean_log_prob_pos).mean()

        return loss
