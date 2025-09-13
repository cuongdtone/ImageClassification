import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization"""
    
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_label = one_hot * self.confidence + (1 - one_hot) * self.smoothing / (self.num_classes - 1)
        loss = torch.sum(-smooth_label * F.log_softmax(pred, dim=1), dim=1)
        return loss.mean()


class CenterLoss(nn.Module):
    """Center loss for feature learning"""
    
    def __init__(self, num_classes, feat_dim, lambda_c=0.001):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
    
    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers.index_select(0, labels.long())
        loss = (features - centers_batch).pow(2).sum() / 2.0 / batch_size
        return self.lambda_c * loss


def get_loss_function(loss_type='cross_entropy', num_classes=None, **kwargs):
    """Factory function to get loss function"""
    
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss_type == 'focal':
        alpha = kwargs.get('alpha', 1)
        gamma = kwargs.get('gamma', 2)
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == 'label_smoothing':
        smoothing = kwargs.get('smoothing', 0.1)
        return LabelSmoothingLoss(num_classes=num_classes, smoothing=smoothing)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    