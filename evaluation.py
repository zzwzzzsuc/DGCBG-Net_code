import torch
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
import numpy as np

def get_sensitivity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    TP = ((SR==1).byte()+(GT==1).byte())==2
    FN = ((SR==0).byte()+(GT==1).byte())==2

    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-8)
    
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    TN = ((SR==0).byte()+(GT==0).byte())==2
    FP = ((SR==1).byte()+(GT==0).byte())==2

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-8)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    TP = ((SR==1).byte()+(GT==1).byte())==2
    FP = ((SR==1).byte()+(GT==0).byte())==2

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-8)

    return PC

def get_TPR(SR,GT,threshold=0.5):

    SR = SR > threshold
    GT = GT == torch.max(GT)

    TP = ((SR==1).byte()+(GT==1).byte())==2
    FN = ((SR==0).byte()+(GT==1).byte())==2

    TPR = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-8)

    return TPR

def get_FPR(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    TN = ((SR==0).byte()+(GT==0).byte())==2
    FP = ((SR==1).byte()+(GT==0).byte())==2

    FPR = float(torch.sum(FP))/(float(torch.sum(TN+FP)) + 1e-8)

    return FPR

def get_JS(SR,GT,threshold=0.5):

    SR = SR > threshold
    GT = GT == torch.max(GT)
    Inter = torch.sum((SR.byte()+GT.byte())==2)
    Union = torch.sum((SR.byte()+GT.byte())>=1)
    
    JS = float(Inter)/(float(Union) + 1e-8)
    
    return JS

def get_DC(SR,GT,threshold=0.5):

    SR = SR.byte() > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR.byte() + GT.byte())== 2)
    DC = float(2 * Inter) / (float(torch.sum(SR.byte()) + torch.sum(GT.byte())) + 1e-8)

    return DC


def get_Recall(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    true_positives = torch.sum((GT.byte() == 1) & (SR.byte() == 1))
    false_negatives = torch.sum((GT.byte() == 1) & (SR.byte() == 0))

    recall = true_positives / (true_positives + false_negatives + 1e-8)
    return recall


def get_HD(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    gt_points = torch.nonzero(GT)
    sr_points = torch.nonzero(SR)
    gt_points_np = gt_points.cpu().numpy()
    sr_points_np = sr_points.cpu().numpy()
    hausdorff_dist = max(directed_hausdorff(gt_points_np, sr_points_np)[0],
                         directed_hausdorff(sr_points_np, gt_points_np)[0])

    return hausdorff_dist
