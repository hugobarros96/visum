'''
THIS FILE CONTAINS EXAMPLES OF  LOSSES
'''
import numpy as np

import torch

def iou_pytorch(outputs: torch.Tensor, target: torch.Tensor):
    SMOOTH = 1e-12
    
    #outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs * target).sum()  # Will be zero if Truth=0 or Prediction=0
    union = (outputs + target).sum()    # Will be zzero if both are 0
    
    iou = (intersection) / (union - intersection + SMOOTH)  #Smooth  devision to avoid 0/0
    
    loss_iou = 1-iou
    return loss_iou

def get_AP(outputs: torch.Tensor, target: torch.Tensor):

        iou = range(0.5, 1, 0.5)

        for treshold in iou:

            outputs[outputs>threshold] = 0


            precision = (outputs * target).sum() / (target).sum() 
            recall = (outputs * target).sum() / (outputs).sum()

            if threshold == 0.5:
                ap50 = ap
            if threshold == 0.75:
                ap75 = ap

def bbox_loss(prediction, bbox):
    """bbox-loss: its a loss that aims for the segmentation predcition to be inside the bbox. It also regulates 
    the racio of the area between the bbox and segmentation to be as close as a previous calculated racio
        -->prediction: is a segemntation mask produced by the model as prediction (no batch accepted - only 1 image)
        -->bbox: is a image of zeros with the same dimension as prediction and with bbox (rectangles) filled with ones
    """
    SMOOTH = 1e-12
    area_prediction = len(torch.nonzero(prediction))
    area_bbox = len(torch.nonzero(bbox))

    intersection = (prediction * bbox).sum()
    union = (prediction + bbox).sum()    # Will be zzero if both are 0
    iou = (intersection) / (union - intersection + SMOOTH)

    loss_iou = (1-iou)

    # True racio= 0.46 if we consider 46 /100-46 = 46/54 = 0.85 it means that the loss iou, respecting the acio shoud be
    #1-0.85 = 0.15
    loss_racio =  torch.abs(loss_iou - 0.30) # penalizes prediction outside bbox and racio diferent from true racio
    racio = torch.abs((prediction * bbox).mean() - 0.46 * bbox.mean())

    out_pixels = torch.abs((prediction * (1-bbox)).mean())

    
    return loss_racio #+ out_pixels



