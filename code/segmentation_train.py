#%%
from segmentation_function import segNet, loader
from dataload.get_data import findExtension
import wandb
from tqdm import tqdm
import torch
import argparse
import csv
import numpy as np
from datetime import datetime

def options():
    parser = argparse.ArgumentParser(description='Segmentation')
    
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    parser.add_argument('--exp_name', type=str, default='exp_seg_'+str(current_time), metavar='N', help='Name of the experiment')
    # Important settings for on training
    parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N', help='mini-batch size (default: 2)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
   
    #DEFINE DATA
    parser.add_argument('--csv_train_path', type=str, default='data/train.csv', metavar='PATH', help='csv for train dataset')
    parser.add_argument('--csv_val_path', type=str, default='data/val.csv', metavar='PATH', help='csv for val dataset')
    
    #DEFINE NETWORK
    parser.add_argument('--dimension', nargs=2, default=(128, 256), metavar='HIGHT WIDTH', help='hight and width to resize the input into the net. For resunet needs to be 2^n ')
    parser.add_argument('--net', default='resunet', metavar='N', help='type of netwrok model resunet or dispnet')

    #USE BBOX
    parser.add_argument('--use_bbox', default=False, type=bool, metavar='N', help='use or not bbox loss and annotations')
    parser.add_argument('--json_bbox_train', default=[], metavar='N', help='json for bbox annotations')
    parser.add_argument('--json_bbox_val', default=[], metavar='N', help='json for bbox annotations')

    #DEFINE LOSS
    parser.add_argument('--loss', default='iou', metavar='N', help='iou or cross_entropy')
    parser.add_argument('--num_classes', default=3, type=int, metavar='N', help='number of classes, indicate in case loss is cross_entropy')
    
    #OTHERS
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--pretrained', default='', type=str, metavar='PATH', help='path to pretrained model file (default: null (no-use))')
    parser.add_argument('--n_filters', default=16, type=int, metavar='N', help='number of channel for first conv')
    parser.add_argument('--num_workers', default=0, type=int, metavar='N', help='number workers')
    
    args = parser.parse_args()
    return args
#%%%

args = options()

assert (args.loss == 'iou' or args.loss == 'cross_entropy'), 'loss can only be iou or cross_entropy'
assert (args.net == 'resunet' or args.net == 'dispnet'), 'net can only be resunet or dispnet'
if args.loss == 'iou':
    assert args.num_classes < 4, 'if more than 3 classes it must be cross-entropy loss'
#parameters
batch_size = args.batch_size
in_channels = 3
n_filters = args.n_filters
num_workers = args.num_workers
start_epoch = int(args.start_epoch)
epochs = args.epochs
hight = int(args.dimension[0])
width = int(args.dimension[1])
#%%
print("============================Initialize========================")

#================READ DATASET =================
with open(args.csv_train_path, newline='') as f:
    reader = csv.reader(f)
    reader = list(reader)
del reader[0]

reader = np.asarray(reader)
train_rgb = reader[:,0]
train_seg = reader[:,1]

with open(args.csv_val_path, newline='') as f:
    reader = csv.reader(f)
    reader = list(reader)
del reader[0]

reader = np.asarray(reader)
val_rgb = reader[:,0]
val_seg = reader[:,1]

del reader
#%%
wandb.init(name=args.exp_name)
wandb.config.update(args) 

#====================get loaders===============================
trainloader = loader(train_rgb, train_seg, batch_size, hight, width, args.loss, val=False, bbox_json = args.json_bbox_train)

valloader = loader(val_rgb, val_seg, batch_size, hight, width, args.loss, val=True, bbox_json = args.json_bbox_val)
len_trainloader = len(trainloader)
len_valloader = len(valloader)

#====================define & train model=============================
net = segNet(args.exp_name, in_channels, num_classes=args.num_classes, loss = args.loss, n_filters=n_filters, load = args.pretrained, lr=0.0001, batchnorm=True, coordconv=True, train = True, net = args.net, use_bbox = args.use_bbox)

print("Batch size: {}. Train loader lenght: {}. ".format(batch_size, len(trainloader)))
for epoch in tqdm(range(start_epoch, epochs)):
    loss = 0.0
    loss_bb = 0.0
    loss_seg = 0.0
    for data in trainloader:

        net.get_input(data)
    
        net.optimize_parameters()

        error_seg, error_bbox, error = net.get_loss()

        loss += error
        loss_bb += error_bbox
        loss_seg += error_seg
    
    #Save model
    net.save_model(epoch)
    loss = loss / len_trainloader
    loss_bb = loss_bb / len_trainloader
    loss_seg = loss_seg / len_trainloader

    
    #Get example image from train set and tranfer to W&B
    rgb, seg, pred_seg = net.get_images()

    rgb = wandb.Image(rgb[0], caption='rgb')
    wandb.log({"RGB": rgb}, step=epoch)

    if args.loss == 'iou':
        seg = wandb.Image(seg[0], caption='seg')
        wandb.log({"seg GT": seg}, step=epoch)

    pred_seg = wandb.Image(pred_seg[0], caption='pred_seg')
    wandb.log({"seg prediction": pred_seg}, step=epoch)
    
    #Run on the validation set
    val_loss, rgb, seg, pred_seg = net.validation(valloader, len_valloader)
    """
    rgb = wandb.Image(rgb[0], caption='rgb_val')
    wandb.log({"RGB_val": rgb}, step=epoch)

    #seg = wandb.Image(seg[0], caption='seg_val')
    #wandb.log({"seg GT_val": seg}, step=epoch)

    pred_seg = wandb.Image(pred_seg[0], caption='pred_seg_val')
    wandb.log({"seg prediction_val": pred_seg}, step=epoch)
    """
    #print losses and trasnfer to W&B
    wandb.log({"Train loss:": loss}, step=epoch)
    wandb.log({"Train loss_bb:": loss_bb}, step=epoch)
    wandb.log({"Train loss_segmentation:": loss_seg}, step=epoch)
    wandb.log({"Validation loss:": val_loss}, step=epoch)
    
    print(f"Epoch {epoch+1}/{epochs}.. ")
    print('Train loss {}, Val loss {}'.format(loss, val_loss))
