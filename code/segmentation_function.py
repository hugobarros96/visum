from architectures.res_unet import resUNet
from metrics_losses.losses import iou_pytorch, bbox_loss
import torch
from torchvision import transforms
import numpy as np
from architectures.dispnet import DispNetS
import os
import matplotlib.image as mpimg 

class segNet():
    def __init__(self, name_exp, in_channels, num_classes = 3, loss = 'iou', n_filters=16, load = [], lr=0.0001, batchnorm=True, coordconv=True, train = True, net = 'resunet'):
        '''
        Define the network for seg prediction:
        parameter:
                --> in channels (channels in input image)
                --> n_filters (convolutional kernel size in initial conv layer, default=16)
                --> out_channls (number of output channels, default=1 - for seg map)
                --> load (list with any model to load)
                --> lr (float corresponding to learning rate)
                --> batchnorm, coordconv (boolean to wheher or not use bacth norma and coorconv, respectively)
                --> train, wheter trin or use in eval model
                --> net, resunet or dispnet model
                --> loss, iou or cross_entropy

        '''
        self.name_exp = name_exp
        #LOSSES
        if loss =='cross_entropy':
            self.loss_cc = torch.nn.CrossEntropyLoss()
            sigmoid = False
            out_channels = num_classes
        else:
            self.loss_cc = False
            sigmoid = True
            out_channels = 3
        #define arquitecture
        if load:
            self.seg_net = torch.load(load)
        elif net == 'resunet':
            self.seg_net = resUNet(in_channels, n_filters, out_channels, batchnorm, coordconv, sigmoid=sigmoid)
        elif net == 'dispnet':
            self.seg_net = DispNetS(out_channels = out_channels, sigmoid = sigmoid)

        self.cuda_ = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.cuda_ else "cpu")
        
        if self.cuda_:
            self.seg_net.to(self.device)
        
        else:
            print('No GPU available!')
            exit()

        #optimizers:
        self.optimizer_seg = torch.optim.Adam(self.seg_net.parameters(), lr=lr, betas=(0.9, 0.99))
        

    def get_input(self, images, targets_):
        '''
        Use the dataloader to get the input images.
        Parameters:
                -->data (batch from dataloader)
        '''
        self.rgb = images
        self.seg = targets_["masks"]
        self.b_size = self.seg.size(0)
     

    def forward(self):
        '''
        Foward method. Passes the input images through the netwaork producing an ouput.
        '''
        self.pred_seg = self.seg_net(self.rgb)

        print(self.pred_seg, np.size(self.pred_seg))
        

    def backward_loss(self):
        '''
        Backward method. Backpropagation of the loss. 
        '''
        
        loss_s = 0
        loss_bb = 0
        count_s = 0 
        count_bb = 0

        if not self.loss_cc: #=================IOU=================================
                for i in range(self.b_size):
                    count_s+=1
                    loss_s += iou_pytorch(self.pred_seg[i], self.seg[i])
                    
                if loss_s > 0:
                    self.error_seg_real = loss_s / count_s
                else:
                    self.error_seg_real = 0

                self.error_bbox = 0
                self.error_real = self.error_seg_real 
        
        
        
        else: #=================CROSS ENTROPY=================================
            self.error_seg_real = 0
            self.error_bbox=0

            if self.use_bbox:
                self.error_real = self.loss_cc(self.pred_seg, self.seg[:,0,:,:])
            else:
                for i in range(self.b_size):
                    if self.bool[i]:
                        count_s+=1
                        print(np.shape(torch.unsqueeze(self.pred_seg[i], 0)), '==========')
                        print(np.shape(self.seg[i]))
                        print(np.unique(self.seg[i].cpu()))
                        loss_s += self.loss_cc(torch.unsqueeze(self.pred_seg[i], 0), self.seg[i])
                    
                if loss_s > 0:
                    self.error_seg_real = loss_s / count_s
                else:
                    self.error_seg_real = 0

                self.error_real = self.error_seg_real + self.error_bbox
                
        if self.error_real != 0:
            self.error_real.backward()
        
    def optimize_parameters(self):
        '''
        Optimize parameters/weights. 
        '''
        self.forward()

        self.optimizer_seg.zero_grad()
        self.backward_loss()
        self.optimizer_seg.step()

    def save_model(self, epoch):
        if os.path.isdir('models/'+self.name_exp):
            torch.save(self.seg_net, 'models/'+self.name_exp+'/seg'+str(epoch)+'.pth')
        else:
            try:
                if not os.path.isdir('models'):
                    os.mkdir('models/')
                os.mkdir('models/'+self.name_exp)
            except OSError:
                print ("Creation of the directory  failed" )
            else:
                print ("Successfully created the directory" )
  
    def get_loss(self):
        """
        returns loss
        """
        return self.error_seg_real, self.error_bbox, self.error_real

    def get_images(self):
        """
        returns images
        """
        return self.rgb, self.seg, self.pred_seg
    
    def validation(self, valloader, len_valloader):
        '''
        Runs the model in the validation set with no gradient.
        '''
        val_loss = 0
        self.seg_net.eval()
        with torch.no_grad():
            for data in valloader:
                self.get_input(data)
                self.forward()
                if self.loss_cc:
                    batch_loss = self.loss_cc(self.pred_seg, self.seg[:,0,:,:])
                else:
                    batch_loss = 0
                    count_s = 0
                    for i in range(self.b_size):
                        count_s+=1
                        batch_loss += iou_pytorch(self.pred_seg[i], self.seg[i])

                    batch_loss = batch_loss / count_s

                val_loss += batch_loss.item()

                rgb, seg, pred_seg= self.get_images()

        self.seg_net.train()
        return val_loss/len_valloader, rgb, seg, pred_seg