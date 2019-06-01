# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

'''
Contains the following classes:
   - ModelData - High level information encapsulation
   - ObjectDetector - Greedy algorithm to build cuboids from belief maps 
'''

import time
import json
import os, shutil
import sys
import traceback
from os import path
import threading
from threading import Thread
import matplotlib.pyplot as plt

import numpy as np
import cv2

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models

from scipy import ndimage
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage.filters import gaussian_filter

# Import the definition of the neural network model and cuboids
from cuboid_pnp_solver import *

#global transform for image input
transform = transforms.Compose([
    # transforms.Scale(IMAGE_SIZE),
    # transforms.CenterCrop((imagesize,imagesize)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


#================================ Models ================================
def draw_features(width,height,x,savename):
    tic=time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        # plt.tight_layout()
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = (img - pmin) / (pmax - pmin + 0.000001)
        plt.imshow(img, cmap='gray')
        # print("{}/{}".format(i,width*height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time()-tic))

def draw_features_heatmap(width,height,x,img,savename):
    tic=time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)

    # img = img[0, :, :, :]
    # img = np.uint8(255*img)
    # image = img.numpy()
    image = img.cpu().clone()
    image = img.squeeze(0)
    unloader = transforms.ToPILImage()
    image = unloader(image)
    image = np.asarray(image)

    for i in range(width*height):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        # plt.tight_layout()
        heatmap = x[0, i, :, :]
        heatmap = torch.mean(x,dim=1).squeeze()
        heatmap = np.maximum(heatmap,0)
        heatmap /= torch.max(heatmap)
        
        heatmap = (heatmap.min()-heatmap) / (heatmap.max()-heatmap.min()) #normalize + inverse
        heatmap = np.uint8(255*heatmap)
        heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap,(image.shape[1],image.shape[0]))
        
        superimposed_img = image
        superimposed_img = cv2.addWeighted(heatmap,1,image,0.3,0.0)

        plt.imshow(superimposed_img.squeeze())
        # print("{}/{}".format(i,width*height))
    # plt.show()
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time()-tic))


class DopeNetwork(nn.Module):
    def __init__(
            self,
            numBeliefMap=9,
            numAffinity=16,
            stop_at_stage=1  # number of stages to process (if less than total number of stages)
        ):
        super(DopeNetwork, self).__init__()

        self.stop_at_stage = stop_at_stage

        vgg_full = models.vgg19(pretrained=False).features
        # self.vgg = nn.Sequential()
        
        self.vgg0 = nn.Sequential()
        self.vgg1 = nn.Sequential()
        self.vgg2 = nn.Sequential()
        self.vgg3 = nn.Sequential()
        self.vgg4 = nn.Sequential()
        self.vgg5 = nn.Sequential()
        self.vgg6 = nn.Sequential()
        self.vgg7 = nn.Sequential()
        self.vgg8 = nn.Sequential()
        self.vgg9 = nn.Sequential()
        self.vgg10 = nn.Sequential()
        self.vgg11 = nn.Sequential()
        self.vgg12 = nn.Sequential()
        self.vgg13 = nn.Sequential()
        self.vgg14 = nn.Sequential()
        self.vgg15 = nn.Sequential()
        self.vgg16 = nn.Sequential()
        self.vgg17 = nn.Sequential()
        self.vgg18 = nn.Sequential()
        self.vgg19 = nn.Sequential()
        self.vgg20 = nn.Sequential()
        self.vgg21 = nn.Sequential()
        self.vgg22 = nn.Sequential()
        self.vgg23 = nn.Sequential()
        self.vgg24 = nn.Sequential()
        self.vgg25 = nn.Sequential()
        self.vgg26 = nn.Sequential()
        # for i_layer in range(24):
        #     self.vgg.add_module(str(i_layer), vgg_full[i_layer])
        self.vgg0.add_module(str(0), vgg_full[0])
        self.vgg1.add_module(str(1), vgg_full[1])
        self.vgg2.add_module(str(2), vgg_full[2])
        self.vgg3.add_module(str(3), vgg_full[3])
        self.vgg4.add_module(str(4), vgg_full[4])
        self.vgg5.add_module(str(5), vgg_full[5])
        self.vgg6.add_module(str(6), vgg_full[6])
        self.vgg7.add_module(str(7), vgg_full[7])
        self.vgg8.add_module(str(8), vgg_full[8])
        self.vgg9.add_module(str(9), vgg_full[9])
        self.vgg10.add_module(str(10), vgg_full[10])
        self.vgg11.add_module(str(11), vgg_full[11])
        self.vgg12.add_module(str(12), vgg_full[12])
        self.vgg13.add_module(str(13), vgg_full[13])
        self.vgg14.add_module(str(14), vgg_full[14])
        self.vgg15.add_module(str(15), vgg_full[15])
        self.vgg16.add_module(str(16), vgg_full[16])
        self.vgg17.add_module(str(17), vgg_full[17])
        self.vgg18.add_module(str(18), vgg_full[18])
        self.vgg19.add_module(str(19), vgg_full[19])
        self.vgg20.add_module(str(20), vgg_full[20])
        self.vgg21.add_module(str(21), vgg_full[21])
        self.vgg22.add_module(str(22), vgg_full[22])



        # Add some layers
        # i_layer = 23
        # self.vgg.add_module(str(i_layer), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1))
        # self.vgg.add_module(str(i_layer+1), nn.ReLU(inplace=True))
        # self.vgg.add_module(str(i_layer+2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        # self.vgg.add_module(str(i_layer+3), nn.ReLU(inplace=True))

        i_layer = 23
        self.vgg23.add_module(str(i_layer), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1))
        self.vgg24.add_module(str(i_layer+1), nn.ReLU(inplace=True))
        self.vgg25.add_module(str(i_layer+2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        self.vgg26.add_module(str(i_layer+3), nn.ReLU(inplace=True))

        
        # print('---Belief------------------------------------------------')
        # _2 are the belief map stages
        self.m1_2 = DopeNetwork.create_stage(128, numBeliefMap, True)
        self.m2_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m3_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m4_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m5_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m6_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)

        # print('---Affinity----------------------------------------------')
        # _1 are the affinity map stages
        self.m1_1 = DopeNetwork.create_stage(128, numAffinity, True)
        self.m2_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m3_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m4_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m5_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m6_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)


    def forward(self, x):
        '''Runs inference on the neural network'''
        savepath = "/root/catkin_ws/src/dope/test"
        
        # out1 = self.vgg(x)
        # draw_features(8,8,out1.cpu().detach().numpy(),"{}/out_vgg.png".format(savepath))


        out0 = self.vgg0(x)
        out1 = self.vgg1(out0)
        out2 = self.vgg2(out1)
        out3 = self.vgg3(out2)
        out4 = self.vgg4(out3)
        out5 = self.vgg5(out4)
        out6 = self.vgg6(out5)
        out7 = self.vgg7(out6)
        out8 = self.vgg8(out7)
        out9 = self.vgg9(out8)
        out10 = self.vgg10(out9)
        out11 = self.vgg11(out10)
        out12 = self.vgg12(out11)
        out13 = self.vgg13(out12)
        out14 = self.vgg14(out13)
        out15 = self.vgg15(out14)
        out16 = self.vgg16(out15)
        out17 = self.vgg17(out16)
        out18 = self.vgg18(out17)
        out19 = self.vgg19(out18)
        out20 = self.vgg20(out19)
        out21 = self.vgg21(out20)
        out22 = self.vgg22(out21)
        out23 = self.vgg23(out22) #add own layers
        out24 = self.vgg24(out23)
        
        out25 = self.vgg25(out24)
        out26 = self.vgg26(out25)

        # print("x 's size:" + str(x.size()))
        # print("out0 's size:" + str(out0.size()))
        # print("out1 's size:" + str(out1.size()))
        # print("out2 's size:" + str(out2.size()))
        # print("out3 's size:" + str(out3.size()))
        # print("out4 's size:" + str(out4.size()))
        # print("out5 's size:" + str(out5.size()))
        # print("out6 's size:" + str(out6.size()))
        # print("out7 's size:" + str(out7.size()))
        # print("out8 's size:" + str(out8.size()))
        # print("out9 's size:" + str(out9.size()))
        # print("out10 's size:" + str(out10.size()))
        # print("out11 's size:" + str(out11.size()))
        # print("out12 's size:" + str(out12.size()))
        # print("out13 's size:" + str(out13.size()))
        # print("out14 's size:" + str(out14.size()))
        # print("out15 's size:" + str(out15.size()))
        # print("out16 's size:" + str(out16.size()))
        # print("out17 's size:" + str(out17.size()))
        # # print("out18 's size:" + str(out18.size()))
        # # print("out19 's size:" + str(out19.size()))
        # # print("out20 's size:" + str(out20.size()))
        # # print("out21 's size:" + str(out21.size()))
        # # print("out22 's size:" + str(out22.size()))
        # # print("out23 's size:" + str(out23.size()))
        # # print("out24 's size:" + str(out24.size()))
        # print("out25 's size:" + str(out25.size()))
        # print("out26 's size:" + str(out26.size()))
        
        
        # print("out0\n")
        # draw_features_heatmap(5,5,out0.cpu().detach(),x.cpu().detach(),"{}/out0.png".format(savepath));print("out1\n")
        # draw_features_heatmap(5,5,out1.cpu().detach(),x.cpu().detach(),"{}/out1.png".format(savepath));print("out2\n")
        # draw_features_heatmap(5,5,out2.cpu().detach(),x.cpu().detach(),"{}/out2.png".format(savepath));print("out3\n")
        # draw_features_heatmap(5,5,out3.cpu().detach(),x.cpu().detach(),"{}/out3.png".format(savepath));print("out4\n")
        # draw_features_heatmap(5,5,out4.cpu().detach(),x.cpu().detach(),"{}/out4.png".format(savepath));print("out5\n")
        # draw_features_heatmap(5,5,out5.cpu().detach(),x.cpu().detach(),"{}/out5.png".format(savepath));print("out6\n")
        # draw_features_heatmap(5,5,out6.cpu().detach(),x.cpu().detach(),"{}/out6.png".format(savepath));print("out7\n")
        # draw_features_heatmap(5,5,out7.cpu().detach(),x.cpu().detach(),"{}/out7.png".format(savepath));print("out8\n")
        # draw_features_heatmap(5,5,out8.cpu().detach(),x.cpu().detach(),"{}/out8.png".format(savepath));print("out9\n")
        # draw_features_heatmap(5,5,out9.cpu().detach(),x.cpu().detach(),"{}/out9.png".format(savepath));print("out10\n")
        # draw_features_heatmap(5,5,out10.cpu().detach(),x.cpu().detach(),"{}/out10.png".format(savepath));print("out11\n")
        # draw_features_heatmap(5,5,out11.cpu().detach(),x.cpu().detach(),"{}/out11.png".format(savepath));print("out12\n")
        # draw_features_heatmap(5,5,out12.cpu().detach(),x.cpu().detach(),"{}/out12.png".format(savepath));print("out13\n")
        # draw_features_heatmap(5,5,out13.cpu().detach(),x.cpu().detach(),"{}/out13.png".format(savepath));print("out14\n")
        # draw_features_heatmap(5,5,out14.cpu().detach(),x.cpu().detach(),"{}/out14.png".format(savepath));print("out15\n")
        # draw_features_heatmap(5,5,out15.cpu().detach(),x.cpu().detach(),"{}/out15.png".format(savepath));print("out16\n")
        # draw_features_heatmap(5,5,out16.cpu().detach(),x.cpu().detach(),"{}/out16.png".format(savepath));print("out17\n")
        # draw_features_heatmap(5,5,out17.cpu().detach(),x.cpu().detach(),"{}/out17.png".format(savepath));print("out18\n")
        # draw_features_heatmap(5,5,out18.cpu().detach(),x.cpu().detach(),"{}/out18.png".format(savepath));print("out19\n")
        # draw_features_heatmap(5,5,out19.cpu().detach(),x.cpu().detach(),"{}/out19.png".format(savepath));print("out20\n")
        # draw_features_heatmap(5,5,out20.cpu().detach(),x.cpu().detach(),"{}/out20.png".format(savepath));print("out21\n")
        # draw_features_heatmap(5,5,out21.cpu().detach(),x.cpu().detach(),"{}/out21.png".format(savepath));print("out22\n")
        # draw_features_heatmap(5,5,out22.cpu().detach(),x.cpu().detach(),"{}/out22.png".format(savepath));print("out23\n")
        # draw_features_heatmap(5,5,out23.cpu().detach(),x.cpu().detach(),"{}/out23.png".format(savepath));print("out24\n")
        # draw_features_heatmap(5,5,out24.cpu().detach(),x.cpu().detach(),"{}/out24.png".format(savepath));print("out25\n")
        # draw_features_heatmap(5,5,out25.cpu().detach(),x.cpu().detach(),"{}/out25.png".format(savepath));print("out26\n")
        # draw_features_heatmap(5,5,out26.cpu().detach(),x.cpu().detach(),"{}/out26.png".format(savepath));print("END\n")

        out1_2 = self.m1_2(out26) #out1
        out1_1 = self.m1_1(out26)   #out1
        # draw_features_heatmap(1,8,out1_2.cpu().detach(),x.cpu().detach(),"{}/out1_2.png".format(savepath));print("out1_2\n")
        # draw_features_heatmap(1,8,out1_1.cpu().detach(),x.cpu().detach(),"{}/out1_1.png".format(savepath));print("out1_1\n")
        if self.stop_at_stage == 1:
            return [out1_2],\
                   [out1_1]

        out2 = torch.cat([out1_2, out1_1, out26], 1) #out1
        out2_2 = self.m2_2(out2)
        out2_1 = self.m2_1(out2)

        # draw_features_heatmap(1,8,out2_2.cpu().detach(),x.cpu().detach(),"{}/out2_2.png".format(savepath));print("out2_2\n")
        # draw_features_heatmap(1,8,out2_1.cpu().detach(),x.cpu().detach(),"{}/out2_1.png".format(savepath));print("out2_1\n")
        if self.stop_at_stage == 2:
            return [out1_2, out2_2],\
                   [out1_1, out2_1]

        out3 = torch.cat([out2_2, out2_1, out26], 1) #out1
        out3_2 = self.m3_2(out3)
        out3_1 = self.m3_1(out3)

        # draw_features_heatmap(1,8,out3_2.cpu().detach(),x.cpu().detach(),"{}/out3_2.png".format(savepath));print("out3_2\n")
        # draw_features_heatmap(1,8,out3_1.cpu().detach(),x.cpu().detach(),"{}/out3_1.png".format(savepath));print("out3_1\n")
        if self.stop_at_stage == 3:
            return [out1_2, out2_2, out3_2],\
                   [out1_1, out2_1, out3_1]

        out4 = torch.cat([out3_2, out3_1, out26], 1) #out1
        out4_2 = self.m4_2(out4)
        out4_1 = self.m4_1(out4)

        if self.stop_at_stage == 4:
            return [out1_2, out2_2, out3_2, out4_2],\
                   [out1_1, out2_1, out3_1, out4_1]

        out5 = torch.cat([out4_2, out4_1, out26], 1) #out1
        out5_2 = self.m5_2(out5)
        out5_1 = self.m5_1(out5)

        if self.stop_at_stage == 5:
            return [out1_2, out2_2, out3_2, out4_2, out5_2],\
                   [out1_1, out2_1, out3_1, out4_1, out5_1]

        out6 = torch.cat([out5_2, out5_1, out26], 1) #out1
        out6_2 = self.m6_2(out6)
        out6_1 = self.m6_1(out6)

        return [out1_2, out2_2, out3_2, out4_2, out5_2, out6_2],\
               [out1_1, out2_1, out3_1, out4_1, out5_1, out6_1],\
                        
    @staticmethod
    def create_stage(in_channels, out_channels, first=False):
        '''Create the neural network layers for a single stage.'''

        model = nn.Sequential()
        mid_channels = 128
        if first:
            padding = 1
            kernel = 3
            count = 6
            final_channels = 512
        else:
            padding = 3
            kernel = 7
            count = 10
            final_channels = mid_channels

        # First convolution
        model.add_module("0",
                         nn.Conv2d(
                             in_channels,
                             mid_channels,
                             kernel_size=kernel,
                             stride=1,
                             padding=padding)
                        )

        # Middle convolutions
        i = 1
        while i < count - 1:
            model.add_module(str(i), nn.ReLU(inplace=True))
            i += 1
            model.add_module(str(i),
                             nn.Conv2d(
                                 mid_channels,
                                 mid_channels,
                                 kernel_size=kernel,
                                 stride=1,
                                 padding=padding))
            i += 1

        # Penultimate convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i), nn.Conv2d(mid_channels, final_channels, kernel_size=1, stride=1))
        i += 1

        # Last convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i), nn.Conv2d(final_channels, out_channels, kernel_size=1, stride=1))
        i += 1

        return model



class ModelData(object):
    '''This class contains methods for loading the neural network'''

    def __init__(self, name="", net_path="", gpu_id=0):
        self.name = name
        self.net_path = net_path  # Path to trained network model
        self.net = None  # Trained network
        self.gpu_id = gpu_id

    def get_net(self):
        '''Returns network'''
        if not self.net:
            self.load_net_model()
        return self.net

    def load_net_model(self):
        '''Loads network model from disk'''
        if not self.net and path.exists(self.net_path):
            self.net = self.load_net_model_path(self.net_path)
        if not path.exists(self.net_path):
            print("ERROR:  Unable to find model weights: '{}'".format(
                self.net_path))
            exit(0)

    def load_net_model_path(self, path):
        '''Loads network model from disk with given path'''
        model_loading_start_time = time.time()
        print("Loading DOPE model '{}'...".format(path))
        net = DopeNetwork()
        dict_new = net.state_dict().copy()
        dict_trained = torch.load(path)
 
        new_list = list(net.state_dict().keys())
        trained_list = list(dict_trained.keys())

        dict_new.clear()        
        for i in range(len(new_list)):
            # print("new_list",i,new_list[i])
            dict_new[ "module."+str(new_list[i]) ] = dict_trained[ trained_list[i] ]
        # for i in range(len(trained_list)):
            # print("trained_list",i,trained_list[i])

        net = torch.nn.DataParallel(net, [0]).cuda()
        # net.load_state_dict(torch.load(path))
        net.load_state_dict(dict_new)

        net.eval()
        print('    Model loaded in {} seconds.'.format(
            time.time() - model_loading_start_time))
        return net

    def __str__(self):
        '''Converts to string'''
        return "{}: {}".format(self.name, self.net_path)


#================================ ObjectDetector ================================
class ObjectDetector(object):
    '''This class contains methods for object detection'''

    @staticmethod
    def detect_object_in_image(net_model, pnp_solver, in_img, config):
        '''Detect objects in a image using a specific trained network model'''

        if in_img is None:
            return []

        # Run network inference
        image_tensor = transform(in_img)
        image_torch = Variable(image_tensor).cuda().unsqueeze(0)
        out, seg = net_model(image_torch)

        
        # img = ObjectDetector.draw_feature(1,1,feature_maps[0].detach().numpy())
        # img  = image_torch.cpu.detach().numpy()
        # plt.imshow(img)
        # plt.show()

        # out_img0 = ObjectDetector.tensor_to_PIL(seg[0])
        # out_img1 = ObjectDetector.tensor_to_PIL(seg[1])
        # out_img2 = ObjectDetector.tensor_to_PIL(seg[2])
        # out_img3 = ObjectDetector.tensor_to_PIL(seg[3])
        # out_img4 = ObjectDetector.tensor_to_PIL(seg[4])
        # out_img5 = ObjectDetector.tensor_to_PIL(seg[5])
        # plt.subplot(3,2,1)
        # plt.imshow(out_img0)
        # plt.subplot(3,2,2)
        # plt.imshow(out_img1)
        # plt.subplot(3,2,3)
        # plt.imshow(out_img2)
        # plt.subplot(3,2,4)
        # plt.imshow(out_img3)
        # plt.subplot(3,2,5)
        # plt.imshow(out_img4)
        # plt.subplot(3,2,6)
        # plt.imshow(out_img5)
        # plt.show()
        vertex2 = out[-1][0]
        aff = seg[-1][0]

        # Find objects from network output
        detected_objects,all_peaks = ObjectDetector.find_object_poses(vertex2, aff, pnp_solver, config)

        return detected_objects,all_peaks
    
    @staticmethod
    def tensor_to_PIL(tensor):
        # unloader = transforms.ToPILImage()
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        # image = unloader(image)
        # image = transforms.ToPILImage()(image).convert('RGB')
        return image

    @staticmethod
    def find_object_poses(vertex2, aff, pnp_solver, config):
        '''Detect objects given network output'''

        # Detect objects from belief maps and affinities
        objects, all_peaks = ObjectDetector.find_objects(vertex2, aff, config)
        detected_objects = []
        obj_name = pnp_solver.object_name

        for obj in objects:
            # Run PNP
            points = obj[1] + [(obj[0][0]*8, obj[0][1]*8)]
            cuboid2d = np.copy(points)
            location, quaternion, projected_points = pnp_solver.solve_pnp(points)

            # Save results
            detected_objects.append({
                'name': obj_name,
                'location': location,
                'quaternion': quaternion,
                'cuboid2d': cuboid2d,
                'projected_points': projected_points,
            })

        return detected_objects,all_peaks

    @staticmethod
    def find_objects(vertex2, aff, config, numvertex=8):
        '''Detects objects given network belief maps and affinities, using heuristic method'''

        all_peaks = []
        peak_counter = 0
        for j in range(vertex2.size()[0]):
            belief = vertex2[j].clone()
            map_ori = belief.cpu().data.numpy()
            
            map = gaussian_filter(belief.cpu().data.numpy(), sigma=config.sigma)
            p = 1
            map_left = np.zeros(map.shape)
            map_left[p:,:] = map[:-p,:]
            map_right = np.zeros(map.shape)
            map_right[:-p,:] = map[p:,:]
            map_up = np.zeros(map.shape)
            map_up[:,p:] = map[:,:-p]
            map_down = np.zeros(map.shape)
            map_down[:,:-p] = map[:,p:]

            peaks_binary = np.logical_and.reduce(
                                (
                                    map >= map_left, 
                                    map >= map_right, 
                                    map >= map_up, 
                                    map >= map_down, 
                                    map > config.thresh_map)
                                )
            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) 
            
            # Computing the weigthed average for localizing the peaks
            peaks = list(peaks)
            win = 5
            ran = win // 2
            peaks_avg = []
            for p_value in range(len(peaks)):
                p = peaks[p_value]
                weights = np.zeros((win,win))
                i_values = np.zeros((win,win))
                j_values = np.zeros((win,win))
                for i in range(-ran,ran+1):
                    for j in range(-ran,ran+1):
                        if p[1]+i < 0 \
                                or p[1]+i >= map_ori.shape[0] \
                                or p[0]+j < 0 \
                                or p[0]+j >= map_ori.shape[1]:
                            continue 

                        i_values[j+ran, i+ran] = p[1] + i
                        j_values[j+ran, i+ran] = p[0] + j

                        weights[j+ran, i+ran] = (map_ori[p[1]+i, p[0]+j])

                # if the weights are all zeros
                # then add the none continuous points
                OFFSET_DUE_TO_UPSAMPLING = 0.4395
                try:
                    peaks_avg.append(
                        (np.average(j_values, weights=weights) + OFFSET_DUE_TO_UPSAMPLING, \
                         np.average(i_values, weights=weights) + OFFSET_DUE_TO_UPSAMPLING))
                except:
                    peaks_avg.append((p[0] + OFFSET_DUE_TO_UPSAMPLING, p[1] + OFFSET_DUE_TO_UPSAMPLING))
            # Note: Python3 doesn't support len for zip object
            peaks_len = min(len(np.nonzero(peaks_binary)[1]), len(np.nonzero(peaks_binary)[0]))

            peaks_with_score = [peaks_avg[x_] + (map_ori[peaks[x_][1],peaks[x_][0]],) for x_ in range(len(peaks))]

            id = range(peak_counter, peak_counter + peaks_len)

            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += peaks_len

        objects = []

        # Check object centroid and build the objects if the centroid is found
        for nb_object in range(len(all_peaks[-1])):
            if all_peaks[-1][nb_object][2] > config.thresh_points:
                objects.append([
                    [all_peaks[-1][nb_object][:2][0],all_peaks[-1][nb_object][:2][1]],
                    [None for i in range(numvertex)],
                    [None for i in range(numvertex)],
                    all_peaks[-1][nb_object][2]
                ])

        # Working with an output that only has belief maps
        if aff is None:
            if len (objects) > 0 and len(all_peaks)>0 and len(all_peaks[0])>0:
                for i_points in range(8):
                    if  len(all_peaks[i_points])>0 and all_peaks[i_points][0][2] > config.threshold:
                        objects[0][1][i_points] = (all_peaks[i_points][0][0], all_peaks[i_points][0][1])
        else:
            # For all points found
            for i_lists in range(len(all_peaks[:-1])):
                lists = all_peaks[i_lists]

                for candidate in lists:
                    if candidate[2] < config.thresh_points:
                        continue

                    i_best = -1
                    best_dist = 10000 
                    best_angle = 100
                    for i_obj in range(len(objects)):
                        center = [objects[i_obj][0][0], objects[i_obj][0][1]]

                        # integer is used to look into the affinity map, 
                        # but the float version is used to run 
                        point_int = [int(candidate[0]), int(candidate[1])]
                        point = [candidate[0], candidate[1]]

                        # look at the distance to the vector field.
                        v_aff = np.array([
                                        aff[i_lists*2, 
                                        point_int[1],
                                        point_int[0]].data.item(),
                                        aff[i_lists*2+1, 
                                            point_int[1], 
                                            point_int[0]].data.item()]) * 10

                        # normalize the vector
                        xvec = v_aff[0]
                        yvec = v_aff[1]

                        norms = np.sqrt(xvec * xvec + yvec * yvec)

                        xvec/=norms
                        yvec/=norms
                            
                        v_aff = np.concatenate([[xvec],[yvec]])

                        v_center = np.array(center) - np.array(point)
                        xvec = v_center[0]
                        yvec = v_center[1]

                        norms = np.sqrt(xvec * xvec + yvec * yvec)
                            
                        xvec /= norms
                        yvec /= norms

                        v_center = np.concatenate([[xvec],[yvec]])
                        
                        # vector affinity
                        dist_angle = np.linalg.norm(v_center - v_aff)

                        # distance between vertexes
                        dist_point = np.linalg.norm(np.array(point) - np.array(center))
                        
                        if dist_angle < config.thresh_angle \
                                and best_dist > 1000 \
                                or dist_angle < config.thresh_angle \
                                and best_dist > dist_point:
                            i_best = i_obj
                            best_angle = dist_angle
                            best_dist = dist_point

                    if i_best is -1:
                        continue
                    
                    if objects[i_best][1][i_lists] is None \
                            or best_angle < config.thresh_angle \
                            and best_dist < objects[i_best][2][i_lists][1]:
                        objects[i_best][1][i_lists] = ((candidate[0])*8, (candidate[1])*8)
                        objects[i_best][2][i_lists] = (best_angle, best_dist)

        return objects, all_peaks
