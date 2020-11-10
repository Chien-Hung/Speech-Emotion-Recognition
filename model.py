#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:05:03 2017

@author: hxj
"""

from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import numpy as np
from acrnn import acrnn
import pickle
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix as confusion
import os
import torch
import torch.optim as optim
import pdb


num_epoch = 5000
num_classes = 4
batch_size = 60
is_adam = True
learning_rate = 0.00001
dropout_keep_prob = 1
image_height = 300
image_width = 40
image_channel = 3
traindata_path = './IEMOCAP.pkl'
validdata_path = 'inputs/valid.pkl'
checkpoint = './checkpoint'
model_name = 'best_model.pth'
clip = 0

def load_data(in_dir):
    f = open(in_dir,'rb')
    train_data,train_label,test_data,test_label,valid_data,valid_label,Valid_label,Test_label,pernums_test,pernums_valid = pickle.load(f)
    #train_data,train_label,test_data,test_label,valid_data,valid_label = pickle.load(f)
    return train_data,train_label,test_data,test_label,valid_data,valid_label,Valid_label,Test_label,pernums_test,pernums_valid


def train():
    #####load data##########
    train_data, train_label, test_data, test_label, valid_data, valid_label, Valid_label, Test_label, pernums_test, pernums_valid = load_data('./IEMOCAP.pkl')

    train_label = train_label.reshape(-1)
    valid_label = valid_label.reshape(-1)
    Valid_label = Valid_label.reshape(-1)

    valid_size = valid_data.shape[0]
    dataset_size = train_data.shape[0]
    vnum = pernums_valid.shape[0]
    best_valid_uw = 0
    device = 'cuda'

    ##########tarin model###########

    def init_weights(m):
        if type(m) == torch.nn.Linear:
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(0.1)
        elif type(m) == torch.nn.Conv2d:
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(0.1)

    model = acrnn()
    model.apply(init_weights)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # print(train_data.shape)        # (1200, 300, 40, 3)  # (B, H, W, C)
    train_data = train_data.transpose((0, 3, 1, 2))
    test_data = test_data.transpose((0, 3 ,1 ,2))
    valid_data = valid_data.transpose((0, 3 ,1 ,2))
    # print(train_data.shape)        # (1200, 3, 300, 40)  # (B, C, H, W)
    
    num_epoch = 250
    train_iter = divmod(dataset_size, batch_size)[0]

    for epoch in range(num_epoch):
        # training
        model.train()
        shuffle_index = list(range(len(train_data)))
        np.random.shuffle(shuffle_index)
        
        for i in range(train_iter):
            start = (i*batch_size) % dataset_size
            end = min(start+batch_size, dataset_size)

            if i == (train_iter-1) and end < dataset_size:
                end = dataset_size
        
            inputs = torch.tensor(train_data[shuffle_index[start:end]]).to(device)
            targets = torch.tensor(train_label[shuffle_index[start:end]], dtype=torch.long).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
        
            loss = criterion(outputs, targets)
            loss.backward()
            if clip:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        
        if epoch % 1 == 0:
             # validation
             model.eval()
             valid_iter = divmod(valid_size, batch_size)[0]
             y_pred_valid = np.empty((valid_size, num_classes),dtype=np.float32)
             y_valid = np.empty((vnum, 4), dtype=np.float32)
             index = 0     
             cost_valid = 0
             
             if (valid_size < batch_size):

                 # inference
                 with torch.no_grad():
                     inputs = torch.tensor(valid_data[v_begin:v_end]).to(device)
                     targets = torch.tensor(Valid_label[v_begin:v_end], dtype=torch.long).to(device)
                     outputs = model(inputs)
                     y_pred_valid[v_begin:v_end,:] = outputs.cpu().detach().numpy()
                     loss = criterion(outputs, targets).cpu().detach().numpy()

                 cost_valid = cost_valid + np.sum(loss)
             
             for v in range(valid_iter):
                 v_begin, v_end = v*batch_size, (v+1)*batch_size

                 if v == (valid_iter-1) and v_end < valid_size:
                     v_end = valid_size

                 # inference
                 with torch.no_grad():
                     inputs = torch.tensor(valid_data[v_begin:v_end]).to(device)
                     targets = torch.tensor(Valid_label[v_begin:v_end], dtype=torch.long).to(device)
                     outputs = model(inputs)
                     y_pred_valid[v_begin:v_end,:] = outputs.cpu().detach().numpy()
                     loss = criterion(outputs, targets).cpu().detach().numpy()
                  
                 cost_valid = cost_valid + np.sum(loss)

             cost_valid = cost_valid/valid_size

             for s in range(vnum):
                 y_valid[s,:] = np.max(y_pred_valid[index:index+pernums_valid[s],:], 0)
                 index = index + pernums_valid[s]
                 
             # compute evaluated results
             valid_acc_uw = recall(valid_label, np.argmax(y_valid, 1), average='macro')
             valid_conf = confusion(valid_label, np.argmax(y_valid, 1))

             # save the best val result
             if valid_acc_uw > best_valid_uw:
                 best_valid_uw = valid_acc_uw
                 best_valid_conf = valid_conf

                 if not os.path.isdir(checkpoint):
                     os.mkdir(checkpoint)
                 torch.save(model.state_dict(), os.path.join(checkpoint, model_name))

             # print results
             print ("*****************************************************************")
             print ("Epoch: %05d" %(epoch+1))
             # print ("Training cost: %2.3g" %tcost)   
             # print ("Training accuracy: %3.4g" %tracc) 
             print ("Valid cost: %2.3g" %cost_valid)
             print ("Valid_UA: %3.4g" %valid_acc_uw)    
             print ("Best valid_UA: %3.4g" %best_valid_uw) 
             print ('Valid Confusion Matrix:["ang","sad","hap","neu"]')
             print (valid_conf)
             print ('Best Valid Confusion Matrix:["ang","sad","hap","neu"]')
             print (best_valid_conf)
             print ("*****************************************************************" )

                
if __name__=='__main__':
    train()
