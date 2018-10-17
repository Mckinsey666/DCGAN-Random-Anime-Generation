# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 14:47:07 2018

@author: USER
"""

import torch
import torch.nn
from torch.autograd import grad, Variable

import matplotlib.pyplot as plt
import numpy as np

def denorm(img):
    output = img / 2 + 0.5
    return output.clamp(0, 1)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

def save_model(model, optimizer, step, log, file_path):
    state = {'model' : model.state_dict(),
             'optim' : optimizer.state_dict(),
             'step' : step,
             'log' : log}
    torch.save(state, file_path)
    return

def load_model(model, optimizer, file_path):
    prev_state = torch.load(file_path)
    
    model.load_state_dict(prev_state['model'])
    optimizer.load_state_dict(prev_state['optim'])
    start_epoch = prev_state['step']
    log = prev_state['log']
    
    return model, optimizer, start_epoch, log
    

def show_process(steps, step_i, gen_log, discrim_log):
    print('Step {}/{}: G_loss [{:8f}], D_loss [{:8f}]'.format(
            step_i,
            steps,
            gen_log[-1], 
            discrim_log[-1]))
    return

def plot_loss(gen_log, discrim_log, file_path):
    epochs = list(range(len(gen_log)))
    plt.semilogy(epochs, gen_log)
    plt.semilogy(epochs, discrim_log)
    plt.legend(['Generator Loss', 'Discriminator Loss'])
    plt.title("Loss ({} epochs)".format(len(epochs)))
    plt.savefig(file_path)
    plt.close()
    return
    