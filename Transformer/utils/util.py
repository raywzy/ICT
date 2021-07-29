import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import os
import sys
import cv2
from PIL import Image
import glob
import scipy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):

    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x


## TODO: support both unconditional generation and image completion [√]
def sample_new(model, context, length, num_sample=1, temperature=1.0, top_k=None):

    if context is None: ## unconditional generation
        counter=0
        output=torch.zeros(num_sample,length,dtype=torch.long).cuda()
    else:  ## completion
        seq_len=context.shape[1]
        counter=seq_len
        output=torch.zeros(num_sample,length,dtype=torch.long).cuda()
        output[:,:seq_len]=context
    

    pad = torch.zeros(num_sample, 1, dtype=torch.long).cuda()  # to pad prev output
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(length), leave=False):
            
            if i<counter:
                continue
    
            logits,_ = model(torch.cat((output[:,:counter], pad), dim=1))
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            pred = torch.multinomial(probs, num_samples=1)
            output[:,counter] = pred[:,0]

            counter+=1

    return output


## Currently, iterative sampling
def sample_mask(model, context, length, num_sample=1, temperature=1.0, top_k=None, mask=None, no_bar=False):

    # output = torch.zeros(num_sample,length,dtype=torch.long).cuda()
    output = context.cuda()
    mask = mask.cuda()

    model.eval()
    with torch.no_grad():

        if no_bar:
            looper=range(length)
        else:
            looper=tqdm(range(length), leave=False)
        for i in looper:

            if mask[0,i] == 0:
                continue

            logits,_ = model(output,masks=mask)
            logits = logits[:, i, :] / temperature
            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            pred = torch.multinomial(probs, num_samples=1)
            output[:,i] = pred[:,0]
            mask[:,i] = 0.

    return output

## Forward once, sample all, Ablation use
def sample_mask_all(model, context, length, num_sample=1, temperature=1.0, top_k=None, mask=None,no_bar=False):

    # output = torch.zeros(num_sample,length,dtype=torch.long).cuda()
    output = context.cuda()
    mask = mask.cuda()

    model.eval()
    with torch.no_grad():

        logits,_ = model(output,masks=mask)

        if no_bar:
            looper=range(length)
        else:
            looper=tqdm(range(length), leave=False)
        for i in looper:

            if mask[0,i] == 0:
                continue
            logits_i = logits[:, i, :] / temperature
            if top_k is not None:
                logits_i = top_k_logits(logits_i, top_k)
            probs = F.softmax(logits_i, dim=-1)
            pred = torch.multinomial(probs, num_samples=1)
            output[:,i] = pred[:,0]

    return output



## Forward once, sample all, Ablation use
def sample_mask_all_probability(model, context, length, num_sample=1, temperature=1.0, top_k=None, mask=None,no_bar=False):

    # output = torch.zeros(num_sample,length,dtype=torch.long).cuda()
    output = context.cuda()
    mask = mask.cuda()

    model.eval()
    with torch.no_grad():

        logits,_ = model(output,masks=mask)
        logits=logits[0]
        output=F.softmax(logits,dim=-1)
        # if no_bar:
        #     looper=range(length)
        # else:
        #     looper=tqdm(range(length), leave=False)
        # for i in looper:

        #     if mask[0,i] == 0:
        #         continue
        #     logits_i = logits[:, i, :] / temperature
        #     if top_k is not None:
        #         logits_i = top_k_logits(logits_i, top_k)
        #     probs = F.softmax(logits_i, dim=-1)
        #     pred = torch.multinomial(probs, num_samples=1)
        #     output[:,i] = pred[:,0]

    return output

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            os.makedirs(os.path.dirname(fpath),exist_ok=True)
            #mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'a')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def generate_stroke_mask(im_size, max_parts=15, maxVertex=25, maxLength=100, maxBrushWidth=24, maxAngle=360):
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    parts = random.randint(5, 13)
    # print(parts)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    mask = np.concatenate([mask, mask, mask], axis = 2)
    return mask


def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask