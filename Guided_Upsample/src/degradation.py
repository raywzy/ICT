import os.path
import io
import zipfile
from PIL import Image
import cv2
import torchvision.transforms as transforms
import numpy as np
import random
from io import BytesIO
from scipy import ndimage, misc
import argparse
import os

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)



def normalize_img(img):
  return img/127.5 - 1

def squared_euclidean_distance_np(a,b):
  b = b.T
  a2 = np.sum(np.square(a),axis=1)
  b2 = np.sum(np.square(b),axis=0)
  ab = np.matmul(a,b)
  d = a2[:,None] - 2*ab + b2[None,:]
  return d

def color_quantize_np(x, clusters):
    x = x.reshape(-1, 3)
    d = squared_euclidean_distance_np(x, clusters)
    return np.argmin(d,axis=1)

def prior_degradation(img,clusters,prior_size): ## Downsample into 32x32, using origin dictionary cluster to remap the pixel intensity

    img_np=np.array(img)

    LR_img_cv2=cv2.resize(img_np,(prior_size,prior_size), interpolation = cv2.INTER_AREA)
    x_norm=normalize_img(LR_img_cv2)
    token_id=color_quantize_np(x_norm,clusters)
    primers = token_id.reshape(-1,prior_size*prior_size)
    primers_img = [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [prior_size,prior_size, 3]).astype(np.uint8) for s in primers]

    degraded=Image.fromarray(primers_img[0])

    return degraded ## degraded by prior cluster 


def color_quantize_np_topK(x, clusters,K):
    x = x.reshape(-1, 3)
    d = squared_euclidean_distance_np(x, clusters)
    # print(np.argmin(d,axis=1))
    top_K=np.argpartition(d, K, axis=1)[:,:K] 

    h,w=top_K.shape
    select_index=np.random.randint(w,size=(h))
    return top_K[range(h),select_index]

def prior_degradation_2(img,clusters,prior_size,K=1): ## Downsample and random change

    LR_img_cv2=img.resize((prior_size,prior_size),resample=Image.BILINEAR)
    LR_img_cv2=np.array(LR_img_cv2)
    x_norm=normalize_img(LR_img_cv2)
    token_id=color_quantize_np_topK(x_norm,clusters,K)
    primers = token_id.reshape(-1,prior_size*prior_size)
    primers_img = [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [prior_size,prior_size, 3]).astype(np.uint8) for s in primers]

    degraded=Image.fromarray(primers_img[0])

    return degraded ## degraded by prior cluster 





if __name__=='__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_data',  type=str, default='/home/ziyuwan/workspace/data/ICCV_Inpainting_Test_Data/n02091635_40_60/gt_image', help='')
    parser.add_argument('--save_url', type=str, default='/home/ziyuwan/workspace/data/n02091635_40_60_degradation_1/condition_1', help='')

    opts = parser.parse_args()

    clusters = np.load('../kmeans_centers.npy')


    os.makedirs(opts.save_url,exist_ok=True)

    for x in os.listdir(opts.origin_data):
        img_url=os.path.join(opts.origin_data,x)
        img=Image.open(img_url).convert("RGB")
        y=prior_degradation(img,clusters,prior_size=32)
        y.save(os.path.join(opts.save_url,x))

