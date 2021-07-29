import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
# from scipy.misc import imread
# from imageio import imread
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from .utils import create_mask
from .degradation import prior_degradation, prior_degradation_2
from random import randrange

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, edge_flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.edge_data = self.load_flist(edge_flist)
        self.mask_data = self.load_flist(mask_flist)


        if not training:  ## During testing, load the transformer prior
            all_data=[]
            all_edge_data=[]
            all_mask_data=[]
            for i, x in enumerate(self.data):
                for j in range(config.condition_num):
                    temp='condition_%d/%s'%(j+1,os.path.basename(x))
                    all_data.append(x)
                    all_mask_data.append(self.mask_data[i])
                    all_edge_data.append(os.path.join(edge_flist,temp))
            self.data=all_data
            self.edge_data=all_edge_data
            self.mask_data=all_mask_data


        self.input_size = config.INPUT_SIZE
        self.edge = config.EDGE
        self.mask = config.MASK
        self.prior_size = config.prior_size

        self.config=config


        self.clusters=np.load('./kmeans_centers.npy')

        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        if config.MODE == 2:
            self.mask = 6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        # print(item[0].shape)
        # print(item[1].shape)
        # print(item[2].shape)
        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        # load image
        #img = imread(self.data[index])
        img = Image.open(self.data[index]).convert("RGB")
        img = np.array(img)

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)


        # load mask
        mask = self.load_mask(img, index)

        # load edge
        prior = self.load_prior(img, index)

        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            prior = prior[:, ::-1, ...]

        if self.augment and np.random.binomial(1, 0.5) > 0:
            mask = mask[:, ::-1, ...]
        if self.augment and np.random.binomial(1, 0.5) > 0:
            mask = mask[::-1, :, ...]

        return self.to_tensor(img), self.to_tensor(prior), self.to_tensor(mask)

    def load_prior(self, img, index):


        # Training, prior_degradation
        if self.edge == 1:

            imgh, imgw = img.shape[0:2]
            x = Image.fromarray(img).convert("RGB")

            if self.config.use_degradation_2:
                prior_lr=prior_degradation_2(x,self.clusters,self.prior_size, self.config.prior_random_degree)
            else:
                prior_lr=prior_degradation(x,self.clusters,self.prior_size)
            prior_lr=np.array(prior_lr).astype('uint8')
            prior_lr=self.resize(prior_lr, imgh, imgw)

            return prior_lr

        # external, from transformer
        else:
            
            imgh, imgw = img.shape[0:2]
            # edge = imread(self.edge_data[index])

            edge = Image.open(self.edge_data[index]).convert("RGB")
            edge = np.array(edge)

            edge = self.resize(edge, imgh, imgw)

            return edge

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # external + random block + half
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        # random block
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # half
        if mask_type == 2:
            # randomly choose right or left
            return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            # mask = imread(self.mask_data[mask_index])

            mask = Image.open(self.mask_data[mask_index]).convert("RGB")
            mask = np.array(mask)

            # print(mask.shape)
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255       # threshold due to interpolation
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            # mask = imread(self.mask_data[index])
            mask = Image.open(self.mask_data[index]).convert("RGB")
            mask = np.array(mask)
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):


        imgh, imgw = img.shape[0:2]


        if self.training:  ## While training, random crop with short side
            img=Image.fromarray(img)
            side = np.minimum(imgh, imgw)
            y1=randrange(0,imgh-side+1)
            x1=randrange(0,imgw-side+1)
            img=img.crop((x1,y1,x1+side,y1+side))
            img=np.array(img.resize((height, width),resample=Image.BICUBIC))
            #img=np.array(img.resize((height, width)))
        else:
            if centerCrop and imgh != imgw:
                # center crop
                side = np.minimum(imgh, imgw)
                j = (imgh - side) // 2
                i = (imgw - side) // 2
                img = img[j:j + side, i:i + side, ...]
            img=np.array(Image.fromarray(img).resize((height, width),resample=Image.BICUBIC))
            #img=np.array(Image.fromarray(img).resize((height, width)))

        #img = scipy.misc.imresize(img, [height, width])

        return img

    def load_flist(self, flist):
        
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                # flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist = self.getfilelist(flist)
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

    def getfilelist(self, path):
        all_file=[]
        for dir,folder,file in os.walk(path):
            for i in file:
                t = "%s/%s"%(dir,i)
                if t.endswith('.png') or t.endswith('.jpg') or t.endswith('.JPG') or t.endswith('.PNG') or t.endswith('.JPEG'):
                    all_file.append(t)
        return all_file
