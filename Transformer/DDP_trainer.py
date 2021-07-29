import math
import logging

# from tqdm import tqdm
import numpy as np

import os
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
import torch.distributed as dist

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = False
    warmup_tokens = 375e6
    final_tokens = 260e9
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, gpu, global_rank):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device=gpu

        self.model=model.cuda(gpu)
        self.global_rank=global_rank
        self.train_sampler=DistributedSampler(train_dataset, num_replicas=config.world_size, rank=global_rank)
        self.test_sampler=DistributedSampler(test_dataset, num_replicas=config.world_size, rank=global_rank)

    def save_checkpoint(self, epoch, optim, tokens, validation_loss,save_name):
         if self.global_rank==0: ## Only save in global rank 0
            # DataParallel wrappers keep raw model object in .module attribute
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            save_url=os.path.join(self.config.ckpt_path,save_name+'.pth')
            logger.info("saving %s", save_url)
            torch.save({'model': raw_model.state_dict(),
                        'epoch': epoch,
                        'optimizer':optim.state_dict(),
                        'tokens': tokens,
                        'best_validation_loss': validation_loss}, save_url)

    def load_checkpoint(self, resume_path):
        if os.path.exists(resume_path):
            #data = torch.load(resume_path, map_location = lambda storage, loc: set_device(storage)) 
            #data = torch.load(resume_path)
            data = torch.load(resume_path, map_location='cuda:{}'.format(self.device))
            self.model.load_state_dict(data['model'])
            print('Finished reloading the Epoch %d model'%(data['epoch']))
            return data
        else:
            if self.global_rank==0:
                print('Warnning: There is no trained model found. An initialized model will be used.')
        return None

    def train(self, loaded_ckpt):

        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        previous_epoch=-1

        if loaded_ckpt is not None:
            optimizer.load_state_dict(loaded_ckpt['optimizer'])
            self.tokens=loaded_ckpt['tokens']
            best_loss=loaded_ckpt['best_validation_loss']
            previous_epoch=loaded_ckpt['epoch']
            print('Finished reloading the Epoch %d optimizer'%(loaded_ckpt['epoch']))
        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

        
        #model = DDP(self.model,device_ids=[self.global_rank],output_device=self.global_rank,broadcast_buffers=True)
        model = DDP(self.model,device_ids=[self.device])
                    
        ## TODO: Use different seeds to initialize each worker. (This issue is caused by the bug of pytorch itself)
        train_loader = DataLoader(self.train_dataset, shuffle=False, pin_memory=True,
                            batch_size=config.batch_size // config.world_size, ## BS of each GPU
                            num_workers=config.num_workers,sampler=self.train_sampler)
        test_loader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True,
                            batch_size=config.batch_size // config.world_size, ## BS of each GPU
                            num_workers=config.num_workers,sampler=self.test_sampler)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            loader = train_loader if is_train else test_loader
            

            losses = []
            scaler = GradScaler()
            for it, (x, y) in enumerate(loader):

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # forward the model
                if self.config.AMP: ## use AMP
                    with autocast():
                        with torch.set_grad_enabled(is_train):
                            if self.config.BERT:
                                logits, loss = model(x, x, y)
                            else:
                                logits, loss = model(x, y)
                            loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                            losses.append(loss.item())
                else:
                    with torch.set_grad_enabled(is_train):
                        if self.config.BERT:
                            logits, loss = model(x, x, y)
                        else:
                            logits, loss = model(x, y)
                        loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()

                    if self.config.AMP:
                        scaler.scale(loss).backward()

                        ## AMP+Gradient Clip
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)

                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (x >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        # print(self.tokens)
                        # print(config.warmup_tokens)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    if it % self.config.print_freq == 0:
                        print(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        if loaded_ckpt is None:
            self.tokens = 0 # counter used for learning rate decay
            best_loss = float('inf')
                
        for epoch in range(config.max_epochs):

            if previous_epoch!=-1 and epoch<=previous_epoch:
                continue

            if epoch==previous_epoch+1:
                print("Resume from Epoch %d"%(epoch))

            self.train_sampler.set_epoch(epoch) ## Shuffle each epoch

            epoch_start=time.time()
            run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')
            
            print("Epoch: %d, test loss: %f, time for one epoch: %d seconds"%(epoch, test_loss, time.time() - epoch_start))
            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model and self.global_rank==0: ## Validation on the global_rank==0 process

                best_loss = test_loss
                print("current best epoch is %d"%(epoch))
                self.save_checkpoint(epoch, optimizer, self.tokens, best_loss,save_name='best')
            
            if not np.isnan(test_loss):
                self.save_checkpoint(epoch, optimizer, self.tokens, best_loss,save_name='latest')
            else:
                print('NaN happens, try to reload the previous normal checkpoint')
