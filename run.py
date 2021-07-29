import os
import argparse
import shutil
import sys
from subprocess import call

def run_cmd(command):
    try:
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image",type=str,default='./',help='The test input image path')
    parser.add_argument("--input_mask",type=str,default='./',help='The test input mask path')
    parser.add_argument("--sample_num",type=int,default=10,help='# of completion results')
    parser.add_argument("--save_place",type=str,default='./save',help='Please use the absolute path')
    parser.add_argument("--FFHQ",action='store_true',help='FFHQ pretrained model')
    parser.add_argument("--Places2_Nature",action='store_true',help='Places2_Nature pretrained model')
    parser.add_argument("--ImageNet",action='store_true',help='ImageNet pretrained model')
    parser.add_argument("--visualize_all",action='store_true',help='show the diverse results in one row')

    opts=parser.parse_args()

    ### Stage1: Reconstruction of Appearance Priors using Transformer

    prior_url=os.path.join(opts.save_place,"AP")
    if os.path.exists(prior_url):
        print("Please change the save path")
        sys.exit(1)
    os.chdir("./Transformer")

    if opts.visualize_all:
        suffix_opt = " --same_face"
        test_batch_size = str(opts.sample_num)
    else:
        suffix_opt = ""
        test_batch_size = str(1)

    if opts.ImageNet:
        stage_1_command = "CUDA_VISIBLE_DEVICES=0 python inference.py --ckpt_path ../ckpts_ICT/Transformer/ImageNet.pth \
                                --BERT --image_url " + opts.input_image + " \
                                --mask_url " + opts.input_mask + " \
                                --n_layer 35 --n_embd 1024 --n_head 8 --top_k 40 --GELU_2 \
                                --save_url " + prior_url + " \
                                --image_size 32 --n_samples " + str(opts.sample_num)
    elif opts.FFHQ:
        stage_1_command = "CUDA_VISIBLE_DEVICES=0 python inference.py --ckpt_path ../ckpts_ICT/Transformer/FFHQ.pth \
                                --BERT --image_url " + opts.input_image + " \
                                --mask_url " + opts.input_mask + " \
                                --n_layer 30 --n_embd 512 --n_head 8 --top_k 40 --GELU_2 \
                                --save_url " + prior_url + " \
                                --image_size 48 --n_samples " + str(opts.sample_num)
    elif opts.Places2_Nature:
        stage_1_command = "CUDA_VISIBLE_DEVICES=0 python inference.py --ckpt_path ../ckpts_ICT/Transformer/Places2_Nature.pth \
                                --BERT --image_url " + opts.input_image + " \
                                --mask_url " + opts.input_mask + " \
                                --n_layer 35 --n_embd 512 --n_head 8 --top_k 40 --GELU_2 \
                                --save_url " + prior_url + " \
                                --image_size 32 --n_samples " + str(opts.sample_num)
    else:
        print("ERROR: Please use right checkpoints.")
        sys.exit(1)

    
    run_cmd(stage_1_command)

    print("Finish the Stage 1 - Appearance Priors Reconstruction using Transformer")


    os.chdir("../Guided_Upsample")
    if opts.ImageNet:
        stage_2_command = "CUDA_VISIBLE_DEVICES=0,1 python test.py --input " + opts.input_image + " \
                                        --mask " + opts.input_mask + " \
                                        --prior " + prior_url + " \
                                        --output " + opts.save_place + " \
                                        --checkpoints ../ckpts_ICT/Upsample/ImageNet \
                                        --test_batch_size " + test_batch_size + " --model 2 --Generator 4 --condition_num " + str(opts.sample_num) +  suffix_opt
    elif opts.FFHQ:
        stage_2_command = "CUDA_VISIBLE_DEVICES=0 python test.py --input " + opts.input_image + " \
                                        --mask " + opts.input_mask + " \
                                        --prior " + prior_url + " \
                                        --output " + opts.save_place + " \
                                        --checkpoints ../ckpts_ICT/Upsample/FFHQ \
                                        --test_batch_size " + test_batch_size+ " --model 2 --Generator 4 --condition_num " + str(opts.sample_num) +  suffix_opt
    elif opts.Places2_Nature:
        stage_2_command = "CUDA_VISIBLE_DEVICES=0 python test.py --input " + opts.input_image + " \
                                        --mask " + opts.input_mask + " \
                                        --prior " + prior_url + " \
                                        --output " + opts.save_place + " \
                                        --checkpoints ../ckpts_ICT/Upsample/Places2_Nature \
                                        --test_batch_size " + test_batch_size + " --model 2 --Generator 4 --condition_num " + str(opts.sample_num) +  suffix_opt
    else:
        print("ERROR: Please use right checkpoints.")
        sys.exit(1)

    run_cmd(stage_2_command)

    print("Finish the Stage 2 - Guided Upsampling")
    print("Please check the results ...")

