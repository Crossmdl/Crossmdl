# pad your sequences

from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
import numpy as np
from PIL import Image
import os
from collections import defaultdict
import json
import joblib
from torch.utils.data import Dataset,DataLoader,random_split
from itertools import repeat
import pandas as pd
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA
from argparse import Namespace
from numpy import genfromtxt
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pytorch_lightning as pl
import wandb
import logging
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import clip

#utils

data_dir = '/common/users/vk405/Youcook/'
annotns_file ='/common/home/vk405/Projects/Crossmdl/Data/YouCookII/annotations/youcookii_annotations_trainval.json'
vid_data = [dir for dir in os.listdir(data_dir) if 'joblib' not in dir]

def get_vid_ids(split='training',\
    annotns_file='/common/home/vk405/Projects/Crossmdl/Data/YouCookII/annotations/youcookii_annotations_trainval.json'):
    # Returns vid_ids corresponding to the split: 'training'/'validation'
    
    vid_lis = []
    with open(annotns_file) as json_file:
        annotns = json.load(json_file)['database']
        for key in annotns:
            if annotns[key]['subset'] == split:
                vid_lis.append(key)
    return vid_lis


def get_split_files(split='training',\
    annotns_file='/common/home/vk405/Projects/Crossmdl/Data/YouCookII/annotations/youcookii_annotations_trainval.json',\
        data_dir = '/common/users/vk405/Youcook/'):
    total_ids = get_vid_ids(split,annotns_file)
    downloaded_ids = set([dir for dir in os.listdir(data_dir) if 'joblib' not in dir])
    vid_locs = []
    sents = {}
    segs = {}
    incomplete = []
    for id in total_ids:
        if id in downloaded_ids:
            vid_loc = data_dir+id + '/'
            if len(os.listdir(vid_loc))>=495:
                vid_locs.append(vid_loc)
                seg = joblib.load(data_dir+f'{id}global_segs.joblib')
                sent = joblib.load(data_dir+f'{id}global_sents.joblib')
                try:
                    sents[id] = sent[id]
                    segs[id] = seg[id]
                except:
                    print(f"{id} is no corresponding global sent/seg")
            else:
                #print(f"{id} has only imgs {len(os.listdir(vid_loc))}")
                incomplete.append(id)
    return vid_locs,segs,sents,incomplete 


clipfeat_loc ='/common/users/vk405/CLIP_FEAT/'

def extract_clip(split='training',
annotns_file='/common/home/vk405/Projects/Crossmdl/Data/YouCookII/annotations/youcookii_annotations_trainval.json',\
        data_dir = '/common/users/vk405/Youcook/',model_name = 'ViT-B/32',\
            store_dir = '/common/users/vk405/CLIP_FEAT/'):
    vid_locs,_,sents,_ = get_split_files(split,annotns_file,data_dir)
    useful_vids = []
    for vidloc in vid_locs:
        if len(os.listdir(vidloc)) != 500:
            files = sorted(os.listdir(vidloc),key=lambda x:int(x.split('_')[0]))
            if 'n.' in files[-1]:
                #if last image is not a segment, then we can safely add frames
                fileloc = vidloc + files[-1]
                filecnt = int(files[-1].split('_')[0])
                img = Image.open(fileloc)
                to_fill = 500-len(os.listdir(vidloc))
                for i in range(to_fill):
                    img.save(vidloc + f'{filecnt+i+1}_n.png')
                    print(f"saved at :{vidloc + f'{filecnt+i+1}_n.png'}")
                useful_vids.append(vidloc)
        else:
            useful_vids.append(vidloc)
    
    model,preprocess = clip.load(model_name)
    model.eval().cuda()
    error_cnt = {}
    for vidloc in tqdm(useful_vids):
        vid_id = vidloc.split('/')[-2]
        save_loc_vid = store_dir+split+'/'+f'vid_{vid_id}.joblib'
        save_loc_text = store_dir+split+'/'+f'txt_{vid_id}.joblib'
        if not os.path.exists(save_loc_vid):
            text_tokens = clip.tokenize(sents[vid_id]).cuda()

            # with torch.no_grad():
            #     text_features = model.encode_text(text_tokens).float()
            #     joblib.dump(text_features.detach().cpu().numpy(),save_loc_text)
            files = sorted(os.listdir(vidloc),key=lambda x:int(x.split('_')[0]))
            if len(files) == 500:
                imgs  = []
                cnt = 0
                for file in files:
                    try:
                        im = Image.open(vidloc+file)
                        imgs.append(preprocess(im))
                    except:
                        # hoping here it wont be the first one
                        cnt += 1
                        imgs.append(imgs[-1])
                        #import pdb;pdb.set_trace()
                error_cnt[vid_id]  = cnt
                    
                image_input = torch.tensor(np.stack(imgs)).cuda()
                #import pdb;pdb.set_trace()
                im_emb = []
                with torch.no_grad():
                    # else can throw memory error
                    text_features = model.encode_text(text_tokens).float()
                    joblib.dump(text_features.detach().cpu().numpy(),save_loc_text)
                    out1 = model.encode_image(image_input[:250]).float()
                    out2 = model.encode_image(image_input[250:]).float()
                    im_emb = torch.concat([out1,out2],dim=0)
                    joblib.dump(im_emb.detach().cpu().numpy(),save_loc_vid)
                    print(error_cnt)
    return error_cnt
            

if __name__ == '__main__':
    trn_error_cnt = extract_clip()
    joblib.dump(trn_error_cnt,'trn_error_cnt.joblib')

    val_error_cnt = extract_clip('validation')
    joblib.dump(trn_error_cnt,'val_error_cnt.joblib')
    

    
            
                
    



    
    