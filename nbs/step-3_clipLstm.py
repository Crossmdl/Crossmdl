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
from numpy import linalg as LA
from argparse import Namespace
from numpy import genfromtxt
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
import wandb
import logging
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import clip


import wandb
import logging
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os

logger = logging.getLogger(__name__)
wandb_logger = lambda dir, version: WandbLogger(
    name="wandb", save_dir=dir, version=version
)
csvlogger = lambda dir, version: CSVLogger(dir, name="csvlogs", version=version)
tblogger = lambda dir, version: TensorBoardLogger(dir, name="tblogs", version=version)

def get_loggers(dir,version,lis=["csv"]):
    lgrs = []
    if "wandb" in lis:
        lgrs.append(wandb_logger(dir, version))
    if "csv" in lis:
        lgrs.append(csvlogger(dir, version))
    if "tb" in lis:
        lgrs.append(tblogger(dir, version))
    return lgrs







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

import pathlib

FEAT_DIR = pathlib.Path('/common/users/vk405/CLIP_FEAT')
RAWFRAME_DIR = pathlib.Path('/common/users/vk405/Youcook/')

class Dset(Dataset):
    def __init__(self,data_dir,feat_dir,split):
        self.data_dir = data_dir
        self.feat_dir = feat_dir
        self.split = split
        self.vid_ids,self.sents = self.get_ids()
        self.labels = self.getlabels()
        self.sanitycheck()
        self.data = self.getdata()
        


    def sanitycheck(self):
        mis = []
        #import pdb;pdb.set_trace()
        for key in self.labels.keys():
            txt_loc = self.feat_dir/self.split/f'txt_{key}.joblib'
            txt = joblib.load(txt_loc)
            if len(self.labels[key]) == len(self.sents[key]) == len(txt):
                pass
            else:
                print(key)
                mis.append(key)
        print(f"segs are not matching:{mis}")
        for key in mis:
            self.vid_ids.remove(key)
        self.sents = None

        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.load(self.data[idx])

    def getdata(self):
        data = []
        for id in self.vid_ids:
            segs = self.labels[id]
            #import pdb;pdb.set_trace()
            for i in range(len(segs)):
                data.append((id,i))
        return data

    def load(self,data):
        vid_id,ind = data
        vid_frames_loc = self.feat_dir/self.split/f'vid_{vid_id}.joblib'
        txt_loc = self.feat_dir/self.split/f'txt_{vid_id}.joblib'
        st,end = self.labels[vid_id][ind]
        vid = joblib.load(vid_frames_loc)
        try:
            txt = joblib.load(txt_loc)[ind]
        except:
            import pdb;pdb.set_trace()
        #normalize data
        #import pdb;pdb.set_trace()
        vid = vid/(LA.norm(vid,axis=-1)).reshape(500,1)
        txt = (txt/LA.norm(txt))
        #out = np.squeeze(vid@txt.reshape(512,1))
        #regression outputs
        return vid,txt,st/499,end/499
         

    def getlabels(self):
        label_dict = {}
        for vidid in self.vid_ids:
            vidloc = self.data_dir/vidid
            segs = self.extract_seg(vidloc)
            label_dict[vidid] = segs
        return label_dict
    
    def extract_seg(self,vid_loc):
        imgs = sorted(os.listdir(vid_loc),key=lambda x: int(x.split('_')[0]))
        segs = defaultdict(list)
        for img in imgs:
            ind,rem = int(img.split('_')[0]),img.split('_')[-1]
            
            if 'n.' not in rem:
                #print(ind,rem)
                seg_id = int(rem.split('.')[0])
                segs[seg_id].append(ind)
                #print(seg_id,ind)
        final_segs = []
        #import pdb;pdb.set_trace()
        segids = sorted(segs.keys())
        for segid in segids:
            final_segs.append((min(segs[segid]),max(segs[segid])))
        return final_segs
        
    def get_ids(self):
        annotns_file='/common/home/vk405/Projects/Crossmdl/Data/YouCookII/annotations/youcookii_annotations_trainval.json'
        data_dir = '/common/users/vk405/Youcook/'
        vid_locs,_,sents,_ = get_split_files('training',annotns_file,data_dir)
        ids = [ele.split('/')[-2] for ele in vid_locs]
        files = set(os.listdir(self.feat_dir/self.split))
        finids = []
        missing = []
        for id in ids:
            if f'vid_{id}.joblib' in files:
                finids.append(id)
            else:missing.append(id)
        print(f"missing:{missing}")
        return finids,sents

class ClipLstm(pl.LightningModule):
    
    def __init__(self,hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.vid_encoder,self.text_encoder,self.finlin = self.get_nets(**hparams.network_params)
        
    def get_nets(self,bidirectional = True,vid_lyrs = 2,vid_hidim=64,vid_fsz=64\
        ,txt_lyrs=2,txt_fsz=64,act='Relu'):
        vid_enc = []
        last_dim = 512
        
        self.lstm = nn.LSTM(last_dim,vid_hidim,\
        vid_lyrs,bidirectional=bidirectional,batch_first=True)
        vid_enc.append(self.lstm)
        if act == 'Relu':
            self.activation = nn.ReLU()
        vid_enc.append(self.activation)
        self.vidfinlyr = nn.Linear(vid_hidim,vid_fsz)
        vid_enc.append(self.vidfinlyr)
        self.txtlyr = nn.Linear(512,txt_fsz)
        txt_enc = nn.Sequential(self.txtlyr,self.activation)
        
        return vid_enc,txt_enc,nn.Linear(txt_fsz,2)
            



    def forward(self,vid,txt):
        #fixing for now
        #torch.squeeze(self.start(self.shared(input)))
        lstm,act,lin = self.vid_encoder
        hiddens, (final_h, final_c) = lstm(vid.float())
        vid_out = lin(act(torch.mean(final_h,dim=0)))
        txt_out = self.text_encoder(txt.float())
        return self.finlin(vid_out+txt_out)
        
    def training_step(self,batch,batch_idx):

        vid,txt,st,end = batch
        loss_fn = nn.MSELoss()
        # forward
        lstm,act,lin = self.vid_encoder
        hiddens, (final_h, final_c) = lstm(vid.float())
        vid_out = lin(act(torch.mean(final_h,dim=0)))
        txt_out = self.text_encoder(txt.float())
        preds = self.finlin(vid_out+txt_out)


        #preds = self(vid,txt)
        st_loss = loss_fn(preds[:,0].float(),torch.squeeze(st).float())
        end_loss = loss_fn(preds[:,1].float(),torch.squeeze(end).float())
        #loss_end = nn.CrossEntropyLoss()
        #import pdb;pdb.set_trace()
        #st_l = loss_st(torch.squeeze(self.start(self.shared(input))).float(),st.float())
        #end_l = loss_st(torch.squeeze(self.end(self.shared(input))).float(),end.float())
        loss = st_loss + end_loss
        self.log("train_loss",loss,on_step=True)
        return loss

    def validation_step(self,batch,batch_idx):

        vid,txt,st,end = batch
        loss_fn = nn.MSELoss()
        # forward
        lstm,act,lin = self.vid_encoder
        hiddens, (final_h, final_c) = lstm(vid.float())
        vid_out = lin(act(torch.mean(final_h,dim=0)))
        txt_out = self.text_encoder(txt.float())
        preds = self.finlin(vid_out+txt_out)

        #preds = self(vid,txt)
        st_loss = loss_fn(preds[:,0].float(),torch.squeeze(st).float())
        end_loss = loss_fn(preds[:,1].float(),torch.squeeze(end).float())
        #loss_end = nn.CrossEntropyLoss()
        #import pdb;pdb.set_trace()
        #st_l = loss_st(torch.squeeze(self.start(self.shared(input))).float(),st.float())
        #end_l = loss_st(torch.squeeze(self.end(self.shared(input))).float(),end.float())
        loss = st_loss + end_loss
        self.log("val_loss",loss,on_step=True, on_epoch=True)
        return loss
        
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


from argparse import Namespace
FEAT_DIR = pathlib.Path('/common/users/vk405/CLIP_FEAT')
RAWFRAME_DIR = pathlib.Path('/common/users/vk405/Youcook/')

cfg = Namespace(
    version = 'clip_lstm',
    id = 0,
    FEAT_DIR = FEAT_DIR,
    RAWFRAME_DIR = RAWFRAME_DIR,
    artifacts_loc = "/common/home/vk405/Projects/Crossmdl/nbs/",
    data_dir = "/common/home/vk405/Projects/Crossmdl/Data/YouCookII/",
    trn_split = 0.8,
    mode = 'train',
    split = 'training',
    loggers = ["csv"],
    seed = 0,
    network_params = {'bidirectional':True,'vid_lyrs':2,\
        'vid_hidim':64,'vid_fsz':64\
        ,'txt_lyrs':2,'txt_fsz':64,'act':'Relu'},
    cbs = ["checkpoint","early_stop"],
    trainer = {'log_every_n_steps': 1,
    'max_epochs': 40},
    checkpoint = {"every_n_epochs": 1,
    "monitor": "val_loss"},
    early_stop = {"monitor":"val_loss","mode":"min","patience":5},
    lr = 1e-4

)


def run(cfg):
    #pl.seed_everything(cfg.seed)
    dir = cfg.artifacts_loc
    version = str(cfg.version)
    logger_list = get_loggers(dir, version,cfg.loggers)
    cbs = []
    if "early_stop" in cfg.cbs:
        #? does'nt really work atm
        params = cfg.early_stop
        earlystopcb = EarlyStopping(**params, min_delta=0.00, verbose=False)
        cbs.append(earlystopcb)
    if "checkpoint" in cfg.cbs:
        store_path = dir + "ckpts/" + str(cfg.version) + "/"
        isExist = os.path.exists(store_path)
        if not isExist:
            os.makedirs(store_path)
        fname = "{epoch}-{train_loss:.2f}"
        params = cfg.checkpoint
        checkptcb = ModelCheckpoint(**params, dirpath=store_path, filename=fname)
        cbs.append(checkptcb)

    #wandb.init(project="videoretrieval", config=cfg)
    if cfg.mode == 'train':
        d = Dset(cfg.RAWFRAME_DIR,cfg.FEAT_DIR,cfg.split)
        trn_sz = int(len(d)*cfg.trn_split)
        val_sz = len(d)-trn_sz
        trndset,valdset = random_split(d,[trn_sz,val_sz])
        trnl = DataLoader(trndset,batch_size=64,shuffle=True,num_workers = 5)
        vall = DataLoader(valdset,batch_size=64)
        hparams = cfg    
        net = ClipLstm(hparams)
        trainer = pl.Trainer(
            logger=logger_list,callbacks=cbs,accelerator='gpu',decives=[0],deterministic=True, **cfg.trainer
        )
        trainer.fit(net, trnl,vall)
        return trainer
        #trainer.tune(net,train_loader)
            
    else:
        pass
    


if __name__ == "__main__":
    t = run(cfg)