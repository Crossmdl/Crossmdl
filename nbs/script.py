# pad your sequences

from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
import os
import json
import joblib
from torch.utils.data import Dataset,DataLoader
from itertools import repeat
import pandas as pd
import math
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

#utils

# all values are equally distributed
#df


#utils



def get_labels(ids,annotns_file):

    label_info = {}
    with open(annotns_file) as json_file:
        annotns = json.load(json_file)
        #print(annotns.keys())
        for _,vidname in ids:
            #import pdb;pdb.set_trace()
            if vidname in annotns:
                #import pdb;pdb.set_trace()
                duration = annotns[vidname]['duration']
                annot = annotns[vidname]['annotations']
                labels = []
                #import pdb;pdb.set_trace()
                for segment_info in annot:
                    interval = segment_info['segment']
                    st_end = [interval[0],interval[-1]]
                    sent = segment_info['sentence']
                    labels.append((st_end,sent,duration))

                label_info[vidname] = labels
            else:
                print(f"label for {vidname} not present")
    return label_info

def get_vids(base_dir,split):
    trn_split = base_dir+split
    trn_idlst = []
    trn_vidlst = []

    f = open(trn_split,'r')
    for line in f:
        id_,vid = line.split('/')
        vid = vid.strip('\n')
        trn_idlst.append(id_)
        trn_vidlst.append(vid)
        #print(vid)
        #break
    f.close()
    return trn_idlst,trn_vidlst

    
def get_features(data_dir,split='val',feat_dir='/common/users/vk405/feat_csv/'):
    #feat_dir = data_dir
    splits_dir = data_dir+'splits/'
    if split == 'val':
        feat_split_dir = feat_dir+'val_frame_feat_csv/'  
        vid_num,vid_name = get_vids(splits_dir,'val_list.txt')  
    elif split == 'train':
        feat_split_dir = feat_dir+'train_frame_feat_csv/'  
        vid_num,vid_name = get_vids(splits_dir,'train_list.txt') 
    elif split == 'test':
        feat_split_dir = feat_dir+'test_frame_feat_csv/'  
        vid_num,vid_name = get_vids(splits_dir,'test_list.txt')
    else:
        raise NotImplementedError(f'unknown split: {split}')     
    feat_list = {}
    vid_dtls = []
    for num,name in zip(vid_num,vid_name):
        feat_loc = os.path.join(feat_split_dir, f'{num}/{name}/0001/')
        #import pdb;pdb.set_trace()
        if os.path.isdir(feat_loc):
            feat_files = feat_loc + os.listdir(feat_loc)[0]
            feat_list[name] = feat_files
            #feat_list.append(feat_files)
            vid_dtls.append((num,name))
        else:
            print(f"video : {num}/{name} not found")
    assert len(feat_list) == len(vid_dtls),"get-features is giving incorrect features"
    return feat_list,vid_dtls






def get_raw_labels(ids,annotns_file):

    label_info = {}
    with open(annotns_file) as json_file:
        annotns = json.load(json_file)
        print(annotns.keys())
        for _,vidname in ids:
            #import pdb;pdb.set_trace()
            if vidname in annotns['database']:
                #import pdb;pdb.set_trace()
                duration = annotns['database'][vidname]['duration']
                annot = annotns['database'][vidname]['annotations']
                labels = []
                #import pdb;pdb.set_trace()
                for segment_info in annot:
                    interval = segment_info['segment']
                    sent = segment_info['sentence']
                    labels.append((interval,sent,duration))

                label_info[vidname] = labels
            else:
                print(f"label for {vidname} not present")
    return label_info

def regress_labels(raw_labels):
    regress_labels = {}
    for key in raw_labels:
        new_labels = []
        for item in raw_labels[key]:
            rng,sent,vidlen = item
            mid = sum(rng)/2
            duration = rng[-1]-rng[0]
            mid_pred = (1/vidlen)*mid # location of mid-point w.r.t video length
            duration_pred = (1/vidlen)*duration
            new_labels.append(([mid_pred,duration_pred],sent))
        regress_labels[key] = new_labels
    return regress_labels
            
            
    
    
    


#dataset
# Dataset/loader
# This is newer version
class YoucookDset2(Dataset):
    def __init__(self,data_dir='/common/home/vk405/Projects/Crossmdl/Data/YouCookII/'\
        ,split='train',framecnt=499):
        self.feat_locs = {}
        self.split = split
        self.data_dir = data_dir
        self.framecnt = framecnt
        #self.use_precomp_emb = use_precomp_emb
        self.text_emb = None
        if self.split != 'test':
            self.annotns_file = data_dir+'annotations/segment_youcookii_annotations_trainval.json'
        else:
            raise NotImplementedError(f"Split:{self.split},not yet correctly implemented")
        # if self.use_precomp_emb:
        #     self.txt_emb = joblib.load(os.path.join(self.data_dir,'emb.joblib'))
        #feat_locs = {'Ysh60eirChU': location of the video}
        self.feat_locs,vids = get_features(self.data_dir,split=self.split)
        assert len(vids) == len(self.feat_locs),"features are wrong"
        #import pdb;pdb.set_trace()
        #label_info = get_labels(vids,self.annotns_file)
        #self.labelencoder = LabelEncoder2()
        self.final_labels = get_labels(vids,self.annotns_file)
        #self.labelencoder.fit_transform(label_info)
        
        #regress_labels(label_info)
        #(vid_id,seg_id)
        self.data = self.update_data()

                
            
    def __len__(self):
        return len(self.data)

    def overlap_frac(self,base_rng,tst_rng):
        #1.Returns the fraction of frames that are overlapping in tst_rng with base_rng
        #2.both ends inclusive
        sz = tst_rng[-1]-tst_rng[0]+1
        lbl_ids = set(np.arange(base_rng[0],base_rng[-1]+1))
        frame_ids = set(np.arange(tst_rng[0],tst_rng[-1]+1))
        inter = frame_ids.intersection(lbl_ids)
        assert sz != 0,"base frame rng is zero"
        return len(inter)/sz


    def update_data(self):
        data = []
        max_cnt = 50
        for key in self.final_labels:
            segments = self.final_labels[key]
            for ind,seg in enumerate(segments):
                #trn_points = []
                st_end,txt,vid_len = seg
                main_seg = (key,self.feat_locs[key],st_end[0],st_end[-1],ind,1.0)
                data.append(main_seg)
                frame_width = st_end[-1]-st_end[0] + 1
                extra_frames = []
                for cnt,new_st in enumerate(range(st_end[0]+1,st_end[-1]+1)):
                    #forward sliding
                    new_end = new_st+frame_width
                    if (cnt<max_cnt)and (0<=new_st<self.framecnt and 0<=new_st<self.framecnt):
                        extra_frames.append((new_st,new_end))
                for cnt,new_end in enumerate(range(st_end[-1],st_end[0],-1)):
                    #backward sliding
                    new_st = new_end-frame_width
                    if (cnt<max_cnt)and (0<=new_st<self.framecnt and 0<=new_st<self.framecnt):
                        extra_frames.append((new_st,new_end))
                #import pdb;pdb.set_trace()
                for ex_seg in extra_frames:
                    label = self.overlap_frac(st_end,ex_seg)
                    data.append((key,self.feat_locs[key],ex_seg[0],ex_seg[-1],ind,label))
        return data

    def __getitem__(self,idx):
        return self.data[idx]
        

           



from collections import defaultdict

        
# Now changing the model to account for this change


#model utils

#!pip install transformers

def init_parameters_xavier_uniform(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

def scaled_dot(query, key, mask_key=None):  
    score = torch.matmul(query, key.transpose(-2, -1))
    score /= math.sqrt(query.size(-1))
    if mask_key is not None:
        score = score.masked_fill(mask_key, -1e18)  # Represents negative infinity
    return score      
            
def attend(query, key, value, mask_key=None, dropout=None):
    # TODO: Implement
    # Use scaled_dot, be sure to mask key
    #smax = nn.Softmax(-1)
    #import pdb;pdb.set_trace()
    score = scaled_dot(query,key,mask_key)  
    attention = F.softmax(score,dim=-1)
    if dropout is not None:#do = nn.Dropout(dropout)
        attention = dropout(attention)
    answer = torch.matmul(attention,value) 
    # Convexly combine value embeddings using attention, this should be just a matrix-matrix multiplication.
    return answer, attention



def split_heads(batch, num_heads):  
    (batch_size, length, dim) = batch.size()  # These are the expected batch dimensions.
    assert dim % num_heads == 0  # Assert that dimension is divisible by the number of heads.
    dim_head = dim // num_heads

    # No new memory allocation
    splitted = batch.view(batch_size, -1, num_heads, dim_head).transpose(1, 2)  
    return splitted  # (batch_size, num_heads, length, dim_head), note that now the last two dimensions are compatible with our attention functions. 




def merge_heads(batch):  
    (batch_size, num_heads, length, dim_head) = batch.size()  # These are the expected batch dimensions.

    # New memory allocation (reshape), can't avoid.
    merged = batch.transpose(1, 2).reshape(batch_size, -1, num_heads * dim_head)
    return merged  # (batch_size, length, dim)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout_rate=0.1):
        super().__init__()
        assert dim % num_heads == 0

        self.linear_query = nn.Linear(dim, dim)
        self.linear_key = nn.Linear(dim, dim)
        self.linear_value = nn.Linear(dim, dim)
        self.linear_final = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.num_heads = num_heads

    def forward(self, query, key, value, mask_key=None, layer_cache=None,
              memory_attention=False):
        """
        INPUT
          query: (batch_size, length_query, dim)
          key: (batch_size, length_key, dim)
          value: (batch_size, length_key, dim_value)
          mask_key: (*, 1, length_key) if queries share the same mask, else
                    (*, length_query, length_key)
          layer_cache: if not None, stepwise decoding (cache of key/value)
          memory_attention: doing memory attention in stepwise decoding?
        OUTPUT
          answer: (batch_size, length_query, dim_value)
          attention: (batch_size, num_heads, length_query, length_key) else
        """
        batch_size = query.size(0)

        query = self.linear_query(query)
        query = split_heads(query, self.num_heads)  # (batch_size, num_heads, -1, dim_head)

        def process_key_value(key, value):  # Only called when necessary.
            key = self.linear_key(key)
            key = split_heads(key, self.num_heads)
            value = self.linear_value(value)
            value = split_heads(value, self.num_heads)
            return key, value

        #import pdb;pdb.set_trace()
        if layer_cache is None:
            key, value = process_key_value(key, value)
        else:
            assert query.size(2) == 1  # Stepwise decoding
            
            if memory_attention:
                if layer_cache['memory_key'] is None:  # One-time calculation
                    key, value = process_key_value(key, value)
                    # (batch_size, num_heads, length_memory, dim)
                    layer_cache['memory_key'] = key
                    layer_cache['memory_value'] = value

                key = layer_cache['memory_key']
                value = layer_cache['memory_value']

            else:  # Self-attention during decoding
                key, value = process_key_value(key, value)
                assert key.size(2) == 1 and value.size(2) == 1
                
                # Append to previous.
                if layer_cache['self_key'] is not None:
                    key = torch.cat((layer_cache['self_key'], key), dim=2)
                    value = torch.cat((layer_cache['self_value'], value), dim=2)
                    
                 # (batch_size, num_heads, length_decoded, dim)
                layer_cache['self_key'] = key  # Recache.
                layer_cache['self_value'] = value
        # Because we've splitted embeddings into heads, we must also split the mask. 
        # And because each query uses the same mask for all heads (we don't use different masking for different heads), 
        # we can specify length 1 for the head dimension.
        if mask_key is not None:  
            mask_key = mask_key.unsqueeze(1)  # (batch_size, 1, -1, length_key)

        answer, attention = attend(query, key, value, mask_key, self.dropout)

        answer = merge_heads(answer)  # (batch_size, length_key, dim)
        answer = self.linear_final(answer)

        return answer, attention

class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, dim_hidden, drop_rate=0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, dim_hidden)
        self.w2 = nn.Linear(dim_hidden, dim)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.drop1 = nn.Dropout(drop_rate)
        self.relu = nn.ReLU()
        self.drop2 = nn.Dropout(drop_rate)
    def forward(self, x):
        inter = self.drop1(self.relu(self.w1(self.layer_norm(x))))
        output = self.drop2(self.w2(inter))
        return output + x




class SinusoidalPositioner(nn.Module):
    def __init__(self, dim, drop_rate=0.1, length_max=5000):
        super().__init__()
        frequency = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.) / dim))  # Using different frequency for each dim
        positions = torch.arange(0, length_max).unsqueeze(1)
        wave = torch.zeros(length_max, dim)
        wave[:, 0::2] = torch.sin(frequency * positions)
        wave[:, 1::2] = torch.cos(frequency * positions)
        self.register_buffer('wave', wave.unsqueeze(0))  # (1, length_max, dim)
        self.dropout = nn.Dropout(drop_rate)
        self.dim = dim
        self.length_max = length_max
    def forward(self, x, step=-1):
        assert x.size(-2) <= self.length_max

        if step < 0:  # Take the corresponding leftmost embeddings.
            position_encoding = self.wave[:, :x.size(-2), :]
        else:  # Take the embedding at the step.
            position_encoding = self.wave[:, step, :]

        x = x * math.sqrt(self.dim)
        return self.dropout(x + position_encoding)





class CrossAttentionLayer(nn.Module):
    def __init__(self,dim,num_heads,dim_hidden,drop_rate):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.context_attention = MultiHeadAttention(dim, num_heads, drop_rate)
        self.drop = nn.Dropout(drop_rate)
        self.feedforward = PositionwiseFeedForward(dim, dim_hidden, drop_rate)
        
    def forward(self,target,memory,layer_cache=None,mask_key=None):
        
        cross_attn_target = self.layer_norm(target)
        attended, attention = self.context_attention(cross_attn_target,memory,memory,layer_cache=layer_cache,memory_attention=True,mask_key=mask_key)
        
        attended = target + self.drop(attended)
        
        return self.feedforward(attended),attention



layer_cache = {'memory_key': None, 'memory_value': None, 'self_key': None, 'self_value': None}
    


#model 






class CrossattnModel(pl.LightningModule):
    def __init__(self,hparams,dset=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        #self.hparams = hparams
        #import pdb;pdb.set_trace()
        #self.net= Model(hparams)
        #self.hparams  = hparams
        self.positioner = SinusoidalPositioner(self.hparams.edim, drop_rate=0., length_max=1000)
        self.attn = CrossAttentionLayer(self.hparams.edim,self.hparams.nheads,\
                           self.hparams.attnhdim,self.hparams.dropoutp)
        self.wrdcnn =  nn.Conv1d(self.hparams.wrdim, self.hparams.edim, 1, stride=1)
        self.vidcnn =  nn.Conv1d(self.hparams.vidim, self.hparams.edim, 1, stride=1)
        self.hid_layer = nn.Linear(self.hparams.edim,self.hparams.hdim)
        self.out_layer = nn.Linear(self.hparams.hdim,1)
        #self.init_parameters_xavier_uniform()
        self.dset = dset

    def forward(self,x):
        #keep this for inference
        vid_feat,wrd_feat = x
        mask = self.get_mask(vid_feat)
        # pad the features with zeros
        vid_feat = pad_sequence(vid_feat,batch_first=True)
        out = self.net((vid_feat.float(),wrd_feat.float()),mask_key=mask)
        return out
        
    def net(self,x,mask_key=None):
        vid_x,wrd_x = x
        #import pdb;pdb.set_trace()
        wrd_x = wrd_x.unsqueeze(1).transpose(1,2)
        vid_x = vid_x.transpose(1,2)
        #print(f"inside model, wrd_x:{wrd_x.shape},vi")
        tgt = self.wrdcnn(wrd_x.float()).transpose(1,2)
        src = self.vidcnn(vid_x.float()).transpose(1,2)
        src_posencode = self.positioner(src)
        #for i in range(self.hparams.lyrs):
        # ?create mask_key and send it
        attended,attn_score = self.attn(tgt,src_posencode,mask_key=mask_key)
            #tgt = 
        out = F.sigmoid(self.out_layer(F.relu(self.hid_layer(F.relu(attended)))))
        return out

    def get_mask(self,batch):
        #device = batch.device
        sq_lens = list(map(lambda x:x.size(0),batch))
        ln_key = batch[0].size(-1)
        # mask = (batch_size, 1, length_key)(all queries have same mask)
        mask = torch.ones(len(sq_lens),1,max(sq_lens))
        for ind,ele in enumerate(sq_lens):
            mask[ind,:,:ele] = 0.0
        return mask


    def training_step(self,batch,batch_idx):
        #for tranining
        vid_feat,wrd_feat,labels = batch
        # here vid_feat is list of video frames of varying length
        mask = self.get_mask(vid_feat)
        # pad the features with zeros
        vid_feat = pad_sequence(vid_feat,batch_first=True)
        x_hat = self.net((vid_feat.float(),wrd_feat.float()),mask_key=mask)
        #import pdb;pdb.set_trace()
        #loss = nn.BCELoss()
        loss = F.binary_cross_entropy(x_hat.squeeze().float(), labels.squeeze().float())
        #print(f"inside train step, loss:{loss}")
        self.log("train_loss",loss,on_step=True)
        return loss

    def validation_step(self,batch,batch_idx):
        #for validation
        vid_feat,wrd_feat,labels = batch
        mask = self.get_mask(vid_feat)
        vid_feat = pad_sequence(vid_feat,batch_first=True)
        x_hat = self.net((vid_feat.float(),wrd_feat.float()),mask_key=mask)
        
        loss = F.binary_cross_entropy(x_hat.squeeze().float(), labels.squeeze().float())
        #print(f"inside train step, loss:{loss}")
        self.log("val_loss",loss,on_step=False, on_epoch=True)
        return loss


    def configure_optimizers(self):
        lr = self.hparams.lr if 'lr' in self.hparams else 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

    
    def init_parameters_xavier_uniform(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    
        

#4096 works
import wandb
import logging
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
import shutil

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



# Runnin/training

data_dir='/common/home/vk405/Projects/Crossmdl/Data/YouCookII/'

class MyCollator(object):
    def __init__(self,**kwargs):
        self.kwargs = kwargs
    def __call__(self, data):
        # do something with batch and self.params
        labels = []
        vid_embs = []
        txt_embs = []
        #namedtuple("vid_id", "vid_loc","start","end","segid","label")
        batched_data = pd.DataFrame(data,columns=["vid_id", "vid_loc","start","end","segid","label"])
        unique_locs = batched_data['vid_loc'].unique()
        #import pdb;pdb.set_trace()
        for loc in unique_locs:
            locwise = batched_data[batched_data['vid_loc']==loc]
            tot_vid = pd.read_csv(loc).values
            txtemb = None
            for ind,ele in locwise.iterrows():
                #vid_id,_,st,end,segid,label = ele
                #import pdb;pdb.set_trace()
                if not txtemb:
                    txtemb = self.kwargs['txtemb'][ele['vid_id']]
                vid_embs.append(torch.tensor(tot_vid[ele['start']:ele['end']+1]))
                #import pdb;pdb.set_trace()
                txt_embs.append(torch.tensor(txtemb[ele['segid']]))
                labels.append(torch.tensor(ele['label']))
        #return (np.stack(vid_embs),np.stack(txt_embs)),np.stack(labels)
        return vid_embs,torch.stack(txt_embs),torch.stack(labels)



def run(cfg):
    pl.seed_everything(cfg.seed)
    dir = cfg.artifacts_loc
    version = str(cfg.version)
    logger_list = get_loggers(dir, version,cfg.loggers)
    cbs = []
    if "early_stop" in cfg.cbs:
        #? does'nt really work atm
        params = cfg.model.cbs.early_stop
        earlystopcb = EarlyStopping(**params, min_delta=0.00, verbose=False)
        cbs.append(earlystopcb)
    if "checkpoint" in cfg.cbs:
        store_path = dir + "ckpts/" + str(cfg.version) + "/"
        isExist = os.path.exists(store_path)
        if isExist and os.path.isdir(store_path):
            shutil.rmtree(store_path)
        # then create fresh
        if not isExist:
            os.makedirs(store_path)
        fname = "{epoch}-{train_loss:.2f}"
        params = cfg.checkpoint
        checkptcb = ModelCheckpoint(**params, dirpath=store_path, filename=fname)
        cbs.append(checkptcb)
    if "wandb" in cfg.loggers:
        wandb.init(project="videoretrieval", config=cfg)
    if cfg.mode == 'train':
        youcookdata = YoucookDset2(data_dir=cfg.data_dir,split=cfg.mode)
        if cfg.use_precomp_emb:
            global_txt = joblib.load(os.path.join(data_dir,'emb.joblib'))
            collate_wrapper = MyCollator(txtemb=global_txt)
        #pin_memory=True
        train_loader = DataLoader(youcookdata,\
            batch_size=cfg.batch_size,collate_fn=collate_wrapper,\
                shuffle=True,num_workers=10)
           
        net = CrossattnModel(cfg)
        #gpus=3,accelerator='ddp'
        trainer = pl.Trainer(
            logger=logger_list,callbacks=cbs,deterministic=True, **cfg.trainer
        )
        trainer.fit(net, train_loader)
        return trainer
        #trainer.tune(net,train_loader)
            
    else:
        pass
    


if __name__ == "__main__":
    from argparse import Namespace
    cfg = Namespace(
        version = 'trail_debug',
        id = 0,
        artifacts_loc = "/common/home/vk405/Projects/Crossmdl/nbs/",
        data_dir = "/common/home/vk405/Projects/Crossmdl/Data/YouCookII/",
        mode = 'train',
        loggers = ["csv"],
        seed = 0,
        cbs = ["checkpoint"],
        trainer = {'log_every_n_steps': 50,
        'max_epochs': 3},
        checkpoint = {"every_n_epochs": 1,
        "monitor": "train_loss"},
        use_precomp_emb = True,
        edim = 100,
        attnhdim = 50,
        nheads = 10,
        wrdim = 768,
        vidim = 512,
        hdim = 30,
        dropoutp=0.0,
        batch_size=512,
        framecnt=499


    )
    run(cfg)



    

