import json
from collections import defaultdict
import torch as th
from s3dg import S3D
net = S3D('s3d_dict.npy', 512)
net.load_state_dict(th.load('s3d_howto100m.pth'))
data = json.load(open("youcookii_annotations_trainval.json"))
for id in data["database"]:
    sentences = []
    for seg in range(0,100):
        try:
            sentences.append(data["database"][id]["annotations"][seg]["sentence"])
        except:
            continue
    text_emb = net.text_module(sentences)
    th.save(text_emb, "./text_embeddings/"+id+".pt")
    #for seg in id["annotations"]:
        
#text_output = net.text_module(['spread margarine on two slices of white bread', "place a slice of cheese on the bread", "place the bread slices on top of each other and place in a hot pan", "flip the sandwich over and press down", "cut the sandwich in half diagonally"])
#print(text_output['text_embedding'].shape)

