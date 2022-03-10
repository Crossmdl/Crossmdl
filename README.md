# Crossmdl
A project exploring `Cross Modal Representation Learning` - for retrieval and alignment between different modalities.

## Install instructions(specific to ilab)

1. Set-up git ssh
2. Download and set-up miniconda
3. Check the cuda environment (for ilabgup machines it's 11.4)
4. Create conda environment - `conda create -n Crossmdl python=3.8`
5. Manually install torch specific to env - `pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
6. Now install your packages in `requirements.txt`

 
 # TASKS
 
 a. Video-text Alignment of youtube-2 dataset.
 
 Approach/Plan:
 
 Try something similar to [Zhou et al](https://arxiv.org/abs/1703.09788) but using modern architectures(transformers). 
 
 Problems/Issues:
 
 Unfortunately their implementation is in [LUA](https://github.com/LuoweiZhou/ProcNets-YouCook2).
