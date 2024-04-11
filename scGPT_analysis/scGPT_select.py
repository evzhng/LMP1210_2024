#!/usr/bin/env python
# coding: utf-8

# # GRN Inference on Pre-trained Model
# Here we use the pre-trained blood model as an example for GRN inference, particularly regarding gene program extraction and network visualization. We also present the cell-type specific activations within these gene programs on the Immune Human dataset, as a soft validation for the zero-shot performance. 
# 
# Note that GRN inference can be performed on pre-trained and finetuned models as showcased in our manuscript.
# 
# Users may perform scGPT's gene-embedding-based GRN inference in the following steps:
# 
#      1. Load optimized scGPT model (pre-trained or fine-tuned) and data
#      
#      2. Retrieve scGPT's gene embeddings
#      
#      3. Extract gene programs from scGPT's gene embedding network
#      
#      4. Visualize gene program activations on dataset of interest
#      
#      5. Visualize the interconnectivity of genes within select gene programs
#      

# In[1]:


import copy
import json
import os
from pathlib import Path
import sys
import warnings

import torch
from anndata import AnnData
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import tqdm
import gseapy as gp

from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.tasks import GeneEmbedding
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed 

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')


# In[4]:


set_seed(42)
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
n_hvg = 1200
n_bins = 51
mask_value = -1
pad_value = -2
n_input_bins = n_bins


# ## Step 1: Load pre-trained model and dataset

# ### 1.1  Load pre-trained model
# The blood pre-trained model can be downloaded via this [link](https://drive.google.com/drive/folders/1kkug5C7NjvXIwQGGaGoqXTk_Lb_pDrBU).

# In[5]:


# Specify model path; here we load the pre-trained scGPT blood model
model_dir = Path("../save/scGPT_bc")
model_config_file = model_dir / "args.json"
model_file = model_dir / "best_model.pt"
vocab_file = model_dir / "vocab.json"

vocab = GeneVocab.from_file(vocab_file)
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)

# Retrieve model parameters from config files
with open(model_config_file, "r") as f:
    model_configs = json.load(f)
print(
    f"Resume model from {model_file}, the model args will override the "
    f"config {model_config_file}."
)
embsize = model_configs["embsize"]
nhead = model_configs["nheads"]
d_hid = model_configs["d_hid"]
nlayers = model_configs["nlayers"]
n_layers_cls = model_configs["n_layers_cls"]

gene2idx = vocab.get_stoi()


# In[6]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    vocab=vocab,
    pad_value=pad_value,
    n_input_bins=n_input_bins,
)

try:
    model.load_state_dict(torch.load(model_file))
    print(f"Loading all model params from {model_file}")
except:
    # only load params that are in the model and match the size
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file, map_location=torch.device('cpu'))
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    for k, v in pretrained_dict.items():
        print(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

model.to(device)


# ### 1.2  Load dataset of interest
# The Immune Human dataset can be downloaded via this [link](https://figshare.com/ndownloader/files/25717328).

# In[7]:


# Specify data path; here we load the Immune Human dataset
data_dir = Path("../data")
adata = sc.read(
    str(data_dir / "910_2.h5ad"), cache=True
)  # 33506 × 12303
ori_batch_col = None
adata.obs["celltype"] = adata.obs["Response"].astype(str)
data_is_raw = False


# In[8]:


# Preprocess the data following the scGPT data pre-processing pipeline
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=0,  # step 1
    filter_cell_by_counts=0,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=n_hvg,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)
preprocessor(adata, batch_key=None)


# ## Step 2: Retrieve scGPT's gene embeddings
# 
# Note that technically scGPT's gene embeddings are data independent. Overall, the pre-trained foundation model contains 30+K genes. Here for simplicity, we focus on a subset of HVGs specific to the data at hand.

# In[9]:


# Retrieve the data-independent gene embeddings from scGPT
gene_ids = np.array([id for id in gene2idx.values()])
gene_embeddings = model.encoder(torch.tensor(gene_ids, dtype=torch.long).to(device))
gene_embeddings = gene_embeddings.detach().cpu().numpy()


# In[10]:


# Filter on the intersection between the Immune Human HVGs found in step 1.2 and scGPT's 30+K foundation model vocab
gene_embeddings = {gene: gene_embeddings[i] for i, gene in enumerate(gene2idx.keys()) if gene in adata.var.index.tolist()}
print('Retrieved gene embeddings for {} genes.'.format(len(gene_embeddings)))


# In[11]:


# Construct gene embedding network
embed = GeneEmbedding(gene_embeddings)


# ## Step 3: Extract gene programs from gene embedding network

# ### 3.1  Perform Louvain clustering on the gene embedding network

# In[12]:


# Perform Louvain clustering with desired resolution; here we specify resolution=40
gdata = embed.get_adata(resolution=20)
# Retrieve the gene clusters
metagenes = embed.get_metagenes(gdata)


# ### 3.2  Filter on clusters with 5 or more genes

# In[13]:


# Obtain the set of gene programs from clusters with #genes >= 5
mgs = dict()
for mg, genes in metagenes.items():
    if len(genes) > 4:
        mgs[mg] = genes


# In[14]:

mgs_str = "mgs: " + str(mgs) + "\n"
print(mgs_str)
adata_str = "adata: " + str(adata) + "\n"
print(adata_str)
metagenes_str = "metagenes: " + str(metagenes) + "\n"
print(metagenes_str)

strs = [mgs_str, adata_str, metagenes_str]

# Here are the gene programs identified
with open("my_log_file", 'w') as f:
    f.writelines(strs)

sns.set_theme(rc={'figure.figsize':(20,60)})
sns.set(font_scale=1.5)
embed.score_metagenes(adata, metagenes)
#embed.plot_metagenes_scores(adata, mgs, "celltype")
embed.plot_metagenes_scores(adata, metagenes, "celltype", plot="910_rerun.png")
