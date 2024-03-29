from bulk2space import Bulk2Space
import pandas as pd
from deepforest import CascadeForestClassifier

# load required data

bulk_seq = "./GSE910_1.csv"
sc_seq = "./MEL80_scRNA.csv"
sc_annotation = "./MEL80_annotation.csv"

# run model 

model = Bulk2Space()

generate_sc_meta, generate_sc_data = model.train_vae_and_generate(
    input_bulk_path = bulk_seq,
    input_sc_data_path = sc_seq,
    input_sc_meta_path = sc_annotation,
    input_st_data_path=sc_seq,
    input_st_meta_path=sc_annotation,
    ratio_num=1,
    top_marker_num=500,
    gpu=0,
    batch_size=512,
    learning_rate=1e-4,
    hidden_size=256,
    epoch_num=5000,
    vae_save_dir='./save_model',
    vae_save_name='MEL80_momdel',
    generate_save_dir='.',
    generate_save_name='GSE910_1_sc')
