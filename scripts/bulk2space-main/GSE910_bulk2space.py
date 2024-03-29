from bulk2space import Bulk2Space
import pandas as pd


df = pd.read_csv("../../data/GSE910_scaled_top_100.csv",index_col=0)
df.reset_index(drop=True, inplace=True)
GSE910 = df.apply(pd.to_numeric, errors='coerce')

GSE910_log2 = pd.read_csv("../../data/GSE910_log2_top_100.csv",index_col=0)
GSE910_sample=pd.DataFrame(index= GSE910_log2.columns)
model = Bulk2Space()

sc_seq = "./MEL80_scRNA.csv"
sc_annotation = "./MEL80_annotation.csv"


for samples in GSE910.index:

    GSE910_row = GSE910.iloc[samples,:].to_list()
    GSE910_sample["Sample"]=GSE910_row
    generate_sc_meta, generate_sc_data= model. load_vae_and_generate (
                        input_bulk_path =GSE910_sample,
                        input_sc_data_path=sc_seq,
                        input_sc_meta_path=sc_annotation,
                        input_st_data_path =sc_seq,
                        input_st_meta_path=sc_annotation,
                        vae_load_dir="./save_model/MEL80_momdel.pth",  # load_dir
                        ratio_num=1,
                        top_marker_num=500,
                        generate_save_dir='./save_model',  # file_dir
                        generate_save_name=samples ,  # file_name
                        gpu=0,
                        hidden_size=256)