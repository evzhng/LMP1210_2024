import pandas as pd
import torch

from get_embedding import main_gene_selection


# X_df represents your single cell data with cells in rows and genes in columns
X_df = pd.read_csv("../../data/scaled_data_3.csv")
gene_list_df = pd.read_csv('./OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
gene_list = list(gene_list_df['gene_name'])
X_df, to_fill_columns, var = main_gene_selection(X_df, gene_list)

X_df.to_csv ("scaled_data_sc_label.csv")