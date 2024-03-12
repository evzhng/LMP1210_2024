import pandas as pd
import re
import os



def PD1_pre_select(infoset, dataset):
    '''''
    inputs are the Response csv  ( ending data.Response.csv) and Gene epxression csv (Data.csv) 
    Select Treatment == PD-1 and Pre therapy (if applicable) 
    output number of selected sample into PD-1.txt 
    Return PD-1 specific portion of each dataset (data_PD_1.csv)
    '''''

    info = pd.read_csv(infoset)
    # set gene_symbol (2nd column) as index, drop the subsequent first column (unnamed)
    data = pd.read_csv(dataset, index_col=1)
    data = data.drop(columns=data.columns[0])
    PD_1_pre_id_data_all = pd.DataFrame()
    # some studies included PRE therapy or ON therapy
    if "Treatment" in info.columns:
        PD_1_pre = info[(info["Treatment"] == "PRE") & (info["Therapy"] == "anti-PD-1")][["sample_id", "response_NR"]]
    else:
        PD_1_pre = info[info["Therapy"] == "anti-PD-1"][["sample_id", "response_NR"]]

    PD_1_pre_id = PD_1_pre["sample_id"].tolist()

    for id in PD_1_pre_id:
        if id in data.columns:
            PD_1_pre_id_data = data.loc[:, [id]]

            response = PD_1_pre.loc[PD_1_pre["sample_id"] == id, "response_NR"].values[0]
            PD_1_pre_id_data.loc[len(PD_1_pre_id_data)] = {"Response": response}
            PD_1_pre_id_data_all = pd.concat([PD_1_pre_id_data_all, PD_1_pre_id_data], axis=1)
        else:
            with open ("PD-1.txt", "a") as f:
                f.write(f"{infoset}, {id}, not found in dataset\n")

        PD_1_pre_id_data_all.to_csv(f"{dataset}_PD_1.csv")


    return PD_1_pre_id_data_all

# Get a list of all the dataset, e.g. Melanoma_GSExxx
working_folder = os.getcwd()
name_pattern = re.compile(r'_[^.]+(?=\.csv)')
dataset_name_list = []
files_in_folder = os.listdir(working_folder)

for files in files_in_folder:
    match = name_pattern.search(files)
    if match:
        dataset_name = match.group()
        dataset_name_list.append(dataset_name)

#For all the datasets, generate PD-1 specific datasets, and concat together
    total_PD_1_data = pd.DataFrame()
for item in dataset_name_list:
    info_name = f"Melanoma{item}.Response.csv"
    data_name = f"Melanoma{item}.csv"

    PD1_data = PD1_pre_select(info_name, data_name)

    print(item, PD1_data.shape)
    total_PD_1_data = pd.concat([total_PD_1_data, PD1_data], axis =1 )

total_PD_1_data.to_csv("total_PD_1.csv")
print(total_PD_1_data.shape)
