import os
import pandas as pd
import numpy as np
mypath = "./timeResultsInfereceWithTrain"

all_filenames = []
for folder in os.listdir(mypath):
    
    for file in os.listdir(os.path.join(mypath,folder)):
        if file.endswith(".csv"):
            all_filenames.append(os.path.join(mypath,folder,file))

def getNInferenceInstances(dataset_name, normalization_method = "minmax",past_history_factor=2.0):

    tmp_data_path = os.path.normpath('../data') + "/{}/{}/{}/".format(dataset_name,
            normalization_method, past_history_factor
        )

    x_test = np.load(tmp_data_path + "x_test.np.npy")
    n_inference_instances = x_test.shape[0]

    return min(n_inference_instances,256)


#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f,sep=";") for f in all_filenames ])

combined_csv = combined_csv.drop([combined_csv.columns[0], "EPOCHS","wape","mase","LOSS","VAL_LOSS"] , axis=1)
print(combined_csv.columns)
index = list(range(0,combined_csv.shape[0]))
combined_csv = combined_csv.set_index(pd.Index(index))
combined_csv["N_INFERENCE_INSTANCES"] = combined_csv["DATASET"].apply(getNInferenceInstances)

combined_csv["INFERENCE_TIME_PER_INSTANCE"] = combined_csv["TEST_TIME"] / combined_csv["N_INFERENCE_INSTANCES"]

#export to csv
combined_csv.to_csv( "combined_csv.csv", encoding='utf-8-sig',sep=";")