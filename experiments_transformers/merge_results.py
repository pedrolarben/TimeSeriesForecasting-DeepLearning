import os
import pandas as pd
mypath = "./results"

all_filenames = []
for folder in os.listdir(mypath):
    
    for file in os.listdir(os.path.join(mypath,folder)):
        if file.endswith(".csv"):
            all_filenames.append(os.path.join(mypath,folder,file))



#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f,sep=";") for f in all_filenames ])

combined_csv = combined_csv.drop(combined_csv.columns[[0,14,15]], axis=1)
print(combined_csv.columns)
index = list(range(0,combined_csv.shape[0]))
combined_csv = combined_csv.set_index(pd.Index(index))
#export to csv
combined_csv.to_csv( "combined_csv.csv", encoding='utf-8-sig',sep=";")