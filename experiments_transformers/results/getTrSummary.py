import pandas as pd
import os

dataset,best,mean,worst = [],[],[],[]

directory = os.path.join("c:\\","/usr/desarrollo/tsf/TimeSeriesForecasting-DeepLearning/experiments_transformers/results")
for root,dirs,files in os.walk(directory):
    for file in files:
       if file.endswith(".csv"):
           df = pd.read_csv(root + "/"+ file,";")
           dataset.append(df.DATASET[0])
           best.append(df["wape"].min()*100)
           mean.append(df["wape"].mean()*100)
           worst.append(df["wape"].max()*100)
df = pd.DataFrame({"Model" : ["Tr"]*len(dataset),
      "Dataset": dataset,
      "Best": best,
      "Mean": mean,
      "Worst": worst})

df.to_csv("/usr/desarrollo/tsf/TimeSeriesForecasting-DeepLearning/experiments_transformers/TrSummary.csv",index=False)


