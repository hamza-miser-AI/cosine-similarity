import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
dataset=pd.read_csv(r'C:\Users\Elite\Desktop\data mining\students_scores.csv')
pivot_dataset=dataset.pivot_table(index='student_id',columns='subject',values='score',fill_value=0)
#print(pivot_dataset)
#نحول المصفوفة الى dataframe
sim=cosine_similarity(pivot_dataset)
similarity_dataset=pd.DataFrame(sim,index=pivot_dataset.index,columns=pivot_dataset.index)

np.fill_diagonal(sim,np.nan)
#اعلا طالبين متشابهين
max_sim=np.nanmax(sim)
#اقل تشابه بين طالبين مختلفين
min_sim=np.nanmin(sim)
print(f"maximum similarity between two different students:{max_sim}")
print(f"minimum similarity between two different students:{min_sim}")
print(f"cosine_similarity:{similarity_dataset}")