#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pathlib

import sys
sys.path.append("../")
from idrstream.merge_CP_DP import merge_CP_DP_batch_data, save_merged_CP_DP_run


# In[2]:


cp_data_batch_0 = pd.read_csv(
    "../example_extracted_features/CP_features/batch_0.csv.gz",
    compression="gzip",
    index_col=0,
)
dp_data_batch_0 = pd.read_csv(
    "../example_extracted_features/DP_features/batch_0.csv.gz",
    compression="gzip",
    index_col=0,
)

merge_CP_DP_batch_data(cp_data_batch_0, dp_data_batch_0)


# In[3]:


cp_data_dir_path = pathlib.Path("../example_extracted_features/CP_features/")
dp_data_dir_path = pathlib.Path("../example_extracted_features/DP_features/")
merged_data_dir_path = pathlib.Path("../example_extracted_features/merged_features/")

save_merged_CP_DP_run(cp_data_dir_path, dp_data_dir_path, merged_data_dir_path)

