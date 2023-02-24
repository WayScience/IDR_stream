#!/usr/bin/env python
# coding: utf-8

# # Example of CellProfiler project for idrstream

# ## Import Libraries

# In[1]:


import pathlib
import pandas as pd
import shutil

from cellpose import core

import sys
sys.path.append("../")
from idrstream.CP_idr import CellProfilerRun


# ## Initialize idrstream

# In[2]:


pipeline_path = pathlib.Path("example_files/CP_files/mitocheck_idr_cp.cppipe")
# need to fill in on fig
plugins_directory = pathlib.Path("../idrstream/CP_Plugins")
idr_id = "idr0013"
tmp_dir = pathlib.Path("../tmp/")
final_data_dir = pathlib.Path("../example_extracted_features/CP_features")
try:
    shutil.rmtree(tmp_dir)
    # uncomment the line below if you would like to remove the final data directory (e.g. all .csv.gz files)
    # shutil.rmtree(final_data_dir)
except:
    print("No files to remove")

stream = CellProfilerRun(pipeline_path, plugins_directory, idr_id, tmp_dir, final_data_dir, log='example_logs/cp_idrstream.log')


# ## Initialize CellProfiler metadata compiler

# In[3]:


data_to_process_tsv = pathlib.Path("example_files/data_to_process.tsv")
metadata_save_path = pathlib.Path("example_files/data_to_process.csv")

stream.convert_tsv_to_csv(data_to_process_tsv, metadata_save_path)


# ## Load in metadata

# In[4]:


data_to_process = pd.read_csv("example_files/data_to_process.tsv", sep="\t", index_col=0)
data_to_process


# ## Initialize Aspera downloader

# In[5]:


# path to users home dir
home_dir_path = pathlib.Path.home()

# find the path in terminal using `ascli config ascp show`
aspera_path = pathlib.Path(f"{home_dir_path}/.aspera/ascli/sdk/ascp")
aspera_key_path = pathlib.Path("example_files/asperaweb_id_dsa.openssh")
screens_path = pathlib.Path("example_files/idr0013-screenA-plates.tsv")

stream.init_downloader(aspera_path, aspera_key_path, screens_path)


# ## Initialize Fiji preprocessor

# In[6]:


fiji_path = pathlib.Path(f"{home_dir_path}/Desktop/Fiji.app")
stream.init_preprocessor(fiji_path)


# ## Confirm that GPU is activated for Cellpose to run

# In[7]:


use_GPU = core.use_gpu()
print(f">>> GPU activated? {use_GPU}")
# logger_setup()


# ## Run idrstream batches

# In[8]:


stream.run_cp_stream(data_to_process, batch_size=3, start_batch=0, batch_nums=[0,1,2])

