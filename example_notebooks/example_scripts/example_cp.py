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
final_data_dir = pathlib.Path("../example_extracted_features/CP_features/")
try:
    shutil.rmtree(tmp_dir)
    # uncomment the line below if you would like to remove the final data directory (e.g. all .csv.gz files)
    # shutil.rmtree(final_data_dir)
except:
    print("No files to remove")

stream = CellProfilerRun(pipeline_path, plugins_directory, idr_id, tmp_dir, final_data_dir, log='example_logs/cp_idrstream.log')


# ## Load in metadata

# In[3]:


data_to_process_tsv = pathlib.Path("example_files/data_to_process.tsv")
data_to_process = pd.read_csv("example_files/data_to_process.tsv", sep="\t", index_col=0)
data_to_process


# ## Initialize Aspera downloader

# In[4]:


# path to users home dir
home_dir_path = pathlib.Path.home()

# find the path in terminal using `ascli config ascp show`
aspera_path = pathlib.Path(f"{home_dir_path}/.aspera/ascli/sdk/ascp")
aspera_key_path = pathlib.Path("example_files/asperaweb_id_dsa.openssh")
screens_path = pathlib.Path("example_files/idr0013-screenA-plates.tsv")
idr_index_name = "idr0013-neumann-mitocheck"

stream.init_downloader(aspera_path, aspera_key_path, screens_path, idr_index_name)


# ## Initialize Fiji preprocessor

# In[5]:


fiji_path = pathlib.Path(f"{home_dir_path}/Desktop/Fiji.app")
perform_illumination_correction = True
stream.init_preprocessor(fiji_path, perform_illumination_correction)


# ## Confirm that GPU is activated for Cellpose to run

# In[6]:


use_GPU = core.use_gpu()
print(f">>> GPU activated? {use_GPU}")


# ## Run idrstream batches

# In[7]:


stream.run_cp_stream(data_to_process, batch_size=3, start_batch=0, batch_nums=[0,1,2])

