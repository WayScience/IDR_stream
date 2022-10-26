#!/usr/bin/env python
# coding: utf-8

# # Example of CellProfiler project for idrstream

# ## Import Libraries

# In[1]:


import pathlib
import pandas as pd
import shutil
import logging

from idrstream.idr_stream import IdrStream


# ## Load in `data_to_process.tsv` file

# In[2]:


data_to_process = pd.read_csv("example_files/data_to_process.tsv", sep="\t", index_col=0)
# remove A1 wells because of irregular illumination
data_to_process = data_to_process[data_to_process["Well"] != "A1"]
data_to_process = data_to_process.reset_index(drop=True)
data_to_process


# ## Initialize idrstream

# In[3]:


stream_type = "CP"
idr_id = "idr0013"
tmp_dir = pathlib.Path("tmp/")
final_data_dir = pathlib.Path("mitocheck_control_features/")
try:
    shutil.rmtree(tmp_dir)
    # shutil.rmtree(final_data_dir)
    pass
except:
    print("No files to remove")

stream = IdrStream(stream_type,idr_id, tmp_dir, final_data_dir, log='idr_stream.log')


# ## Initialize Aspera downloader

# In[4]:


# find the path in terminal using `ascli config ascp show`
aspera_path = pathlib.Path("/home/jenna/.aspera/ascli/sdk/ascp")
aspera_key_path = pathlib.Path("example_files/asperaweb_id_dsa.openssh")
screens_path = pathlib.Path("example_files/idr0013-screenA-plates.tsv")
save_dir = pathlib.Path("data/")

stream.init_downloader(aspera_path, aspera_key_path, screens_path)


# ## Initialize Fiji preprocessor

# In[5]:


fiji_path = pathlib.Path("/home/jenna/Desktop/test/Fiji.app")
stream.init_preprocessor(fiji_path)


# ## Initialize CellProfiler metadata compiler

# In[6]:


data_to_process_tsv = pathlib.Path("example_files/data_to_process.tsv")
stream.init_cp_metadata(data_to_process_tsv)


# ## Copy and create CellProfiler files/folders

# In[7]:


cppipe_path = pathlib.Path("example_files/CP_files/mitocheck_idr_cp.cppipe")
stream.copy_CP_files(cppipe_path)


# ## Run idrstream batches

# In[ ]:


stream.run_stream(stream_type, data_to_process, batch_size=3, start_batch=0)

