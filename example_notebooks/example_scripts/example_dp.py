#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import pandas as pd
import shutil
import logging

import sys
sys.path.append("../")
from idrstream.DP_idr import DeepProfilerRun


# In[2]:


data_to_process = pd.read_csv("example_files/data_to_process.tsv", sep="\t", index_col=0)
data_to_process


# In[3]:


idr_id = "idr0013"
tmp_dir = pathlib.Path("../tmp/")
final_data_dir = pathlib.Path("../example_extracted_features/DP_features/")
try:
    shutil.rmtree(tmp_dir)
    # shutil.rmtree(final_data_dir)
except:
    print("No files to remove")

stream = DeepProfilerRun(idr_id, tmp_dir, final_data_dir, log='example_logs/dp_idrstream.log')


# In[4]:


# path to users home dir
home_dir_path = pathlib.Path.home()

aspera_path = pathlib.Path(f"{home_dir_path}/.aspera/ascli/sdk/ascp")
aspera_key_path = pathlib.Path("example_files/asperaweb_id_dsa.openssh")
screens_path = pathlib.Path("example_files/idr0013-screenA-plates.tsv")
idr_index_name = "idr0013-neumann-mitocheck"

stream.init_downloader(aspera_path, aspera_key_path, screens_path, idr_index_name)


# In[5]:


fiji_path = pathlib.Path(f"{home_dir_path}/Desktop/Fiji.app")
perform_illumination_correction = True
stream.init_preprocessor(fiji_path, perform_illumination_correction)


# In[6]:


nuclei_model_specs = {
            "model_type": "cyto",
            "channels": [0, 0],
            "diameter": 0,
            "flow_threshold": 0.8,
            "cellprob_threshold": 0,
            "remove_edge_masks": True,
        }
stream.init_segmentor(nuclei_model_specs)


# In[7]:


config_path = pathlib.Path("example_files/DP_files/mitocheck_profiling_config.json")
checkpoint_path = pathlib.Path("example_files/DP_files/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment.h5")
stream.copy_DP_files(config_path, checkpoint_path)


# In[8]:


stream.run_dp_stream(data_to_process, batch_size=3, start_batch=0, batch_nums=[0,1,2])

