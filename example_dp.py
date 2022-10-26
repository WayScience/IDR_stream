#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import pandas as pd
import shutil
import logging

from idrstream.idr_stream import IdrStream


# In[2]:


data_to_process = pd.read_csv("example_files/data_to_process.tsv", sep="\t", index_col=0)
# remove A1 wells because of irregular illumination
data_to_process = data_to_process[data_to_process["Well"] != "A1"]
data_to_process = data_to_process.reset_index(drop=True)
data_to_process


# In[3]:


stream_type = "DP"
idr_id = "idr0013"
tmp_dir = pathlib.Path("tmp/")
final_data_dir = pathlib.Path("mitocheck_control_features/")
try:
    shutil.rmtree(tmp_dir)
    # shutil.rmtree(final_data_dir)
    pass
except:
    print("No files to remove")

stream = IdrStream(stream_type, idr_id, tmp_dir, final_data_dir, log='idr_stream.log')


# In[4]:


aspera_path = pathlib.Path("/home/roshankern/.aspera/ascli/sdk/ascp")
aspera_key_path = pathlib.Path("example_files/asperaweb_id_dsa.openssh")
screens_path = pathlib.Path("example_files/idr0013-screenA-plates.tsv")
save_dir = pathlib.Path("data/")

stream.init_downloader(aspera_path, aspera_key_path, screens_path)


# In[5]:


fiji_path = pathlib.Path("/home/roshankern/Desktop/Github/IDR_stream_prototype/preprocess_data/Fiji.app")
stream.init_preprocessor(fiji_path)


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


# In[ ]:


stream.run_stream(stream_type, data_to_process, batch_size=3, start_batch=0, batch_nums=[0])

