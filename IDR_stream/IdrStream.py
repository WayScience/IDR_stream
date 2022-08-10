import pandas as pd
import pathlib
import os
import shutil
import math

import download
import preprocess


class IdrStream:
    def __init__(self, tmp_dir: pathlib.Path, final_data_dir: pathlib.path):
        self.tmp_dir = tmp_dir
        self.DP_project_path = pathlib.Path(f"{tmp_dir}/DP_project/")
        self.DP_project_path.mkdir(parents=True, exist_ok=True)
        self.final_data_dir = final_data_dir
        self.final_data_dir.mkdir(parents=True, exist_ok=True)

    def copy_DP_files(self, config_path: pathlib.Path, checkpoint_path: pathlib.Path):
        # copy config file to DP project
        config_save_path = pathlib.Path(
            f"{self.DP_project_path}/inputs/config/{config_path.name}"
        )
        config_save_path.parents[0].mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config_path, config_save_path)

        # copy checkpoint file to DP project
        checkpoint_save_path = pathlib.Path(
            f"{self.DP_project_path}/outputs/results/checkpoint/{checkpoint_path.name}"
        )
        checkpoint_save_path.parents[0].mkdir(parents=True)
        shutil.copyfile(checkpoint_path, checkpoint_save_path)

    def init_downloader(
        self,
        aspera_path: pathlib.Path,
        aspera_key_path: pathlib.Path,
        screens_path: pathlib.Path,
        idr_id: str,
    ):
        self.downloader = download.AsperaDownloader(
            aspera_path, aspera_key_path, screens_path, idr_id
        )

    def init_preprocessor(self, fiji_path: pathlib.Path):
        self.preprocessor = preprocess.BasicpyPreprocessor(fiji_path)
        
    def run_stream(self, data_to_process: pd.DataFrame, batch_size : int = 10, start_batch: int = 0):
        batches = math.ceil(data_to_process.shape[0]/batch_size)
        
        for batch_num in range(batches):
            batch_metadata = data_to_process.iloc[0:batch_size]
            data_to_process = data_to_process.iloc[batch_size:]
            if batch_num < start_batch:
                continue
            print(f"Profiling batch {batch_num}")
            
            try:
                self.prepare_batch(batch_metadata) # download images, segment/illum correct data and place these data into the Deep Profiler project
                self.compile_DP_batch_index_csv(batch_metadata) # compile index csv for DeepProfiler project for the specific batch
                self.profile_batch() # profile batch with Deep Profiler
                features_path = pathlib.Path(f"data/batch_{batch_num}.csv.gz")
                self.preprocess_batch_features(features_path) # preprocess features for specific batch
                self.clear_batch() #delete image/segmentation data for batch
            except Exception as e:
                print(f"Execption for batch {batch_num}")
                print(e)