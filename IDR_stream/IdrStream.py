import pandas as pd
import pathlib
import os
import shutil
import math
import logging
from pycytominer.cyto_utils import DeepProfiler_processing

import sys
sys.path.append("/home/roshankern/Desktop/Github/IDR_stream/IDR_stream")
import download
import preprocess
import segment


class IdrStream:
    def __init__(
        self, idr_id: str, tmp_dir: pathlib.Path, final_data_dir: pathlib.Path, log = False,
    ):
        self.idr_id = idr_id
        self.tmp_dir = tmp_dir
        self.DP_project_path = pathlib.Path(f"{tmp_dir}/DP_project/")
        self.DP_project_path.mkdir(parents=True, exist_ok=True)
        self.final_data_dir = final_data_dir
        self.final_data_dir.mkdir(parents=True, exist_ok=True)
        
        # create logger for IDR stream
        self.logger = logging.getLogger('idr_stream')
        if log:
            # only print log messsages above warn level
            self.logger.setLevel(logging.INFO)
            # create file handler to save log messages from debug level
            file_handler = logging.FileHandler('idr_stream.log')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.info("IDR stream initialized")
        else:
            self.logger.setLevel(logging.WARN)

    def copy_DP_files(self, config_path: pathlib.Path, checkpoint_path: pathlib.Path):
        # copy config file to DP project
        config_save_path = pathlib.Path(
            f"{self.DP_project_path}/inputs/config/{config_path.name}"
        )
        config_save_path.parents[0].mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config_path, config_save_path)
        self.config_name = config_path.name

        # copy checkpoint file to DP project
        checkpoint_save_path = pathlib.Path(
            f"{self.DP_project_path}/outputs/results/checkpoint/{checkpoint_path.name}"
        )
        checkpoint_save_path.parents[0].mkdir(parents=True)
        shutil.copyfile(checkpoint_path, checkpoint_save_path)
        self.checkpoint_name = checkpoint_path.name
        
        self.logger.info("Copied Deep Profiler config and checkpoint files")

    def init_downloader(
        self,
        aspera_path: pathlib.Path,
        aspera_key_path: pathlib.Path,
        screens_path: pathlib.Path,
    ):
        self.downloader = download.AsperaDownloader(
            aspera_path, aspera_key_path, screens_path, self.idr_id
        )
        self.logger.info("Aspera downloader initialized")

    def init_preprocessor(self, fiji_path: pathlib.Path):
        self.preprocessor = preprocess.BasicpyPreprocessor(fiji_path)
        self.logger.info("Basicpy preprocessor initialized")

    def init_segmentor(self, model_specs: dict):
        self.segmentor = segment.CellPoseSegmentor(model_specs)
        self.logger.info("CellPose segmentor initialized")
        
        

    def prepare_batch(self, batch_metadata: pd.DataFrame):
        for index, row in batch_metadata.iterrows():
            plate = row["Plate"]
            well = row["Well"]
            well_num = row["Well Number"]
            # load frames saved as "x,y,z" and convert to [x,y,z]
            frame_nums = str(row["Frames"]).split(",")
            frame_nums = [int(frame_num) for frame_num in frame_nums]
            self.logger.info(f"Processing well {well} from {plate}, frames {frame_nums}")

            download_save_path = pathlib.Path(f"{self.tmp_dir}/downloads/{plate}/")
            well_movie_path = self.downloader.download_image(
                plate, well_num, download_save_path
            )
            self.logger.info(f"well_movie_path: {well_movie_path}")

            frames_save_path = pathlib.Path(
                f"{self.DP_project_path}/inputs/images/{plate}/"
            )
            self.preprocessor.save_corrected_frames(
                plate, well_num, well_movie_path, frames_save_path, frame_nums
            )
            self.logger.info("Saved corrected frames")

            objects_save_path = pathlib.Path(
                f"{self.DP_project_path}/inputs/locations/{plate}/"
            )
            self.segmentor.save_nuclei_locations(
                plate, well_num, frames_save_path, frame_nums, objects_save_path
            )
            self.logger.info("Saved nuclei locations")

    def compile_DP_batch_index_csv(self, batch_metadata: pd.DataFrame):

        index_csv_save_path = pathlib.Path(
            f"{self.DP_project_path}/inputs/metadata/index.csv"
        )
        index_csv_save_path.parents[0].mkdir(parents=True, exist_ok=True)

        index_csv_data = []

        # path to deepprofiler project images folder
        DP_images_path = pathlib.Path(f"{self.DP_project_path}/inputs/images/")

        for index, row in batch_metadata.iterrows():
            plate = row["Plate"]
            well_num = row["Well Number"]
            # load frames saved as "x,y,z" and convert to [x,y,z]
            frame_nums = str(row["Frames"]).split(",")
            frame_nums = [int(frame_num) for frame_num in frame_nums]
            gene = row["Original Gene Target"]

            for frame_num in frame_nums:
                image_file_path = pathlib.Path(
                    f"{DP_images_path}/{plate}/{plate}_{well_num}_{frame_num}.tif"
                )
                image_data = {
                    "Metadata_Plate": plate,
                    "Metadata_Well": f"{well_num}_{frame_num}",
                    "Metadata_Site": 1,
                    "Plate_Map_Name": f"{plate}_{well_num}",
                    "DNA": image_file_path.relative_to(DP_images_path),
                    "Gene": gene,
                    "Gene_Replicate": 1,
                }
                index_csv_data.append(image_data)
        index_csv_data = pd.DataFrame(index_csv_data)
        index_csv_data.to_csv(index_csv_save_path, index=False)
        self.logger.info(f"Compiled index.csv file to {index_csv_save_path}")

    def profile_batch(self):
        command = f"python3 -m deepprofiler --gpu 0 --root {self.DP_project_path} --config {self.config_name} profile"
        os.system(command)
        self.logger.info("Deep Profiler run done")
        
    def clear_batch(self):
        images = pathlib.Path(f"{self.DP_project_path}/inputs/images/")
        locations = pathlib.Path(f"{self.DP_project_path}/inputs/locations/")
        metadata = pathlib.Path(f"{self.DP_project_path}/inputs/metadata/")
        features = pathlib.Path(f"{self.DP_project_path}/outputs/results/features/")
        paths_to_remove = [images, locations, metadata, features]
        
        for path in paths_to_remove:
            shutil.rmtree(path)
        
        self.logger.info("Temporary batch files cleared")

    def compile_batch_features(self, output_path: pathlib.Path):
        index_file = pathlib.Path(f"{self.DP_project_path}/inputs/metadata/index.csv")
        profile_dir = pathlib.Path(f"{self.DP_project_path}/outputs/results/features")
        deep_data = DeepProfiler_processing.DeepProfilerData(
            index_file, profile_dir, filename_delimiter="/"
        )
        deep_single_cell = DeepProfiler_processing.SingleCellDeepProfiler(deep_data)
        deep_single_cell.get_single_cells(output=True).to_csv(
            output_path, compression={"method": "gzip", "compresslevel": 1, "mtime": 1}
        )
        
        self.logger.info("Batch features compiled with PyCytominer")

    def run_stream(
        self, data_to_process: pd.DataFrame, batch_size: int = 10, start_batch: int = 0
    ):
        batches = math.ceil(data_to_process.shape[0] / batch_size)
        self.logger.info(f"Running IDR stream with: \nbatch_size {batch_size} \nstart_batch {start_batch} \nbatches {batches}")

        for batch_num in range(batches):
            batch_metadata = data_to_process.iloc[0:batch_size]
            data_to_process = data_to_process.iloc[batch_size:]
            if batch_num < start_batch:
                continue

            self.logger.info(f"Profiling batch {batch_num}")
            self.prepare_batch(batch_metadata)
            self.compile_DP_batch_index_csv(
                batch_metadata
            )  # compile index csv for DeepProfiler project for the specific batch
            self.profile_batch()  # profile batch with Deep Profiler
            features_path = pathlib.Path(
                f"{self.final_data_dir}/batch_{batch_num}.csv.gz"
            )
            self.compile_batch_features(features_path)
            self.clear_batch() #delete image/segmentation data for batch
