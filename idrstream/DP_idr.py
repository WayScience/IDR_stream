"""
This file holds the DeepProfilerRun class with all of the functions to use DeepProfiler for feature extraction on a batch. These functions
include copying the files into a DP project, creating an index.csv file, profile the batch with DeepProfiler, and compile the
outputted features.
"""
import time
import pandas as pd
import pathlib
import os
import shutil
import math
import logging
from pycytominer.cyto_utils import DeepProfiler_processing

import idrstream.download as download
import idrstream.preprocess as preprocess
import idrstream.segment as segment


class DeepProfilerRun:
    """
    This class holds all the functions needed to perform a DeepProfiler run

    Attributes
    ----------
    idr_id : str
        idr accession number for the desired IDR study
    tmp_dir : pathlib.Path
        path to directory to use for intermediate files
    DP_project_path : pathlib.Path
        path to directory of DeepProfiler project
    final_data_dir : pathlib.Path
        path to directory of final output data
    logger : str
        path of log file
    config_name : str
        name of config file used for DP project
    downloader : AsperaDownloader
        AsperaDownloader used to download IDR data
    preprocessor : BasicpyPreprocessor
        BasicpyPreprocessor used to preprocess IDR data
    segmentor : CellPoseSegmentor
        CellPoseSegmentor used to segment IDR data

    Methods
    -------
    copy_DP_files(cppipe_path, config_path, checkpoint_path)
        copy project files into temporary directory of the DP project
    init_downloader(aspera_path, aspera_key_path, screens_path)
        initialize aspera downloader
    init_preprocessor(fiji_path)
        initialize basicpy preprocessor
    init_segmentor(model_specs)
        initialize cellpose segmentor
    prepare_batch(batch_metadata)
        download data from a batch image and run further processes
        image and location data are saved in DP project folder
    compile_batch_index_csv(batch_metadata)
        create index.csv file for batch, index.csv file is used by DeepProfiler to profile a batch
    profile_batch_with_DP()
        profile batch with DeepProfiler
    compile_batch_DP_features(output_path)
        compile single cell features into one dataframe and save as compressed csv to final output folder
    clear_batch()
        remove all intermediate files that are unecessary for next batch to run
    run_dp_stream(data_to_process, batch_size, start_batch)
        extract features from IDR study given metadata of images to extract features from DeepProfiler
    """

    def __init__(
        self,
        idr_id: str,
        tmp_dir: pathlib.Path,
        final_data_dir: pathlib.Path,
        log="",
    ):
        """
        __init__ function for the DeepProfilerRun class

        Parameters
        ----------
        idr_id : str
            idr accession number for the desired IDR study
        tmp_dir : pathlib.Path
            path to directory to use for intermediate files
        final_data_dir : pathlib.Path
            path to directory of final output data
        log : str
            Path to log file or "" if no logging is desired, by default ""
        """
        self.idr_id = idr_id
        self.tmp_dir = tmp_dir

        self.DP_project_path = pathlib.Path(f"{tmp_dir}/DP_project/")
        self.DP_project_path.mkdir(parents=True, exist_ok=True)

        self.final_data_dir = final_data_dir
        self.final_data_dir.mkdir(parents=True, exist_ok=True)

        # create logger for IDR stream
        self.logger = logging.getLogger("idr_stream")
        if log == "":
            self.logger.setLevel(logging.WARN)
        else:
            # log messages at info level
            self.logger.setLevel(logging.INFO)
            # create file handler to save log messages
            file_handler = logging.FileHandler(log)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.info("IDR stream initialized")

    def copy_DP_files(
        self,
        config_path: pathlib.Path,
        checkpoint_path: pathlib.Path,
    ):
        """
        copy project files into temporary directory of the DeepProfiler project

        Parameters
        ----------
        config_path : pathlib.Path
            path to config file to copy
        checkpoint_path : pathlib.Path
            path to checkpoint to copy
        """
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
        """
        initialize aspera downloader

        Parameters
        ----------
        aspera_path : pathlib.Path
            path to aspera sdk
        aspera_key_path : pathlib.Path
            path to aspera ssh key
        screens_path : pathlib.Path
            path to screens file used to locate IDR data
        """
        self.downloader = download.AsperaDownloader(
            aspera_path, aspera_key_path, screens_path, self.idr_id
        )
        self.logger.info("Aspera downloader initialized")

    def init_preprocessor(self, fiji_path: pathlib.Path):
        """
        initialize basicpy preprocessor

        Parameters
        ----------
        fiji_path : pathlib.Path
            path to Fiji.app folder
        """
        self.preprocessor = preprocess.BasicpyPreprocessor(fiji_path)
        self.logger.info("Basicpy preprocessor initialized")

    def init_segmentor(self, model_specs: dict):
        """
        initialize cellpose segmentor

        Parameters
        ----------
        model_specs : dict
            information for how to segment image data

            example: model_specs = {
                "model_type": "cyto",
                "channels": [0, 0],
                "diameter": 0,
                "flow_threshold": 0.8,
                "cellprob_threshold": 0,
                "remove_edge_masks": True,
            }
        """
        self.segmentor = segment.CellPoseSegmentor(model_specs)
        self.logger.info("CellPose segmentor initialized")

    def prepare_batch(self, batch_metadata: pd.DataFrame):
        """
        download data from a batch image and the images and location data are saved in DP project folder

        Parameters
        ----------
        batch_metadata : pd.DataFrame
            metadata of images to extract features from
        """
        # iterate through image metadatas and download, preprocess, and segment each frame into the batch project
        for _, row in batch_metadata.iterrows():
            plate = row["Plate"]
            well = row["Well"]
            well_num = row["Well Number"]
            # load frames saved as "x,y,z" and convert to [x,y,z]
            frame_nums = str(row["Frames"]).split(",")
            frame_nums = [int(frame_num) for frame_num in frame_nums]
            self.logger.info(
                f"Processing well {well} from {plate}, frames {frame_nums}"
            )

            download_save_path = pathlib.Path(f"{self.tmp_dir}/downloads/{plate}/")
            well_movie_path = self.downloader.download_image(
                plate, well_num, download_save_path
            )
            self.logger.info(f"Movie downloaded to {well_movie_path}")

            # Aspera downloader called in self.downloader.download_image can take extra time to finalize download depending on network speed.
            # This gives time for movie to fully save before trying to open it
            time.sleep(0.3)

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
                plate,
                well_num,
                frames_save_path,
                frame_nums,
                objects_save_path,
                self.extra_metadata,
            )
            self.logger.info("Saved nuclei locations")

    def compile_batch_index_csv(self, batch_metadata: pd.DataFrame):
        """
        create index.csv file for batch, index.csv file is used by DeepProfiler to profile a batch

        Parameters
        ----------
        batch_metadata : pd.DataFrame
            metadata of images to extract features from
        """

        index_csv_save_path = pathlib.Path(
            f"{self.DP_project_path}/inputs/metadata/index.csv"
        )
        index_csv_save_path.parents[0].mkdir(parents=True, exist_ok=True)

        index_csv_data = []

        # path to deepprofiler project images folder
        DP_images_path = pathlib.Path(f"{self.DP_project_path}/inputs/images/")

        # iterate through images and append data relevant to DP index file
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

    def profile_batch_with_DP(self):
        """
        profile batch with DeepProfiler
        """
        command = f"python3 -m deepprofiler --gpu 0 --root {self.DP_project_path} --config {self.config_name} profile"
        os.system(command)

        self.logger.info("Deep Profiler run done")

    def compile_batch_DP_features(self, output_path: pathlib.Path):
        """
        compile single cell features from DeepProfiler into one dataframe and save as compressed csv to final output folder

        Parameters
        ----------
        output_path : pathlib.Path
            path of final data folder
        """
        index_file = pathlib.Path(f"{self.DP_project_path}/inputs/metadata/index.csv")
        profile_dir = pathlib.Path(f"{self.DP_project_path}/outputs/results/features")
        deep_data = DeepProfiler_processing.DeepProfilerData(
            index_file, profile_dir, filename_delimiter="/"
        )
        # create and save single cell df with feature data and metadata
        deep_single_cell = DeepProfiler_processing.SingleCellDeepProfiler(deep_data)
        deep_single_cell_df = deep_single_cell.get_single_cells(output=True)
        self.logger.info("Batch features compiled with PyCytominer")
        if "object_outlines" in self.extra_metadata:
            deep_single_cell_df = self.add_batch_object_outlines(deep_single_cell_df)
        deep_single_cell_df.to_csv(
            output_path, compression={"method": "gzip", "compresslevel": 1, "mtime": 1}
        )

        self.logger.info(f"Saved compiled batch features to {output_path}")

    def clear_batch(self):
        """
        remove all intermediate files that are unnecessary for next batch to run
        """
        images = pathlib.Path(f"{self.DP_project_path}/inputs/images/")
        locations = pathlib.Path(f"{self.DP_project_path}/inputs/locations/")
        metadata = pathlib.Path(f"{self.DP_project_path}/inputs/metadata/")
        features = pathlib.Path(f"{self.DP_project_path}/outputs/results/features/")
        paths_to_remove = [images, locations, metadata, features]

        for path in paths_to_remove:
            shutil.rmtree(path)

        self.logger.info("Temporary batch files cleared")
        
    def add_batch_object_outlines(
        self, batch_single_cell_df: pd.DataFrame, object_metadata_channel = "Metadata_DNA"
    ) -> pd.DataFrame:
        """
        add object outlines to single cell data for current batch
        necessary because we compile original single cell df with pycytominer
        but pycytominer cannot append additional metadata (like object outlines)
        Parameters
        ----------
        batch_single_cell_df : pd.DataFrame
            original batch single cell df
        object_metadata_channel : str
            DeepProfiler channel name (from index.csv) to get object outlines from
        Returns
        -------
        pd.DataFrame
            new batch single cell df with object outlines appended
        """

        locations_save_path = pathlib.Path(f"{self.DP_project_path}/inputs/locations/")
        new_batch_single_cell_df = []

        # iterate over location files in order of plate map data in batch_single_cell_df
        for image_path in batch_single_cell_df[object_metadata_channel].unique():
            # split single cell dataframe into image dataframes to find object outlines for that image
            image_single_cell_df = batch_single_cell_df.loc[
                batch_single_cell_df[object_metadata_channel] == image_path
            ].reset_index(drop=True)
            image_plate = image_single_cell_df["Metadata_Plate"].unique()[0]
            image_well = image_single_cell_df["Metadata_Well"].unique()[0]
            image_site = image_single_cell_df["Metadata_Site"].unique()[0]

            # load object outlines for a particular image
            image_locations_path = pathlib.Path(
                f"{locations_save_path}/{image_plate}/{image_well}-{image_site}-Nuclei.csv"
            )
            image_outline_data = pd.read_csv(image_locations_path)[
                "object_outline"
            ].reset_index(drop=True)
            # insert object outlines to the single cell df
            image_single_cell_df.insert(
                loc=0, column="Object_Outline", value=image_outline_data
            )

            # add to full batch single cell df
            new_batch_single_cell_df.append(image_single_cell_df)

        # compile and return new batch single cell df
        new_batch_single_cell_df = pd.concat(new_batch_single_cell_df).reset_index(
            drop=True
        )
        self.logger.info("Object outlines added for batch")
        return new_batch_single_cell_df

    def run_dp_stream(
        self,
        data_to_process: pd.DataFrame,
        batch_size: int = 10,
        start_batch: int = 0,
        batch_nums="all",
        extra_metadata=[],
    ):
        """
        extract features from IDR study given metadata of images to extract features from DeepProfiler

        Parameters
        ----------
        data_to_process : pd.DataFrame
            Metadata df with metadata of images to extract features from
        batch_size : int, optional
            number of images to process in one batch, by default 10
        start_batch : int, optional
            batch to start feature extraction from, by default 0
        batch_nums : str, list, optional
            list of batch numbers to extract features from, by default "all"
        extra_metadata : str, list, optional
            list of extra metadata to include in final dataframe outputs (object_outlines, object_boxes, etc), by default [])
        """
        start_time = time.time()
        
        batches = math.ceil(data_to_process.shape[0] / batch_size)
        self.logger.info(
            f"Running IDR stream with: \nbatch_size {batch_size} \nstart_batch {start_batch} \nbatches {batches} \nbatch nums {batch_nums} \nextra metadata {extra_metadata}"
        )
        self.extra_metadata = extra_metadata
        # prepare, profile, compile, and delete intermediate files for each batch
        for batch_num in range(batches):
            batch_metadata = data_to_process.iloc[0:batch_size]
            data_to_process = data_to_process.iloc[batch_size:]

            # skip batches before start batch and those not in batch nums
            if batch_num < start_batch:
                continue
            if batch_nums != "all":
                if batch_num not in batch_nums:
                    continue

            self.logger.info(f"Profiling batch {batch_num} with DeepProfiler")
            try:
                self.prepare_batch(
                    batch_metadata
                )  # put image and location data in DP-required locations
                self.compile_batch_index_csv(
                    batch_metadata
                )  # compile index csv for DeepProfiler project for the specific batch
                self.profile_batch_with_DP()  # profile batch with Deep Profiler
                features_path = pathlib.Path(
                    f"{self.final_data_dir}/batch_{batch_num}.csv.gz"
                )
                # compile and save features with PyCytominer
                self.compile_batch_DP_features(features_path)  
                # delete image/segmentation data for batch
                self.clear_batch()  
            except Exception as e:
                self.logger.error(f"Error while profiling batch {batch_num}: {e}")

        end_time = time.time()
        total_time = end_time-start_time
        self.logger.info(f"Stream run done in {total_time} seconds!")
