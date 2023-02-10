"""
This file holds the CellProfilerRun class with all of the functions to use CellProfiler for segmentation and feature extraction on a batch. These functions
include converting the data_to_process.tsv file in a csv for CellProfiler to use as metadata, profile the batch with CellProfiler, and compile the
Nuclei.csv outputted features.
"""
import pandas as pd
import os
import pathlib
import time
import shutil
import math
import logging

import idrstream.download as download
import idrstream.preprocess as preprocess


class CellProfilerRun:
    """
    This class holds all the functions needed to perform a CellProfiler run (segmentation and feature
    extraction)

    Attributes
    ----------
    pipeline_path : pathlib.Path
        path to CellProfiler pipeline to perform segmentation and feature extraction
    plugins_directory : pathlib.Path
        path to CellProfiler directory with Cellpose plugin
    idr_id : str
        idr accession number for the desired IDR study
    tmp_dir : pathlib.Path
        path to directory to use for intermediate files
    final_data_dir : pathlib.Path
        path to directory of final output data
    logger : str
        path of log file
    CP_output_path : pathlib.Path
        path to where the .csv files outputted from each CellProfiler batch run will go
    CP_images_path : pathlib.Path
        path to the images for each batch
    downloader : AsperaDownloader
        AsperaDownloader used to download IDR data
    preprocessor : BasicpyPreprocessor
        BasicpyPreprocessor used to preprocess IDR data

    Methods
    -------
    copy_CP_files(metadata_path)
        copy project files into temporary directory of the CP project
    init_downloader(aspera_path, aspera_key_path, screens_path)
        initialize aspera downloader
    init_preprocessor(fiji_path)
        initialize basicpy preprocessor
    convert_tsv_to_csv(data_to_process_tsv, metadata_save_path)
        convert `data_to_process.tsv` file into a .csv for CellProfiler to read in when processing the images
    prepare_batch(batch_metadata)
        download data from a batch image and run further processes
        image data is saved in the CP project folder
    profile_batch_with_CP()
        profile batch with CellProfiler (runs segmentation and feature extraction)
    compile_batch_CP_features(output_path)
        compile single cell features from CellProfiler into one dataframe (to look like the output from DeepProfiler) and save as compressed csv
        to final output folder
    clear_batch()
        remove all intermediate files that are unnecessary for next batch to run
    run_cp_stream(data_to_process, batch_size, start_batch)
        extract features from IDR study given metadata of images to segment and extract features from CellProfiler
    """

    def __init__(
        self,
        pipeline_path: pathlib.Path,
        plugins_directory: pathlib.Path,
        idr_id: str,
        tmp_dir: pathlib.Path,
        final_data_dir: pathlib.Path,
        log="",
    ):
        """
        __init__ function for the CellProfilerRun class

        Parameters
        ----------
        pipeline_path : pathlib.Path
            path to CellProfiler pipeline to perform segmentation and feature extraction
        plugins_directory : pathlib.Path
            path to CellProfiler directory with Cellpose plugin
        idr_id : str
            idr accession number for the desired IDR study
        tmp_dir : pathlib.Path
            path to directory to use for intermediate files
        final_data_dir : pathlib.Path
            path to directory of final output data
        log : str
            Path to log file or "" if no logging is desired, by default ""
        """
        self.pipeline_path = pipeline_path
        self.plugins_directory = plugins_directory

        self.idr_id = idr_id
        self.tmp_dir = tmp_dir

        self.CP_project_path = pathlib.Path(f"{tmp_dir}/CP_project/")
        self.CP_project_path.mkdir(parents=True, exist_ok=True)

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

    def copy_CP_files(
        self,
        metadata_path: pathlib.Path,
    ):
        """
        copy project files into temporary directory of the CellProfiler project

        Parameters
        ----------
        metadata_path : pathlib.Path
            path to "data_to_process.tsv" to convert into a ".csv" file for CellProfiler to use as metadata
        """
        # make directory for the input of cellprofiler and copy metadata file into images folder
        metadata_save_path = pathlib.Path(
            f"{self.CP_project_path}/inputs/images/{metadata_path.name}"
        )
        metadata_save_path.parents[0].mkdir(parents=True, exist_ok=True)
        shutil.copyfile(metadata_path, metadata_save_path)

        # make directory for the output of cellprofiler
        output_save_path = pathlib.Path(f"{self.CP_project_path}/outputs/features/")
        output_save_path.mkdir(parents=True, exist_ok=True)
        self.CP_output_path = output_save_path

        self.logger.info("Copied metadata file for CellProfiler run to use")

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

    def convert_tsv_to_csv(
        self, data_to_process_tsv: pathlib.Path, metadata_save_path: pathlib.Path
    ):
        """
        Convert `data_to_process.tsv` file into a .csv for CellProfiler to read in when processing the images

        Parameters
        ----------
        data_to_process_tsv : str
            string path to the `data_to_process.tsv` file
        """
        # read in metadata tsv file
        data_to_process_csv = pd.read_csv(data_to_process_tsv, sep="\t")

        # converting tsv file into csv
        data_to_process_csv.to_csv(metadata_save_path, index=False)

        self.logger.info("CellProfiler metadata has been converted and saved")

    def prepare_batch(self, batch_metadata: pd.DataFrame):
        """
        download data from a batch image and image data is saved in the CP project folder

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
            self.logger.info(f"Movie downloaded to: {well_movie_path}")

            # give time for movie to fully save before trying to open it
            # otherwise ImageJ tries to open to movie before it has been completely saved and it errors out
            time.sleep(0.3)

            frames_save_path = pathlib.Path(
                f"{self.CP_project_path}/inputs/images/{plate}/"
            )
            self.CP_images_path = pathlib.Path(f"{self.CP_project_path}/inputs/images/")
            self.preprocessor.save_corrected_frames(
                plate, well_num, well_movie_path, frames_save_path, frame_nums
            )
            self.logger.info("Saved corrected frames")

    def profile_batch_with_CP(self):
        """
        profile batch with CellProfiler (runs segmentation and feature extraction)
        """
        # need to specify plugin directory for CellProfiler to find the Cellpose plugin
        command = f"cellprofiler -c -r -p {self.pipeline_path} -o {self.CP_output_path} -i {self.CP_images_path} --plugins-directory {self.plugins_directory}"
        os.system(command)
        self.logger.info("CellProfiler run done")

    def compile_batch_CP_features(self, output_path: pathlib.Path):
        """
        compile single cell features from CellProfiler into one dataframe (to look like the output from DeepProfiler) and save as compressed csv
        to final output folder

        Parameters
        ----------
        output_path (pathlib.Path):
            path of final data folder
        """
        # load in the "Nuclei.csv" file that is created from the batch
        nuclei_table = pathlib.Path(
            f"{self.tmp_dir}/CP_project/outputs/features/Nuclei.csv"
        )
        cp_output = pd.read_csv(nuclei_table, dtype=object)

        # change 'Metadata_Well' column data format
        cp_output = cp_output.drop("Metadata_Well", axis=1)
        cp_output["Metadata_Well"] = (
            cp_output["Metadata_Well_Number"] + "_" + cp_output["Metadata_Frames"]
        )

        # list of all unnecessary columns in the outputted .csv file from CellProfiler
        columns_to_drop = [
            "Metadata_FileLocation",
            "Metadata_Frame",
            "Metadata_Series",
            "Metadata_Control Type",
            "ImageNumber",
            "ObjectNumber",
            "Metadata_Frames",
            "Metadata_Well_Number",
            "Metadata_Unnamed: 0",
        ]

        # remove unnecessary metadata columns
        cp_output = cp_output.drop(columns_to_drop, axis=1)

        # change the name of 'Metadata_Orginal Gene Replicate' to 'Metadata_Gene'
        cp_output = cp_output.rename(
            columns={"Metadata_Original Gene Target": "Metadata_Gene"}
        )

        # list of the metadata in a specified order
        metadata = [
            "Location_Center_X",
            "Location_Center_Y",
            "Metadata_Plate",
            "Metadata_Well",
            "Metadata_Site",
            "Metadata_Plate_Map_Name",
            "Metadata_DNA",
            "Metadata_Gene",
            "Metadata_Gene_Replicate",
        ]

        # change the order of the metadata within the file
        new_metadata_order = metadata + (cp_output.columns.drop(metadata).tolist())
        cp_features_compiled = cp_output[new_metadata_order]

        # save the compiled features into a compressed .csv file
        cp_features_compiled.to_csv(
            output_path, compression={"method": "gzip", "compresslevel": 1, "mtime": 1}
        )
        self.logger.info("Batch features compiled for CellProfiler features")

    def clear_batch(self):
        """
        remove all intermediate files that are unecessary for next batch to run
        """
        # remove all of the plate folders (with images) from images folder
        images = pathlib.Path(f"{self.CP_project_path}/inputs/images/")
        for item in images.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
        # remove features folders for next batch
        features = pathlib.Path(f"{self.CP_project_path}/outputs/features/")
        shutil.rmtree(features)

        self.logger.info("Temporary batch files cleared")

    def run_cp_stream(
        self,
        data_to_process: pd.DataFrame,
        batch_size: int = 10,
        start_batch: int = 0,
        batch_nums: str = "all",
        extra_metadata: list = [],
    ):
        """
        extract features from IDR study given metadata of images to extract features from using a specific method

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
            list of extra metadata to include in final dataframe outputs (object_outlines, object_boxes, etc), by default []
        """
        start_time = time.time()

        batches = math.ceil(data_to_process.shape[0] / batch_size)
        self.logger.info(
            f"Running IDR stream with: \nbatch_size {batch_size} \nstart_batch {start_batch} \nbatches {batches}"
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

            self.logger.info(f"Profiling batch {batch_num} with CellProfiler")
            try:
                self.prepare_batch(
                    batch_metadata
                )  # put image and location data in CP-required locations
                self.profile_batch_with_CP()  # profile batch with CellProfiler
                output_path = pathlib.Path(
                    f"{self.final_data_dir}/batch_{batch_num}.csv.gz"
                )
                self.compile_batch_CP_features(
                    output_path
                )  # compile and save features with PyCytominer
                self.clear_batch()  # delete image/segmentation data for batch
            except Exception as e:
                self.logger.info(f"Error while profiling batch {batch_num}:")
                self.logger.error(e)

        end_time = time.time()
        total_time = end_time - start_time
        self.logger.info(f"Stream run done in {total_time} seconds!")
