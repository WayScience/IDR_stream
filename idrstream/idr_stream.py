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
import idrstream.cellprofiler_metadata as cellprofiler_metadata


class IdrStream:
    """
    This class holds all functions needs to stream feature extraction from IDR image data

    Attributes
    ----------
    idr_id : str
        idr accession number for the desired IDR study
    tmp_dir : pathlib.Path
        path to directory to use for intermediate files
    CP_project_path:
        path to directory of CellProfiler project
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
    copy_project_files(stream_type, cppipe_path, config_path, checkpoint_path)
        copy project files into temporary directory of the project (DP or CP)
    init_downloader(aspera_path, aspera_key_path, screens_path)
        initialize aspera downloader
    init_preprocessor(fiji_path)
        initialize basicpy preprocessor
    init_segmentor(model_specs)
        initialize cellpose segmentor
    init_cellprofiler_metadata(data_to_process_tsv)
        initialize CellProfiler metadata creator
    prepare_batch(stream_type, batch_metadata)
        download data from a batch image and run further processes based on the stream_type
        image and location data are saved in DP project folder
        image data is saved in the CP project folder
    compile_DP_batch_index_csv(batch_metadata)
        create index.csv file for batch, index.csv file is used by DeepProfiler to profile a batch
    profile_batch_with_CP()
        profile batch with CellProfiler
    profile_batch_with_DP()
        profile batch with DeepProfiler
    clear_batch(stream_type)
        remove all intermediate files that are unecessary for next batch to run
    compile_batch_DP_features(output_path)
        compile single cell features into one dataframe and save as compressed csv to final output folder
    compile_batch_CP_features(output_path)
        compile single cell features from CellProfiler into one dataframe (to look like the output from DeepProfiler) and save as compressed csv
        to final output folder
    run_stream(stream_type, data_to_process, batch_size, start_batch)
        extract features from IDR study given metadata of images to extract features from using a specific method (either DP or CP)
    """

    def __init__(
        self,
        stream_type: str,
        idr_id: str,
        tmp_dir: pathlib.Path,
        final_data_dir: pathlib.Path,
        log="",
    ):
        """
        __init__ function for IdrStream class

        Parameters
        ----------
        stream_type: str
            segmention/feature extraction method (either DP or CP)
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

        if stream_type == "DP":
            # DP project will be in tmp dir
            self.DP_project_path = pathlib.Path(f"{tmp_dir}/DP_project/")
            self.DP_project_path.mkdir(parents=True, exist_ok=True)

        if stream_type == "CP":
            # CP project will be in tmp dir
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

    def copy_CP_files(
        self,
        metadata_path : pathlib.Path,
    ):
        """
        copy project files into temporary directory of the CellProfiler project

        """
        # make directory for the input of cellprofiler and copy metdata file into images folder
        metadata_save_path = pathlib.Path(f"{self.CP_project_path}/inputs/images/{metadata_path.name}")
        metadata_save_path.parents[0].mkdir(parents=True, exist_ok=True)
        shutil.copyfile(metadata_path, metadata_save_path)

        # make directory for the output of cellprofiler
        output_save_path = pathlib.Path(f"{self.CP_project_path}/outputs/features/")
        output_save_path.mkdir(parents=True, exist_ok=True)
        self.CP_output_path = output_save_path

        self.logger.info("Copied CellProfiler cppipe file")

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

    def init_cp_metadata(self, data_to_process_tsv: pathlib.Path):
        """initialize cellprofiler metadata compiler

        Parameters
        ----------
        data_to_process : pd.DataFrame
            metadata for CellProfiler to process
        """
        self.cp_metadata = cellprofiler_metadata.convert_tsv_to_csv(data_to_process_tsv)
        self.logger.info("CellProfiler metadata has been converted and saved")

    def prepare_batch(self, stream_type: str, batch_metadata: pd.DataFrame):
        """
        download data from a batch image and run further processes based on the stream_type
        image and location data are saved in DP project folder
        image data is saved in the CP project folder

        Parameters
        ----------
        stream_type: str
            segmention/feature extraction method (either DP or CP)
        batch_metadata : pd.DataFrame
            metadata of images to extract features from
        """
        # iterate through image metadatas and download, preprocess, and segment each frame into the batch project
        for index, row in batch_metadata.iterrows():
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
            self.logger.info(f"well_movie_path: {well_movie_path}")

            # give time for movie to fully save before trying to open it
            # otherwise ImageJ tries to open to movie before it has been completely saved and it errors out
            time.sleep(0.3)

            if stream_type == "CP":
                frames_save_path = pathlib.Path(
                    f"{self.CP_project_path}/inputs/images/{plate}/"
                )
                self.CP_images_path = pathlib.Path(
                    f"{self.CP_project_path}/inputs/images/"
                )

                self.preprocessor.save_corrected_frames(
                    plate, well_num, well_movie_path, frames_save_path, frame_nums
                )
                self.logger.info("Saved corrected frames")

            if stream_type == "DP":
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

    def profile_batch_with_CP(self):
        """
        profile batch with CellProfiler (runs segmentation and feature extraction)
        """
        # need to specify plugin directory for CellProfiler to find the Cellpose plugin
        plugins_directory = (
            "/home/jenna/Desktop/Github/CellProfiler/cellprofiler/modules/plugins"
        )
        pipeline_path = "example_files/CP_files/mitocheck_idr_cp.cppipe"
        command = f"cellprofiler -c -r -p {pipeline_path} -o {self.CP_output_path} -i {self.CP_images_path} --plugins-directory {plugins_directory}"
        os.system(command)
        self.logger.info("CellProfiler run done")

    def profile_batch_with_DP(self):
        """
        profile batch with DeepProfiler
        """
        command = f"python3 -m deepprofiler --gpu 0 --root {self.DP_project_path} --config {self.config_name} profile"
        os.system(command)
        self.logger.info("Deep Profiler run done")

    def clear_batch(self, stream_type: str):
        """
        remove all intermediate files that are unecessary for next batch to run

        Parameters
        ----------
        stream_type: str
            segmention/feature extraction method (either DP or CP)
        """
        if stream_type == "CP":
            # remove all of the plate folders (with images) from images folder 
            images = pathlib.Path(f"{self.CP_project_path}/inputs/images/")
            for item in images.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
            # remove features folders for next batch
            features = pathlib.Path(f"{self.CP_project_path}/outputs/features/")
            shutil.rmtree(features)

        if stream_type == "DP":
            images = pathlib.Path(f"{self.DP_project_path}/inputs/images/")
            locations = pathlib.Path(f"{self.DP_project_path}/inputs/locations/")
            metadata = pathlib.Path(f"{self.DP_project_path}/inputs/metadata/")
            features = pathlib.Path(f"{self.DP_project_path}/outputs/results/features/")
            paths_to_remove = [images, locations, metadata, features]

            for path in paths_to_remove:
                shutil.rmtree(path)

            self.logger.info("Temporary batch files cleared")

    def compile_batch_CP_features(self, output_path: pathlib.Path):
        """
        compile single cell features from CellProfiler into one dataframe (to look like the output from DeepProfiler) and save as compressed csv
        to final output folder

        Args:
            output_path (pathlib.Path):
                path of final data folder
        """
        # load in the "Nuclei.csv" file that is created from the batch
        nuclei_table = pathlib.Path("tmp/CP_project/outputs/features/Nuclei.csv")
        cp_output = pd.read_csv(nuclei_table, dtype=object)

        # change 'Metadata_Well' column data format
        cp_output.drop("Metadata_Well", inplace=True, axis=1)
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
        ]

        # remove unnecessary metadata columns
        cp_output.drop(columns_to_drop, inplace=True, axis=1)

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
        deep_single_cell.get_single_cells(output=True).to_csv(
            output_path, compression={"method": "gzip", "compresslevel": 1, "mtime": 1}
        )

        self.logger.info("Batch features compiled with PyCytominer")

    def run_stream(
        self,
        stream_type: str,
        data_to_process: pd.DataFrame,
        batch_size: int = 10,
        start_batch: int = 0,
        batch_nums="all",
        extra_metadata=[],
    ):
        """
        extract features from IDR study given metadata of images to extract features from using a specific method (either DP or CP)

        Parameters
        ----------
        stream_type: str
            segmention/feature extraction method (either DP or CP)
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

            if stream_type == "CP":
                # stream_type = "CP"
                self.logger.info(f"Profiling batch {batch_num} with CellProfiler")
                try:
                    self.prepare_batch(
                        stream_type, batch_metadata
                    )  # put image and location data in CP-required locations
                    self.profile_batch_with_CP()  # profile batch with CellProfiler
                    output_path = pathlib.Path(
                        f"{self.final_data_dir}/batch_{batch_num}.csv.gz"
                    )
                    self.compile_batch_CP_features(
                        output_path
                    )  # compile and save features with PyCytominer
                    self.clear_batch(
                        stream_type
                    )  # delete image/segmentation data for batch
                except Exception as e:
                    self.logger.info(f"Error while profiling batch {batch_num}:")
                    self.logger.error(e)

            if stream_type == "DP":
                # stream_type = "DP"
                self.logger.info(f"Profiling batch {batch_num} with DeepProfiler")
                try:
                    self.prepare_batch(
                        stream_type, batch_metadata
                    )  # put image and location data in DP-required locations
                    self.compile_batch_index_csv(
                        stream_type, batch_metadata
                    )  # compile index csv for DeepProfiler project for the specific batch
                    self.profile_batch_with_DP()  # profile batch with Deep Profiler
                    features_path = pathlib.Path(
                        f"{self.final_data_dir}/batch_{batch_num}.csv.gz"
                    )
                    self.compile_batch_DP_features(
                        features_path
                    )  # compile and save features with PyCytominer
                    self.clear_batch(
                        stream_type
                    )  # delete image/segmentation data for batch
                except Exception as e:
                    self.logger.info(f"Error while profiling batch {batch_num}:")
                    self.logger.error(e)

        self.logger.info("Stream run done!")
