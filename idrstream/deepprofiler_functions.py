import pandas as pd
import pathlib
import os
from pycytominer.cyto_utils import DeepProfiler_processing

class DeepProfilerRun:

    def __init__(self, DP_project_path: pathlib.Path, config_name: str):
        self.DP_project_path = DP_project_path
        self.config_name = config_name

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
        # How would logging work?
        # self.logger.info(f"Compiled index.csv file to {index_csv_save_path}")

    def profile_batch_with_DP(self):
        """
        profile batch with DeepProfiler
        """
        command = f"python3 -m deepprofiler --gpu 0 --root {self.DP_project_path} --config {self.config_name} profile"
        os.system(command)
        # How would logging work?
        # self.logger.info("Deep Profiler run done")

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
