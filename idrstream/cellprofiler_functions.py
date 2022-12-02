"""
This file holds the CellProfilerRun class with all of the functions to use CellProfiler for segmentation and feature extraction on a batch. These functions
include converting the data_to_process.tsv file in a csv for CellProfiler to use as metadata, profile the batch with CellProfiler, and compile the
Nuclei.csv outputted features.
"""
import pandas as pd
import os
import pathlib


class CellProfilerRun:
    """
    This class holds all the functions needed to perform a CellProfiler run (segmentation and feature
    extraction)

    Attributes
    ----------
    CP_output_path : pathlib.Path
        path to where the .csv files outputted from each CellProfiler batch run will go
    CP_images_path : pathlib.Path
        path to the images for each batch
    pipeline_path : pathlib.Path
        path to CellProfiler pipeline to perform segmentation and feature extraction
    plugins_directory : pathlib.Path
        path to CellProfiler directory with Cellpose plugin

    Methods
    -------
    convert_tsv_to_csv(data_to_process_tsv, metadata_save_path)
        convert `data_to_process.tsv` file into a .csv for CellProfiler to read in when processing the images
    profile_batch_with_CP()
        profile batch with CellProfiler (runs segmentation and feature extraction)
    compile_batch_CP_features(output_path)
        compile single cell features from CellProfiler into one dataframe (to look like the output from DeepProfiler) and save as compressed csv 
        to final output folder
    """

    def __init__(
        self,
        CP_output_path: pathlib.Path,
        CP_images_path: pathlib.Path,
        pipeline_path: pathlib.Path,
        plugins_directory: pathlib.Path,
    ):
        """
        __init__ function for the CellProfilerRun class

        Parameters
        ----------
        CP_output_path : pathlib.Path
            path to where the .csv files outputted from each CellProfiler batch run will go
        CP_images_path : pathlib.Path
            path to the images for each batch
        pipeline_path : pathlib.Path
            path to CellProfiler pipeline to perform segmentation and feature extraction
        plugins_directory : pathlib.Path
            path to CellProfiler directory with Cellpose plugin
        """
        self.CP_output_path = CP_output_path
        self.CP_images_path = CP_images_path
        self.pipeline_path = pipeline_path
        self.plugins_directory = plugins_directory

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

    def profile_batch_with_CP(self):
        """
        profile batch with CellProfiler (runs segmentation and feature extraction)
        """
        # need to specify plugin directory for CellProfiler to find the Cellpose plugin
        command = f"cellprofiler -c -r -p {self.pipeline_path} -o {self.CP_output_path} -i {self.CP_images_path} --plugins-directory {self.plugins_directory}"
        os.system(command)

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
