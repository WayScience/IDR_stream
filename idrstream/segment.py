from cellpose import models, core, io, utils

import pathlib
import pandas as pd
import skimage
import numpy as np


class CellPoseSegmentor:
    """
    This class holds all functions needed to segment image data

    Attributes
    ----------
    use_GPU : bool
        whether or not GPU is available for use with CellPose
    model_specs : dict
        dictionary with parameters for segmentation

    Methods
    -------
    get_object_locations(image)
        finds center coords of objects using specs from model_specs and return pandas array with objects coords
    frames_to_objects(plate, well_num, frames_save_path, frame_nums)
        get object coords for each frame image from DP project format (plate/plate_well_frame.tif)
    save_nuclei_locations(plate, well_num, frames_save_path, frame_nums, objects_save_path)
        save object coords for each image in DP project format (plate/well_num_frame_num-1-Nuclei.csv)
    """

    def __init__(self, model_specs: dict):
        """
        __init__ function for CellPoseSegmentor class

        Parameters
        ----------
        model_specs : dict
            information for how to segment image data

            example:
            model_specs = {
                "model_type": "cyto",
                "channels": [0, 0],
                "diameter": 0,
                "flow_threshold": 0.8,
                "cellprob_threshold": 0,
                "remove_edge_masks": True,
            }
        """
        self.use_GPU = core.use_gpu()
        print(">>> GPU activated? %d" % self.use_GPU)
        self.model_specs = model_specs

    def get_object_locations(
        self, image: np.ndarray, extra_metadata: list = []
    ) -> pd.DataFrame:
        """
        finds center coords of objects using specs from model_specs and return pandas array with objects coords

        Parameters
        ----------
        image : np.ndarray
            image with objects to segment
        extra_metadata : list, optional
            list of extra metadata to include in final dataframe outputs (object_outlines, object_boxes, etc), by default []

        Returns
        -------
        pd.DataFrame
            dataframe with object center coords
        """
        objects_data = []

        cellpose_model = models.Cellpose(
            gpu=self.use_GPU, model_type=self.model_specs["model_type"]
        )
        masks, flows, styles, diams = cellpose_model.eval(
            image,
            diameter=self.model_specs["diameter"],
            channels=self.model_specs["channels"],
            flow_threshold=self.model_specs["flow_threshold"],
            cellprob_threshold=self.model_specs["cellprob_threshold"],
        )
        # remove cell masks if they are on the edge
        if self.model_specs["remove_edge_masks"]:
            masks = utils.remove_edge_masks(masks)

        outlines = utils.outlines_list(masks)
        for outline in outlines:
            centroid = outline.mean(axis=0)
            object_data = {
                "Location_Center_X": centroid[0],
                "Location_Center_Y": centroid[1],
            }
            if "object_outlines" in extra_metadata:
                object_data["object_outline"] = outline
            objects_data.append(object_data)

        objects_data = pd.DataFrame(objects_data)
        return objects_data

    def frames_to_objects(
        self,
        plate: str,
        well_num: int,
        frames_save_path: pathlib.Path,
        frame_nums: list,
        extra_metadata: list = [],
    ) -> list:
        """
        get object coords for each frame image from DP project format (plate/plate_well_frame.tif)

        Parameters
        ----------
        plate : str
            plate image data is from
        well_num : int
            well number image data is from
        frames_save_path : pathlib.Path
            path to directory to load frames from
        frame_nums : list
            frame number to segment
        extra_metadata : list, optional
            list of extra metadata to include in final dataframe outputs (object_outlines, object_boxes, etc), by default []

        Returns
        -------
        list
            list of object locations for each frame number given as argument
        """

        frame_objects_list = []
        for frame_num in frame_nums:
            frame_load_path = pathlib.Path(
                f"{frames_save_path}/{plate}_{well_num}_{frame_num}.tif"
            )

            frame = skimage.io.imread(frame_load_path)
            frame_object_locations = self.get_object_locations(frame, extra_metadata)
            frame_object_locations = frame_object_locations.rename(
                columns={
                    "Location_Center_X": "Nuclei_Location_Center_X",
                    "Location_Center_Y": "Nuclei_Location_Center_Y",
                }
            )
            frame_objects_list.append(frame_object_locations)

        return frame_objects_list

    def save_nuclei_locations(
        self,
        plate: str,
        well_num: int,
        frames_save_path: pathlib.Path,
        frame_nums: list,
        objects_save_path: pathlib.Path,
        extra_metadata: list = [],
    ):
        """
        save object coords for each image in DP project format (plate/well_num_frame_num-1-Nuclei.csv)

        Parameters
        ----------
        plate : str
            plate image data is from
        well_num : int
            well number image data is from
        frames_save_path : pathlib.Path
            path to directory to to load frames from
        frame_nums : list
            frame numbers to process images for
        objects_save_path : pathlib.Path
            path to directory to save nuclei locations to
        extra_metadata : list, optional
            list of extra metadata to include in final dataframe outputs (object_outlines, object_boxes, etc), by default []
        """
        objects_save_path.mkdir(parents=True, exist_ok=True)

        frame_objects_list = self.frames_to_objects(
            plate, well_num, frames_save_path, frame_nums, extra_metadata
        )
        for index, frame_objects in enumerate(frame_objects_list):
            frame_num = frame_nums[index]
            frame_objects_save_path = pathlib.Path(
                f"{objects_save_path}/{well_num}_{frame_num}-1-Nuclei.csv"
            )
            frame_objects.to_csv(frame_objects_save_path, index=False)
