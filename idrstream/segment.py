from cellpose import models, core, io, utils
import logging
logging.getLogger('cellpose.core').setLevel(logging.WARNING)
logging.getLogger('cellpose.io').setLevel(logging.WARNING)

import pathlib
import pandas as pd
import skimage
import numpy as np

class CellPoseSegmentor:
    def __init__(self, model_specs: dict):
        self.use_GPU = core.use_gpu()
        print(">>> GPU activated? %d" % self.use_GPU)
        self.model_specs = model_specs

    def get_object_locations(self, image: np.ndarray) -> pd.DataFrame:
        """finds center X,Y of objects using specs from model_specs and return pandas array with center X,Y of objects
        Args:
            image (np.ndarray): image with objects to segment
            model_specs (dict): specifications for cellpose segmentation
        Returns:
            pd.DataFrame: dataframe with object center coords
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
            objects_data.append(object_data)

        objects_data = pd.DataFrame(objects_data)
        return objects_data

    def frames_to_objects(
        self,
        plate: str,
        well_num: int,
        frames_save_path: pathlib.Path,
        frame_nums: list,
    ):

        frame_objects_list = []
        for frame_num in frame_nums:
            frame_load_path = pathlib.Path(
                f"{frames_save_path}/{plate}_{well_num}_{frame_num}.tif"
            )

            frame = skimage.io.imread(frame_load_path)
            frame_object_locations = self.get_object_locations(frame)
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
    ):
        objects_save_path.mkdir(parents=True, exist_ok=True)

        frame_objects_list = self.frames_to_objects(
            plate, well_num, frames_save_path, frame_nums
        )
        for index, frame_objects in enumerate(frame_objects_list):
            frame_num = frame_nums[index]
            frame_objects_save_path = pathlib.Path(
                f"{objects_save_path}/{well_num}_{frame_num}-1-Nuclei.csv"
            )
            frame_objects.to_csv(frame_objects_save_path, index=False)
