from importlib.resources import path
import pandas as pd
import os
import pathlib

import imagej
import skimage
import numpy as np
from IPython.utils.io import capture_output

import warnings
import PyBaSiC.pybasic as pybasic

from random import choice

class BasicpyPreprocessor:
    def __init__(self, fiji_path: pathlib.Path):
        original_path = os.getcwd()
        self.ij = imagej.init(fiji_path)
        # imagej init sets directory to fiji_path so have to go back to original dir
        os.chdir(original_path)
        
    def load_movie_data(self, movie_load_path: pathlib.Path) -> np.ndarray:
        # imagej prints lots of output that isnt necessary, unfortunately some will still come through
        with capture_output():
            jmovie = self.ij.io().open(str(movie_load_path.resolve()))
            movie = self.ij.py.from_java(jmovie)
            movie_arr = movie.values[:, :, :, 0]
            return movie_arr
        
    def pybasic_illumination_correction(self, brightfield_images: np.ndarray):
        """
        PyBaSiC Illumination correction as described in http://www.nature.com/articles/ncomms14836
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flatfield, darkfield = pybasic.basic(
                brightfield_images, darkfield=True, verbosity=False
            )
            baseflour = pybasic.background_timelapse(
                images_list=brightfield_images,
                flatfield=flatfield,
                darkfield=darkfield,
                verbosity=False,
            )
            brightfield_images_corrected_original = pybasic.correct_illumination(
                images_list=brightfield_images,
                flatfield=flatfield,
                darkfield=darkfield,
                background_timelapse=baseflour,
            )

            # convert corrected images to numpy array, normalize, and convert to uint8
            brightfield_images_corrected = np.array(brightfield_images_corrected_original)
            brightfield_images_corrected[
                brightfield_images_corrected < 0
            ] = 0  # make negatives 0
            brightfield_images_corrected = brightfield_images_corrected / np.max(
                brightfield_images_corrected
            )  # normalize the data to 0 - 1
            brightfield_images_corrected = (
                255 * brightfield_images_corrected
            )  # Now scale by 255
            corrected_movie = brightfield_images_corrected.astype(np.uint8)

            return corrected_movie
        
    def random_movie_sample(self, frame: int, max_frame_num: int, num_added_frames: int=2):
        frame_nums = []
        frame_nums.append(frame)

        # get unique frame numbers to give pybasic some context for movie without processing entire movie
        for _ in range(num_added_frames):
            random_frame = choice(list(set(range(0, max_frame_num)) - set(frame_nums)))
            frame_nums.append(random_frame)

        return frame_nums
    
    def movie_to_corrected_frames(self, movie_load_path: pathlib.Path, frame_nums: list):
        # if only 1 frame give basicpy 3 random frames of context
        if len(frame_nums) == 1:
            new_frame_nums = self.random_movie_sample(frame_nums[0], 90)
            original_movie = self.load_movie_data(movie_load_path, self.ij)
            choice_frames = []
            for frame_num in new_frame_nums:
                choice_frames.append(original_movie[frame_num - 1])
            corrected_movie = self.pybasic_illumination_correction(choice_frames)

            frames_list = [corrected_movie[0]]
            return frames_list

        # if more than one frame to correct process all of movie with basicpy
        else:
            original_movie = self.load_movie_data(movie_load_path, ij)
            corrected_movie = self.pybasic_illumination_correction(original_movie)

            frames_list = []
            for frame_num in frame_nums:
                frames_list.append(corrected_movie[frame_num - 1])
            return frames_list
        
    def save_corrected_frames(
        self,
        plate: str,
        well_num: int,
        movie_load_path: pathlib.Path,
        frames_save_path: pathlib.Path,
        frame_nums: list,
    ):
        frames_save_path.mkdir(parents=True, exist_ok=True)

        corrected_frames_list = self.movie_to_corrected_frames(movie_load_path, frame_nums)
        for index, frame in enumerate(corrected_frames_list):
            frame_save_path = pathlib.Path(
                f"{frames_save_path}/{plate}_{well_num}_{frame_nums[index]}.tif"
            )

            skimage.io.imsave(frame_save_path, frame)

        os.remove(movie_load_path)