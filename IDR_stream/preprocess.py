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
    """
    This class holds all functions needed to preprocess IDR-downloaded mitosis movies with PyBaSiC.
    When preprocessing a mitosis movie, imagej is used to load the movie and convert it to a numpy array before illumination correction with PyBaSiC.

    Attributes
    ----------
    ij : ImageJ gateway
        loaded pyimagej wrapper

    Methods
    -------
    load_movie_data(movie_load_path)
        get numpy array of movie data from .ch5 file
    pybasic_illumination_correction(brightfield_images)
        use pybasic to correct brighfield images
    random_movie_sample(frame, max_frame_num, num_added_frames)
        get 2 random frames from movie in addition to frame of interest
    movie_to_corrected_frames(movie_load_path, frame_nums)
        get list corrected frames from mitosis movie at movie load path
    save_corrected_frames(plate, well_num, movie_load_path, frames_save_path, frame_nums)
        save corrected frames from the desired moive at movie load path to the frames save path
    """

    def __init__(self, fiji_path: pathlib.Path):
        """
        __init__ function for BasicpyPreprocessor class.

        Parameters
        ----------
        fiji_path : pathlib.Path
            path to installed FIJI program, ex pathlib.Path("/home/user/Fiji.app")
        """
        original_path = os.getcwd()
        self.ij = imagej.init(fiji_path)
        # imagej init sets directory to fiji_path so have to go back to original dir
        os.chdir(original_path)

    def load_movie_data(self, movie_load_path: pathlib.Path) -> np.ndarray:
        """
        get numpy array of movie data from .ch5 file

        Parameters
        ----------
        movie_load_path : pathlib.Path
            path to mitosis movie

        Returns
        -------
        np.ndarray
            array of movie data
        """
        # imagej prints lots of output that isnt necessary, unfortunately some will still come through
        with capture_output():
            jmovie = self.ij.io().open(str(movie_load_path.resolve()))
            movie = self.ij.py.from_java(jmovie)
            movie_arr = movie.values[:, :, :, 0]
            return movie_arr

    def pybasic_illumination_correction(self, brightfield_images: np.ndarray):
        """
        PyBaSiC Illumination correction as described in http://www.nature.com/articles/ncomms14836

        Parameters
        ----------
        brightfield_images : np.ndarray
            array of frames to perform illumination correction on

        Returns
        -------
        np.ndarray
            illumination corrected frames
        """
        # capture pybasic warnings
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
            brightfield_images_corrected = np.array(
                brightfield_images_corrected_original
            )
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

    def random_movie_sample(
        self, frame: int, max_frame_num: int, num_added_frames: int = 2
    ):
        """
        get random frames in addition to desired frame, need to give pybasic at least 3 frames to perform illumination correction

        Parameters
        ----------
        frame : int
            desired frame to include
        max_frame_num : int
            maximum frame number (length of movie)
        num_added_frames : int, optional
            how many random frames to add, by default 2

        Returns
        -------
        list of random frame and desired frame, with the desired frame at index 0
            
        """
        frame_nums = []
        frame_nums.append(frame)

        # get unique frame numbers that arent already in frame_nums
        for _ in range(num_added_frames):
            random_frame = choice(list(set(range(0, max_frame_num)) - set(frame_nums)))
            frame_nums.append(random_frame)

        return frame_nums

    def movie_to_corrected_frames(
        self, movie_load_path: pathlib.Path, frame_nums: list
    ):
        """
        convert movie to desired corrected frames

        Parameters
        ----------
        movie_load_path : pathlib.Path
            path to mitosis movie to load
        frame_nums : list
            list of frame numbers to keep from movie

        Returns
        -------
        list
            list of desired corrected frames
        """
        # if only 1 frame give basicpy 2 other random frames of context
        if len(frame_nums) == 1:
            new_frame_nums = self.random_movie_sample(frame_nums[0], 90)
            original_movie = self.load_movie_data(movie_load_path)
            choice_frames = []
            for frame_num in new_frame_nums:
                choice_frames.append(original_movie[frame_num - 1])
            corrected_movie = self.pybasic_illumination_correction(choice_frames)

            frames_list = [corrected_movie[0]]
            return frames_list

        # if more than one frame to correct process all of movie with basicpy
        else:
            original_movie = self.load_movie_data(movie_load_path)
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
        """
        load movie, convert to corrected frames, and save corrected frames

        Parameters
        ----------
        plate : str
            plate associated with movie
        well_num : int
            well number associated with movie
        movie_load_path : pathlib.Path
            path to mitosis movie
        frames_save_path : pathlib.Path
            path to save corrected frames
        frame_nums : list
            list of desired frame numbers to extract from mitosis movie
        """
        frames_save_path.mkdir(parents=True, exist_ok=True)

        corrected_frames_list = self.movie_to_corrected_frames(
            movie_load_path, frame_nums
        )
        for index, frame in enumerate(corrected_frames_list):
            frame_save_path = pathlib.Path(
                f"{frames_save_path}/{plate}_{well_num}_{frame_nums[index]}.tif"
            )
            skimage.io.imsave(frame_save_path, frame)

        os.remove(movie_load_path)
