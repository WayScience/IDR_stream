import pandas as pd
import pathlib
import os


class AsperaDownloader:
    """
    This class holds all functions needed to download IDR image data with Aspera.
    
    Attributes
    ----------
    aspera_path : pathlib.Path
        path to aspera sdk
    aspera_key_path : pathlib.Path
        path to aspera openssh key
    screens : pd.Dataframe
        dataframe with curated IDR screen data for specific study
    idr_id: str
        accession ID for IDR study
    
    Methods
    -------
    get_IDR_image_path(plate, well_num)
        get image path for Aspera download by finding screen for specific image in screen metadata
    download_image(plate, well_num)
        download image corresponding to plate and well number with Aspera
        
    Example
    -------
    import pathlib
    from IDR_stream.download import AsperaDownloader
    
    aspera_path = pathlib.Path("/home/user/.aspera/ascli/sdk/ascp")
    aspera_key_path = pathlib.Path("asperaweb_id_dsa.openssh")
    screens_path = pathlib.Path("idr0013-screenA-plates.tsv")
    idr_id = "idr0013"
    save_dir = pathlib.Path("downloaded_images/")
    
    downloader = AsperaDownloader(aspera_path, aspera_key_path, screens_path, idr_id)
    downloader.download_image("LT0001_02", 4, save_dir)
    """
    
    def __init__(
        self,
        aspera_path: pathlib.Path,
        aspera_key_path: pathlib.Path,
        screens_path: pathlib.Path,
        idr_id: str,
    ):
        """
        __init__ function for AsperaDownloader class.

        Parameters
        ----------
        aspera_path : pathlib.Path
            path to aspera sdk, ex pathlib.Path("/home/user/.aspera/ascli/sdk/ascp")
        aspera_key_path : pathlib.Path
            path to aspera openssh key
        screens_path : pathlib.Path
            path to tsv with curated IDR screen data for specific study
        idr_id : str
            accession ID for IDR study, ex "idr0013"
        """
        self.aspera_path = aspera_path
        self.aspera_key_path = aspera_key_path
        self.screens = pd.read_csv(screens_path, sep="\t", header=None, names=["Plate", "Screen"])
        self.idr_id = idr_id

    def get_IDR_image_path(self, plate: str, well_num: int) -> str:
        """
        get image path for Aspera download by finding screen for specific image in screen metadata 

        Parameters
        ----------
        plate : str
            name of plate with desired image data
        well_num : int
            number of well with desired image data

        Returns
        -------
        str
            path to IDR location of image data for use by Aspera
        """
        # get location of screen
        screen_loc = (
            self.screens.loc[self.screens["Plate"] == plate, "Screen"]
            .item()
            .replace("../screens/", "")
            .replace(".screen", "")
        )
        well = str(well_num).zfill(3)
        image_path = (
            f"20150916-mitocheck-analysis/mitocheck/{screen_loc}/hdf5/00{well}_01.ch5"
        )
        return image_path
    
    def download_image(self, plate: str, well_num: int, save_dir: pathlib.Path) -> pathlib.Path:
        """
        download image corresponding to plate and well number with Aspera

        Parameters
        ----------
        plate : str
            name of plate with desired image data
        well_num : int
            number of well with desired image data
        save_dir : pathlib.Path
           path to dir to save IDR download in

        Returns
        -------
        pathlib.Path
            path to saved image data
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        image_path = self.get_IDR_image_path(plate, well_num)
        idr_location = f"{self.idr_id}@fasp.ebi.ac.uk:{image_path} "
        
        command = f"sudo {self.aspera_path} -TQ -l500m -P 33001 -i {self.aspera_key_path} {idr_location} {save_dir}"
        # print(command)
        os.system(command)

        return pathlib.Path(f"{save_dir}/00{str(well_num).zfill(3)}_01.ch5")