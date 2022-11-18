"""
This function will convert the `data_to_process.tsv` file into a .csv so that CellProfiler can read in the metadata and connect the appropriate
metadata to the images from each batch. 
"""
import pandas as pd

def convert_tsv_to_csv(data_to_process_tsv: str):
    """ Convert `data_to_process.tsv` file into a .csv for CellProfiler to read in when processing the images

    Parameters
    ----------
        data_to_process_tsv : str
            string path to the `data_to_process.tsv` file
    """
    # read in metadata tsv file
    data_to_process_csv = pd.read_csv(data_to_process_tsv,sep='\t')
    
    # converting tsv file into csv
    data_to_process_csv.to_csv('example_files/data_to_process.csv',index=False)

