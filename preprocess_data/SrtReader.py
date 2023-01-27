import re
import pandas as pd

class SrtReader():
    # class for reading a list of .srt files and extracting the information
    # compiled into a dataframe
    def __init__(self, df, 
                 srt_col_name='srt_path',  
                 folder_col_name='labelling_foldername',
                 drop_cols=[], 
                 trim_time=True):
        """
        Initialize the SrtReader object.
        
        Parameters:
        df (pandas dataframe) : Dataframe containing srt file paths to be read.
        srt_col_name (str) : Column name in the dataframe containing the srt file paths (default: 'srt_path').
        folder_col_name (str) : Column name in the dataframe containing the folder names (default: 'labelling_foldername')
        drop_cols (list) : List of columns to drop from the dataframe after reading srt files.
        trim_time (bool) : Whether or not to trim the time from the start and end of each subtitle (default: True)."
        """
        self.srt_list = df[srt_col_name]
        self.folder_col_name = folder_col_name
        self.folder_list = df[folder_col_name]
        self.drop_cols = drop_cols
        self.trim_time = trim_time

    def make_df(self):
        """
        Create a dataframe from the srt files passed during initialization.

        Returns:
        final_df (pandas dataframe) : A dataframe containing the extracted information from the srt files.
        """
        # cycle through srt_list and read each file
        self.final_df = pd.DataFrame()
        for folder_name, srt_file in zip(self.folder_list, self.srt_list):
            if srt_file.startswith('drive/'):
                srt_file = srt_file.replace('drive/', '/content/drive/')
            srt_df = self.read_srt(srt_file)

            # create foldername column
            srt_df[self.folder_col_name] = folder_name

            # combine with final df
            self.final_df = pd.concat([self.final_df, srt_df])

        # modify timestamp
        if self.trim_time:
            self._trim_timestamp()

        # set new indexing
        self.final_df.reset_index(drop=True, inplace=True)
        
        # drop columns
        self.final_df.drop(self.drop_cols, axis=1, inplace=True)

        return self.final_df

    def read_srt(self, file_name):
        """
        Reads a single srt file and returns the extracted information as a dataframe.

        Parameters:
        file_name (str) : The file path of the srt file to be read

        Returns:
        content_df (pandas dataframe) : A dataframe containing the extracted information from the srt file.
        """
        # read the file
        with open(file_name, 'r') as f:
            lines = re.split('\n\n', f.read())

        # parse contents to srt conversion
        contents = [self.extract_content(line, n) for n, line in enumerate(lines) if len(line) > 250]
        content_df = pd.DataFrame.from_dict(contents)

        return content_df

    def extract_content(self, content, n):
        """
        Extracts relevant information from a single line of an srt file.

        Parameters:
        content (str) : A line from an srt file containing timestamp and other information
        n (int) : The line number of the srt file

        Returns:
        info (dict) : A dictionary containing the extracted information, timestamp and frame number.
        """
        # extract the time and info data
        time, info = re.sub('<[^>]+>', '', content).split('\n')[3:]
        
        # split info into dictionary items
        info = info.replace('Drone: ', '')
        info = info.replace('abs_alt', ',abs_alt')
        key_pairs = re.split(r'\] \[|\,', info)

        # convert the information to a dictionary
        info = dict(re.sub(r'\[|\]| ', '', pair).split(':', 1) for pair in key_pairs)
        
        # update timestamp to the dictionary
        info['timestamp'] = time    

        # update line number
        info['frame'] = n

        return info

    def _trim_timestamp(self):
        ## TODO
        pass
