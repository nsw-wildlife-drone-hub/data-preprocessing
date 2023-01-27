import pandas as pd
import logging

class Yolo2df:
    def __init__(self, height, width, classes, columns, ext='.jpg'):
        """
        Initialize the Yolo2df object.

        Parameters:
        height (int): the height of the image.
        width (int): the width of the image.
        classes (list): list of classes in the dataset.
        columns (list): list of columns for the dataframe.
        ext (str): the file extension of the image (default: '.jpg').
        """
        self.height = height
        self.width = width
        self.classes = classes
        self.columns = columns
        self.ext = ext

    def write_df(self, files_list, ext='.jpg',):
        """
        Converts the YOLO detection output files to a dataframe.
        
        Parameters:
        files_list (list): a list of file paths for the YOLO output files.
        ext (str): the file extension of the image (default: '.jpg').
        
        Returns:
        df (pandas.DataFrame): a dataframe containing the detection data.
        """
        image_id = 0
        final_df = []
        
        # iterate through the list of files and read then split the contents
        for item in files_list:
            image_id += 1
            with open(item, 'rt') as fd:
                for line in fd.readlines():
                    row = []
                    nums = [float(num) for num in line.split()]
                    try:
                        row.append(self.classes[int(nums[0])])
                        row.extend(self._convert_yolo(*nums[1:]))
                        row.append(str(item.with_suffix(ext)))
                        row.append(self.width)
                        row.append(self.height)
                        row.append(int(item.stem))
                        row.append(item.parent.name)
                        final_df.append(row)
                    except Exception as e:
                        logging.exception(e)

        # convert the list of rows to a dataframe and remove duplicates
        df = pd.DataFrame(final_df, columns=self.columns)
        df = df.drop_duplicates('filename')

        return df

    def _convert_yolo(self, x, y, w, h):
        """
        Convert bounding box coordinates from YOLO format to COCO format.

        Parameters:
        x (float): The x-coordinate of the center of the bounding box, in the range [0, 1].
        y (float): The y-coordinate of the center of the bounding box, in the range [0, 1].
        w (float): The width of the bounding box, in the range [0, 1].
        h (float): The height of the bounding box, in the range [0, 1].

        Returns:
        [X, Y, W, H] (list): representing the bounding box coordinates in COCO format
        """
        X = round((x - w/2) * self.width, 1)
        Y = round((y - h/2) * self.height, 1)
        W = round(w * self.width, 1)
        H = round(h * self.height, 1)

        return X, Y, W, H