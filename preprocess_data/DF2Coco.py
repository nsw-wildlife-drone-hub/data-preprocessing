import pandas as pd

class DF2Coco():
    def __init__(self, version='', date_created='', year=''):
        """
        Initialize the DF2Coco object.

        Args:
        version (str): version of the dataset.
        date_created (str): date when the dataset is created.
        year (str): year when the dataset is created.
        """
        self.year = year
        self.version = version
        self.date_created = date_created

    def convert_df(self, data_input):
        """
        Convert dataframe of labels to COCO format
        
        Parameters:
        data_input: dataframe of labels, containing columns 'filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax'
        
        Returns:
        data_coco: COCO format data, containing keys 'images', 'categories', 'annotations', and 'info'
        """
        data = data_input.copy()
        data['fileid'] = data['filename'].astype('category').cat.codes
        data['categoryid'] = pd.Categorical(data['class'], ordered=True).codes + 1
        data['annid'] = data.index

        annotations = [self._gen_annotation(row) for row in data.itertuples()]

        imagedf = data.drop_duplicates(subset=['fileid']).sort_values(by='fileid')
        images = [self._gen_image(row) for row in imagedf.itertuples()]

        catdf = data.drop_duplicates(subset=['categoryid']).sort_values(by='categoryid')
        categories = [self._gen_category(row) for row in catdf.itertuples()]

        data_coco = {}
        data_coco['images'] = images
        data_coco['categories'] = categories
        data_coco['annotations'] = annotations
        data_coco['info'] = self._gen_info()

        return data_coco

    def _gen_image(self, row):
        """
        Match row fields to a dictionary field for image
        """
        # populate image info
        image = {
            'height': row.height,
            'width': row.width,
            'id': row.fileid,
            'file_name': row.filename,
            'latitude': row.latitude,
            'longitude': row.longitude if 'longitude' in row else row.longtitude,
            'rel_alt': row.rel_alt,
            'abs_alt': row.abs_alt,
            'yaw': row.Yaw,
            'pitch': row.Pitch,
            'roll': row.Roll,
            'timestamp': row.timestamp,
        }
        
        return image

    def _gen_category(self, row):
        """
        Match row fields to a dictionary field for category
        """
        category = {
            'supercategory': 'Animals',
             'id': row.categoryid - 1,                           
             'name': row[1]
             }

        return category

    def _gen_annotation(self, row):
        """
        Match row fields to a dictionary field for annotation
        """
        area = row.w * row.h
        annotation = {
            'segmentation': [],
            'iscrowd': 0,
            'area': area,
            'image_id': row.fileid,
            'bbox': [row.x, row.y, row.w, row.h],
            'category_id': row.categoryid - 1,
            'id': row.annid
            }

        return annotation

    def _gen_info(self):
        """
        Match row fields to a dictionary field for info
        """
        info = {
            'year': self.year,
            'version': self.version,
            'date_created': self.date_created
        }

        return info