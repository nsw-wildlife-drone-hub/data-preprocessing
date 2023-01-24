import PIL
import os
import logging

from multiprocessing.pool import Pool
from glob import glob

class DuplicateRemover():
    # class to check and remove similar images
    
    def __init__(self, lookup_df, remove=True, df_write=True, 
                 resample=1, fill_color='white'):
        self.lookup_df = lookup_df
        self.remove = remove
        self.df_write = df_write
        self.resample = resample
        self.fill_color = fill_color

    def remove_duplicates(self, img_dir):
        # iterate through a glob directory and remove duplicates
        # create list of images and labels
        img_list = glob(img_dir)
        txt_list = [os.path.splitext(img)[0]+'.txt' for img in img_list]
        
        if img_dir in self.lookup_df.img_dir.tolist():
            dup_list = self.lookup_df[self.lookup_df.img_dir==img_dir].dup_list.values[0]
            dup_list = eval(dup_list)
            logging.info(f'Found {len(dup_list)} duplicates in {len(img_list)} from lookup table')
        else:
            dup_list = self.find_duplicates(img_list)
            new_row = {'img_dir':img_dir, 'dup_list':dup_list}
            self.lookup_df = self.lookup_df.append(new_row, ignore_index=True)

        # remove files in the duplicate list
        if self.remove:
            for index in dup_list:
                os.remove(img_list[index[0]])
                os.remove(txt_list[index[0]])


    def find_duplicates(self, img_list):
        # initialize multiprocessing
        duplicates_list = []
        hash_keys = dict()
        with Pool() as pool:
            results = pool.map_async(self._make_hash, img_list)
            for idx, filehash in enumerate(results.get()):
                # check if the filehash is unique
                if filehash not in hash_keys:
                    hash_keys[filehash]=idx
                else:
                    duplicates_list.append((idx,hash_keys[filehash]))
            logging.info(f'Found {len(duplicates_list)} duplicates in {idx+1} files')

        return duplicates_list


    def get_grays(self, image, width, height):
        # convert the image to grayscale and resized
        if image.mode in ('RGBA', 'LA') and self.fill_color is not None:
            cleaned = PIL.Image.new(image.mode[:-1], image.size, self.fill_color)
            cleaned.paste(image, image.split()[-1])
            image = cleaned

        # convert to grayscale format
        image = image.convert('L')

        # resize the image
        image = image.resize((width, height), self.resample)

        return list(image.getdata())


    def _make_hash(self, filename):
        # convert image file into a dhash
        with PIL.Image.open(filename) as image:
            img_hash = self._convert_dhash(image)

            return img_hash

            
    def _convert_dhash(self, image, size=8):
        # convert an image to a difference hash
        width = size + 1
        grays = self.get_grays(image, width, width)

        # iterate across every pixel
        row_hash = 0
        col_hash = 0
        size_range = range(size)
        for y in size_range:
            for x in size_range:
                offset = y * width + x

                # calculate the difference between each row pixel
                row_bit = grays[offset] < grays[offset + 1]
                row_hash = row_hash << 1 | row_bit
                
                # calculate the difference between each col pixel
                col_bit = grays[offset] < grays[offset + width]
                col_hash = col_hash << 1 | col_bit

        return row_hash << (size * size) | col_hash