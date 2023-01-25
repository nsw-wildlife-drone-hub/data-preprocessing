import preprocess_data as pp
from Config import Config

import argparse
import logging
import patoolib
import os
import re
import shutil
import json
import pandas as pd
import numpy as np
import PIL

from multiprocessing.pool import Pool
from datetime import datetime
from glob import glob
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('detect_table_path', type=Path, help='Path to detection table data')
    parser.add_argument('source_path', type=Path,help='Path to all available data')
    parser.add_argument('data_path', type=Path,help='Path to working directory data')
    parser.add_argument('training_data_path', type=Path, help='Path to training data')
    parser.add_argument('duplicate_data_path', type=Path, help='Path to duplicate data')
    parser.add_argument('-output', default='labels.json', help='Name for output COCO file')
    parser.add_argument('-output_csv', default='labels.csv', help='Name for output csv file')
    args = parser.parse_args()

    logging.info('Detection path: ', args.detect_table_path)
    logging.info('Data path: ', args.data_path)
    logging.info('Training data path: ', args.training_data_path)
    logging.info('Duplicate data path: ', args.duplicate_data_path)
    
    return args

def load_data(args):
    detect_df = pd.read_excel(args.detect_table_path)
    
    logging.info('Selecting and filtering data.')

    detect_df = detect_df[(detect_df[Config.ai_col]) & (detect_df[Config.vid_col]) & (detect_df[Config.specie_col]=='koala')]
    detect_df = detect_df[[Config.vid_col, Config.name_col]]
    detect_df = detect_df.drop_duplicates()
    detect_df = detect_df[detect_df[Config.vid_col].str.endswith('.MP4')]
    detect_df[Config.zip_col] = args.source_path / detect_df[Config.name_col] + '.zip'
    zips = args.source_path.glob()
    detect_df = detect_df[detect_df[Config.zip_col].isin(zips)]
    
    logging.info('Selecting SRT data.')
    
    detect_df[Config.srt_col] = detect_df[Config.vid_col].apply(pp.convert_srt)
    if Config.MAX_ARCHIVE:
        detect_df = detect_df.head(min(len(detect_df), 1))
            
    return detect_df

def build_data(detect_df, args):
    logging.info('List of folders to extract')    
    logging.info(detect_df[Config.name_col])
    logging.info('Extracting data.')
        
    count = 0
    for folder, zip_file in zip(detect_df[Config.name_col], detect_df[Config.zip_col]):
        try: 
            logging.info('Extracting {folder}.')
            patoolib.extract_archive(zip_file, outdir=args.data_path)
            count += 1
        except Exception as e:
            logging.info(f'Skipping {folder}')

    logging.info(f'Successfully extracted {count} archives')
    for folder in argparse.glob('*'):
        if len(folder.glob('/*.jpg')) != len(folder.glob('/*.txt')):
            logging.warning(f'{folder} has an irregular number of files')
            logging.info('\tjpg count: ',len(folder.glob('/*.jpg')))
            logging.info('\ttxt count: ',len(folder.glob('/*.txt')))


def reduce_data_similarity(detect_df, args):
    img_dir_list = [args.data_path / name for name in detect_df[Config.name_col]]
    if Config.NEW_LOOKUP:
        lookup_df = pd.DataFrame(columns=['img_dir', 'dup_list'])
    else:
        lookup_df = pd.read_csv(args.duplicate_data_path)
    DR = pp.DuplicateRemover(lookup_df=lookup_df,
                             resample=PIL.Image.LANCZOS)
    for img_dir in img_dir_list:
        DR.remove_duplicates(img_dir)
    
    logging.info(f'Overwriting duplicate lookup table at {args.duplicate_data_path}')
    DR.lookup_df.to_csv(args.duplicate_data_path, index=False)
     
def create_coco(detect_df, args):
    logging.info('Extracting SRT.')
    SR = pp.SrtReader(detect_df, 
                srt_col_name=Config.srt_col,
                folder_col_name=Config.name_col, 
                drop_cols=['color_md'])
    srt_df = SR.make_df()
    label_df = pp.yolo_to_df(args.data_path)
        
    logging.info('Merging SRT and yolo data.')
    join_key = [Config.name_col, 'frame']
    label_srt_df = label_df.join(srt_df.set_index(join_key), on=join_key)
    label_srt_df = label_srt_df.sort_values('timestamp')
    label_srt_df.reset_index(drop=True, inplace=True)
    
    logging.info('Creating coco dataset.')
    date_created = datetime.now().strftime('%d/%m/%y')
    year = datetime.now().year
    DF2COCO = pp.DF2Coco(version=Config.version,
                    year=year,
                    date_created=date_created)
    label_coco = DF2COCO.convert_df(label_srt_df)
    logging.info('Count of images generated:', len(label_coco['images']))
    
    logging.info('Saving COCO data to', args.output)
    json.dump(label_coco, open(args.output, "w"), indent=4)
        
    if Config.GENERATE_CSV:
        logging.info('Saving dataframe to', args.output_csv)
        label_srt_df.to_csv(args.output_csv)
    
def save_data(args):
    exts = {x[-4:] for x in args.data_path.glob('*/*')}
    exts.remove('.jpg', '.txt')

    for ext in exts:
        non_labels = args.data_path.glob('*/*'(ext))
        print('Removing', len(non_labels), ext, 'files')
        for non_label in non_labels:
            os.remove(non_label)
    
    if Config.ZIP_DATA:
        shutil.make_archive(Config.data_name, 'zip', 'data') 
        
def main():
    args = parse_args()
    
    logging.info('Loading data.')
    detect_df = load_data(args)
    
    logging.info('Building data.')
    build_data(detect_df, args)
    
    logging.info('Reducing data similarity.')
    reduce_data_similarity(detect_df, args)
    
    logging.info('Extracting SRT dataframe.')
    create_coco(detect_df, args)
    
    logging.info('Saving data.')
    save_data()
            
if __name__=='__main__':
    pp.start_logging(name='main')
    
    try:
        main()
    except Exception as e:
        logging.error(e)