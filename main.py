import preprocess_data as pp
from Config import Config

import argparse
import logging
import patoolib
import os
import shutil
import json
import pandas as pd
import PIL

from datetime import datetime
from pathlib import Path


def parse_args():
    """
    Parses command line arguments for the script

    Returns:
    Namespace: An object containing the parsed argument values
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-detect_table_path', type=Path,
                        help='Path to detection table data')
    parser.add_argument('-source_path', type=Path,
                        help='Path to all available data')
    parser.add_argument('-data_path', type=Path,
                        help='Path to working directory data')
    parser.add_argument('-training_data_path', type=Path,
                        help='Path to training data')
    parser.add_argument('-duplicate_data_path', type=Path, default='dup_data.csv',
                        help='Path to duplicate data')
    parser.add_argument('-output', default='labels.json',
                        type=str, help='Name for output COCO file')
    parser.add_argument('-output_csv', default='labels.csv',
                        type=str, help='Name for output csv file')
    parser.add_argument('-save_local_data', default=False,
                        type=bool, help='Save a local copy of the data before unzipping')
    parser.add_argument('-overwrite_local_data', default=False,
                        type=bool, help='Overwrite local data if it already exists')
    args = parser.parse_args()

    logging.info('Detection path: ' + str(args.detect_table_path))
    logging.info('Source path: ' + str(args.source_path))
    logging.info('Data path: ' + str(args.data_path))
    logging.info('Training data path: ' + str(args.training_data_path))
    logging.info('Duplicate data path: ' + str(args.duplicate_data_path))

    return args


def load_data(args):
    """
    Loads, filters and selects data for processing

    Returns:
    DataFrame: A DataFrame object containing the filtered and selected data
    """
    logging.info('Selecting and filtering data.')

    # filter for available AI, Video column and exclusively koala dataset
    detect_df = pd.read_excel(args.detect_table_path)
    detect_df = detect_df[(detect_df[Config.ai_col]==True) &
                          (detect_df[Config.vid_col].notna())]
    detect_df = detect_df[[Config.vid_col, Config.name_col]]
    detect_df = detect_df.drop_duplicates()
    detect_df = detect_df[detect_df[Config.vid_col].str.endswith('.MP4')]

    # create column for .zip file names
    detect_df[Config.zip_col] = detect_df[Config.name_col].apply(
        lambda x: str(args.source_path)+'/'+x+'.zip')
    # zips = glob(str(args.source_path / '*'))
    # detect_df = detect_df[detect_df[Config.zip_col].isin(zips)]

    logging.info('Selecting SRT data.')

    # limit number of columns
    detect_df[Config.srt_col] = detect_df[Config.vid_col].apply(pp.convert_srt)
    if Config.MAX_ARCHIVE:
        detect_df = detect_df.head(min(len(detect_df), Config.MAX_ARCHIVE))

    return detect_df


def build_data(detect_df, args):
    """
    Extracts data from folders and archives
    """
    logging.info(f'{len(detect_df)} folders to extract: ')
    logging.info(', '.join(detect_df[Config.name_col].values))
    logging.info('Extracting data.')

    # create the data folder
    try:
        args.data_path.mkdir()
    except FileExistsError:
        if not args.overwrite_local_data:
            return None

    # iterate and extract each archive
    count = 0
    for zip_file in detect_df[Config.zip_col]:
        try:
            logging.info(f'Extracting {zip_file}.')
            if args.save_local_data:
                shutil.copy(zip_file, 'tmp.zip')
                zip_file = 'tmp.zip'
            patoolib.extract_archive(zip_file, outdir=args.data_path)
            count += 1
        except Exception:
            logging.info(f'Skipping {zip_file}')

    logging.info(f'Successfully extracted {count} archives')
    
    # remove empty directories
    for folder in args.data_path.glob('*'):
        if not any(folder.iterdir()):
            os.rmdir(folder)


def reduce_data_similarity(detect_df, args):
    """
    Removes duplicate images from data

    Args:
    detect_df (pandas.DataFrame): DataFrame containing image filepaths.
    args (argparse.Namespace): Namespace containing the following attributes:
        data_path (pathlib.Path): Path to the directory containing the images.
        duplicate_data_path (str): Filepath to the lookup table to check for duplicates.
    """
    # collate new images list
    img_dir_list = [args.data_path /
                    name for name in detect_df[Config.name_col]]

    # create new lookup table if configured
    lookup_df = pd.DataFrame(columns=['img_dir', 'dup_list'])
    if not Config.LOAD_LOOKUP:
        try:
            lookup_df = pd.read_csv(args.duplicate_data_path)
        except Exception as e:
            logging.warning(e)

    # remove duplicates
    DR = pp.DuplicateRemover(lookup_df=lookup_df,
                             resample=PIL.Image.LANCZOS)
    for img_dir in img_dir_list:
        DR.remove_duplicates(img_dir)

    # write new lookup table
    if Config.NEW_LOOKUP:
        logging.info(
            f'Overwriting duplicate lookup table at {args.duplicate_data_path}')
        DR.lookup_df.to_csv(args.duplicate_data_path, index=False)


def create_coco(detect_df, args):
    """
    Create COCO dataset from YOLO labels and SRT data.

    Args:
    detect_df (pandas.DataFrame): DataFrame containing YOLO labels.
    args (argparse.Namespace): Namespace containing the following attributes:
        data_path (pathlib.Path): Path to the directory containing the YOLO labels.
        output (str): Filepath to save the COCO data in JSON format.
        output_csv (str): Filepath to save the merged DataFrame in CSV format.
    """
    logging.info('Extracting SRT.')

    # extract corresponding SRT data
    SR = pp.SrtReader(detect_df,
                      srt_col_name=Config.srt_col,
                      folder_col_name=Config.name_col,
                      drop_cols=['color_md'])
    srt_df = SR.make_df()

    logging.info('Converting YOLO to DataFrame.')

    # convert yolo labels into a dataframe
    dir_list = args.data_path.rglob('*/*.txt')
    YOLO2DF = pp.Yolo2df(Config.HEIGHT, Config.WIDTH,
                         Config.classes, Config.columns)
    label_df = YOLO2DF.write_df(dir_list)

    logging.info('Merging SRT and yolo data.')

    # join SRT and yolo label dataframes
    join_key = [Config.name_col, 'frame']
    label_srt_df = label_df.join(srt_df.set_index(join_key), on=join_key)
    label_srt_df = label_srt_df.sort_values('timestamp')
    label_srt_df.reset_index(drop=True, inplace=True)

    logging.info('Creating coco dataset.')

    # save dataframe to COCO format
    date_created = datetime.now().strftime('%d/%m/%y')
    year = datetime.now().year
    DF2COCO = pp.DF2Coco(version=Config.version,
                         year=year,
                         date_created=date_created)
    label_coco = DF2COCO.convert_df(label_srt_df)

    json_output = os.path.join(args.data_path, args.output)
    logging.info('Count of images generated: ' +
                 str(len(label_coco['images'])))
    logging.info('Saving COCO data to ' + json_output)

    # save data to csv and json files
    json.dump(label_coco, open(json_output, "w"), indent=4)
    if Config.GENERATE_CSV:
        csv_output = os.path.join(args.data_path, args.output_csv)
        logging.info('Saving dataframe to ' + csv_output)
        label_srt_df.to_csv(csv_output)


def save_data(args):
    """
    Save the data in the specified format.

    Args:
    args (argparse.Namespace): Namespace containing the following attributes:
        data_path (pathlib.Path): Path to the directory containing the data to be saved.
    """
    # remove all files in data besides yolo data and directories
    keep_suffix = ['.jpg', '.txt', '']
    other_files = [x for x in args.data_path.glob('*/*') if x.suffix not in keep_suffix]
    if len(other_files):
        logging.info('Removing ', ', '.join(other_files))
        for other_file in other_files:
            os.remove(other_file)
    else:
        logging.info('No unncessary files to remove.')

    # zip data
    if Config.ZIP_DATA:
        shutil.make_archive(Config.data_name, 'zip', 'data')


def main():
    """
    Main function for the program
    """
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
    save_data(args)


if __name__ == '__main__':
    """
    Start logging for the main script
    """
    pp.start_logging(name='main')
    start_time = datetime.now()

    try:
        main()
    except Exception as e:
        logging.exception(e)
    finally:
        duration = str(datetime.now() - start_time).split('.')[0]
        logging.info('Total time taken: ' + duration)
