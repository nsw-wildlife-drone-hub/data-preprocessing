from glob import glob
import pathlib
import logging
import pandas as pd
from datetime import datetime

def start_logging(name='logfile', folder='logs/'):
    pathlib.Path(folder).mkdir(exist_ok=True)
    format = f'%(asctime)s - {name} - %(levelname)s - %(message)s'
    log_name = datetime.now().strftime(f'{folder}{name}_%Y%m%d_%H_%M.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format=format,
        handlers=[logging.FileHandler(log_name),
                logging.StreamHandler()]
    )
    
    logging.info('Logging Start.')


def convert_srt(video_path):
    # function to fetch .SRT file based on video file path
    srt_path = None
    if type(video_path) is str:
        srt_path = video_path[:-3]+'SRT'
    elif type(video_path) == list:
        srt_path = [path.replace('.MP4', '.SRT') for path in video_path]
        
    return srt_path


def convert_yolo(x, y, w, h, width, height):
    # function to convert yolo to coco format
    X = round((x - w/2) * width, 1)
    Y = round((y - h/2) * height, 1)
    W = round(w * width, 1)
    H = round(h * height, 1)

    return [X, Y, W, H]

def yolo_to_df(yolo_dir, width, height, name_col, ext='.jpg',):
    # function to collect yolo labels and return a dataframe
    clmns = ['class', 'x', 'y', 'w', 'h', 'filename', 'width', 'height', 'frame', name_col]
    classes = [
        'koala',
        'glider',
        'bird',
        'macropod',
        'pig',
        'deer',
        'rabbit',
        'bandicoot',
        'horse',
        'fox',
    ]

    myFiles = glob(yolo_dir + '**/*.txt', recursive=True)

    image_id = 0
    final_df = []
    for item in myFiles:
        image_id += 1
        with open(item, 'rt') as fd:
            for line in fd.readlines():
                row = []
                nums = [float(num) for num in line.split()]
                try:
                    row.append(classes[int(nums[0])])
                    row = row + convert_yolo(*nums[1:])
                    row.append(item[:-4] + ext)
                    row.append(width)
                    row.append(height)
                    row.append(int(pathlib.Path(item).stem))
                    row.append(pathlib.PurePath(item).parent.name)
                    final_df.append(row)
                except Exception as e:
                    logging.info(e)
                    logging.info('Error converting {item[:-4]}{ext}')
                    logging.info(row)

    df = pd.DataFrame(final_df, columns=clmns)
    df = df.drop_duplicates('filename')

    return df
