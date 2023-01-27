from glob import glob
import pathlib
import logging
from datetime import datetime

def start_logging(name='logfile', folder='logs/'):
    """
    Start logging to a file and the console.

    Parameters:
    name (str) : The name of the log file (default: 'logfile').
    folder (str) : The folder to save the log file in (default: 'logs/').
    """
    # create logging folder
    pathlib.Path(folder).mkdir(exist_ok=True)
    
    # initialize logging config
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
    """
    Converts a video file path to its corresponding srt file path.

    Parameters:
    video_path (str or list) : path or list of paths of the video files.

    Returns:
    srt_path (str or list) : path or list of paths of the srt files the video files.
    """
    # function to fetch .SRT file based on video file path
    srt_path = None
    if type(video_path) is str:
        srt_path = video_path[:-3]+'SRT'
    elif type(video_path) == list:
        srt_path = [path.replace('.MP4', '.SRT') for path in video_path]
        
    return srt_path


