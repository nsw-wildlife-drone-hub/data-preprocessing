from preprocess_data import *

import pytest

def test_convert_srt():
    video_path = '/path/to/myvideo.MP4'
    expected_output = '/path/to/myvideo.SRT'
    assert convert_srt(video_path) == expected_output

    video_path = ['/path/to/myvideo1.MP4', '/path/to/myvideo2.MP4']
    expected_output = ['/path/to/myvideo1.SRT', '/path/to/myvideo2.SRT']
    assert convert_srt(video_path) == expected_output
    
    video_path = None
    expected_output = None
    assert convert_srt(video_path) == expected_output

def test_convert_yolo():
    x, y, w, h, width, height = 0.5, 0.5, 0.5, 0.5, 10, 10
    expected_output = [2.5, 2.5, 5.0, 5.0]
    
    assert convert_yolo(x, y, w, h, width, height) == expected_output