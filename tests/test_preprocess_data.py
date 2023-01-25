from preprocess_data import *

import pytest
import logging
import os
import pathlib

class TestStartLogging:
    def setup_method(self):
        self.name = 'logfile'
        self.folder = 'logs/'
    
    def test_create_log_folder(self):
        start_logging(name=self.name, folder=self.folder)
        assert os.path.exists(self.folder) == True
        
    def test_create_log_file(self):
        start_logging(name=self.name, folder=self.folder)
        assert len(os.listdir(self.folder)) > 0


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
    