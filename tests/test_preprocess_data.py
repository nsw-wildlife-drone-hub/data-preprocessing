from preprocess_data import *

import pytest
import os
import tempfile

from PIL import Image
from pathlib import Path

# class TestStartLogging:
#     def setup_method(self):
#         self.name = 'logfile'
#         self.folder = 'logs/'

#     def test_create_log_folder(self):
#         start_logging(name=self.name, folder=self.folder)
#         assert os.path.exists(self.folder) == True

#     def test_create_log_file(self):
#         start_logging(name=self.name, folder=self.folder)
#         assert len(os.listdir(self.folder)) > 0


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

# Create a temporary directory for testing


@pytest.fixture
def tempdir():
    import tempfile
    temp_dir = tempfile.TemporaryDirectory()
    yield temp_dir
    temp_dir.cleanup()


def test_Yolo2df(tempdir):
    # Create test YOLO output files
    test_files = [Path('000000.txt'), Path('000001.txt')]
    for file in test_files:
        with open(os.path.join(tempdir.name, file), 'w') as f:
            f.write('0 0.1 0.2 0.3 0.4')

    # Initialize the Yolo2df object
    height, width = 480, 640
    classes = ['class1']
    columns = ['class', 'x', 'y', 'w', 'h', 'filename',
               'width', 'height', 'image_id', 'folder']
    yolo2df = Yolo2df(height, width, classes, columns)

    # Convert the YOLO output files to a dataframe
    files_list = [tempdir.name / file for file in test_files]
    df = yolo2df.write_df(files_list)

    # Assert that the dataframe has the correct number of rows
    assert df.shape[0] == len(test_files)

    # Assert that the dataframe has the correct columns
    assert list(df.columns) == columns

    # Assert that the dataframe contains the correct values
    assert df.loc[0, 'class'] == 'class1'
    assert df.loc[0, 'filename'] == os.path.join(tempdir.name, '000000.jpg')
    assert df.loc[0, 'width'] == width
    assert df.loc[0, 'height'] == height
    assert df.loc[0, 'image_id'] == 0
    assert df.loc[0, 'folder'] == tempdir.name.split('\\')[-1]

# def test_remove_duplicates():
#     with tempfile.TemporaryDirectory() as temp_dir:
#         img = Image.new('L', (10, 10), 255)
#         path1 = os.path.join(temp_dir, "img1.jpg")
#         path2 = os.path.join(temp_dir, "img2.jpg")
#         img.save(path1)
#         img.save(path2)
#         test_images = [path1, path2]
#         remover = DuplicateRemover(pd.DataFrame(columns=['img_dir', 'dup_list']), remove=True)
#         remover.remove_duplicates(Path(temp_dir))

#         assert not Path(path1).exists() or not Path(path2).exists()


def test_find_duplicates():
    with tempfile.TemporaryDirectory() as temp_dir:
        img = Image.new('L', (10, 10), 255)
        path1 = os.path.join(temp_dir, "img1.jpg")
        path2 = os.path.join(temp_dir, "img2.jpg")
        img.save(path1)
        img.save(path2)
        img_list = [path1, path2]
        dr = DuplicateRemover(pd.DataFrame())
        duplicates = dr.find_duplicates(img_list)

        assert len(duplicates) == 1
        assert duplicates[0] == (1, 0)


def test_convert_dhash():
    image = Image.new('L', (10, 10), 255)
    size = 8
    duplicate_remover = DuplicateRemover(pd.DataFrame())
    dhash = duplicate_remover._convert_dhash(image, size)

    assert isinstance(dhash, int)


def test_get_grays():
    img = Image.new('L', (10, 10), 255)
    img.putdata([i for i in range(100)])
    d = DuplicateRemover(pd.DataFrame())
    grays = d._get_grays(img, 5, 5)

    assert len(grays) == 25
    assert all(isinstance(x, int) for x in grays)
