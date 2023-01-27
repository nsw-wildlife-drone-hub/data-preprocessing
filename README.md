# Data Preprocessing

### About

Repository that performs:
- data selection
- duplication removal
- conversion from YOLO to COCO format
- read and merge with .srt data
- data archival

## Getting Started

### Installation

1. Clone the repository

```bash
git clone https://github.com/nsw-wildlife-drone-hub/data-preprocessing.git data-preprocessing
cd data-preprocessing
```

2. Install packages

```bash
pip install -r requirements.txt
```

### Usage

Use the **main.py** file to run the package with the following convention

```bash
main.py [-h] [-detect_table_path]  \
    [-source_path]  \
    [-data_path]  \
    [-training_data_path]  \
    [-duplicate_data_path] 

```

Example

```bash
python main.py main.py -detect_table_path Detection_Labelling_List.xlsx \
    -source_path Data/ \
    -data_path data/ \
    -training_data_path Training_Data/ \
    -duplicate_data_path Duplicate_lookup.csv
```