class Config:
    # versioning
    version   = '0.4'
    data_name = 'data_v'+version

    # dataframe columns
    ai_col      = 'AI Verified'
    vid_col     = 'video_filenames'
    name_col    = 'labelling_foldername'
    zip_col     = 'zip_path'
    srt_col     = 'srt_path'
    specie_col  = 'species_name'
    
    # run details
    HEIGHT = 512
    WIDTH = 640

    SINGLE_RUN = False            # test by using only 1 archive
    MAX_ARCHIVE = None            # limit number of archives to convert into training data
                                  # set to None for no limit
    ZIP_DATA = False              # automatically zip the data
    DOWNLOAD_DATA = False         # download the previously uploaded data                              
    GENERATE_CSV = True           # include a user-friendly csv file for labels
    NEW_LOOKUP = False            # delete old values and create a new lookup table