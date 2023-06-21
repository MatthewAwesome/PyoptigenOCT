"""
This file contains a function dedicated to reading the Bioptigen Engine's configuration file, 
e.g. Engine_xxxx.ini. 

The INI file contains spectrometer calibration data that is useful when reconstructing the 
raw spectrum files (e.g. .OCU files) into OCT images (e.g. B-scans). 
"""
import configparser

def ConfigParser(filePath): 
    """
    Input: a path to an .ini file to be parsed. 

    Output: a dictionary containing data useful for 
    the reconstruction of spectrometer data into OCT data.
    """
    # A parser object: 
    parser = configparser.ConfigParser()
    try: 
        parser.read(filePath)
        # The output dictionary: 
        calDict = {}
        # Let's get the spectrometer calibration data: 
        for el in parser['SPECTROMETER']: 
            calDict[el] = float(parser['SPECTROMETER'][el])
        # As well as some data about the frame-grabber (e.g. number of pixels on the line-scan camera)
        for el in parser['CAPTURE_DRIVER']:
            if el == "line_length" or el == "frame_duration":
                calDict[el] = float(parser['CAPTURE_DRIVER'][el])
        # return the output
        return calDict 
    except: 
        print("error parsing config file, returning None object")
        return None
