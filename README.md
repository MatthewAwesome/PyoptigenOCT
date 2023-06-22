# PyoptigenOCT

For working with Bioptigen OCT data in Python. 

The motivation for this project stems from a desire to implement customized analysis of the raw OCT data. 

Referring to 'raw' OCT data, I mean the spectrometer data that forms the basis of measurment in SD-OCT. 

And if you're starting with raw OCT data, you need to turn it into an image. 

More precisely, there is image data encoded in the spectral data. To reveal the image data we need to tease it out using some mathematics. 

At least initially, that's the primary aim of this project: to process the raw data and reveal such that the image data is revealed. 

Receive as input, Bioptigen OCT data (e.g., .OCT files);

and, 

Provide as output, image data (e.g., an image of my retina; you can see in the example).

(This project is a work-progress. And may be periodically updated to include additional features and refinements of existing ones) 

## Installation and Requirements. 

The modules required to run the examples are included in the requirements.txt file. 

After cloning, enter into the PyoptigenOCT directory and run the following command (perhaps within a virtual environment)

pip install -r requirements.txt

## Usage

This project is not a standalone python module, and rather, is to be viewed as a sandbox. 

The primary motivation is to provide an embarkation point from which OCT processing code can be written and tested. (e.g. the code herein can be extended to include parallelization, such as using CUDA, etc)

Usage is detailed in the examples. 


