"""
This script is a simple example of how to use the OCT_Processor class. 
"""

#%% The imports:
import os 
import numpy as np
from matplotlib import pyplot as plt
# Change the working directory to the root of the project so we can import OCT_Processor:
os.chdir("..")
from OCT_Processor import *


#%%

# Let's initialize our class and see what happens: 
configPath = "sampleData/Engine_Config.ini"
processor = OCT_Processor(configPath=configPath)

# Let's get some upsampled pixels: 
processor.generateUpsampledPixels()

# Set dispersion Coeffs. 
c2 = -3e-6
c3 = -1e-9
processor.setDispersionCoeffs(c2, c3)

#%% We're ready to load some data: 
ocuPath   = "sampleData/raw.OCU"
ocuData   = processor.loadOcu(ocuPath)

#%% Let's give our class some a reference spectrum to work with:  
processor.getReferenceSpectrum(ocuData)
plt.figure()
plt.plot(processor.referenceSpectrum)
plt.plot(processor.lowPassReference)
#%% Now, let's reconstruct a scan. 
spectra = np.squeeze(ocuData[0].volume)
spectra = processor.subtractReference(spectra)
spectra = processor.resampleSpectra(spectra)
spectra = processor.applyDispersionCompensation(spectra)
spectra = processor.windowSpectra(spectra, windowType='hamming')
spectra = processor.padSpectra(spectra)
transformed  = np.fft.fft(spectra,axis=-1)
image = np.abs(transformed[:,:1024].transpose())
plt.figure()
plt.imshow(image,cmap='gray')

# %%
