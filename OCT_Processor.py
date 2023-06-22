"""
This file contains a class that serves to process raw OCT data
(e.g. .OCU files from a bioptigen machine). 
"""

# The imports:
import numpy as np
from scipy import signal 
from scipy.interpolate import interp1d
from oct_converter.readers import *
from GetEngineConfig import *

"""
This Class is dedicated to working with raw OCT data, 
and keeps track of pertinent parameters that are used throughout the reconstruction process 
"""
class OCT_Processor(): 
    # We initialize config params. 
    def __init__(self,configPath=None):
        if configPath:
            self.configPath = configPath
            self.config = ConfigParser(configPath)
            self.generateLinearizedIndices()
            self.upsampledPixels = None
            self.upFactor = 4
    
    # Loads a bunch of raw spectrum data and returns it: 
    def loadOcu(self,path): 

        # Link the path to a loader object:
        self.ocuPath = path
        ocuLoader = BOCT(path)

        # Read the data into a numpy array:
        ocuData = ocuLoader.read_oct_volume(diskbuffered=True)

        # return the array:
        return ocuData
    
    # A function to generate inearized indices
    def generateLinearizedIndices(self):
    
        # Defining some parameters (e.g. wavelength spacing, etc.)
        lambda0   = self.config['start_wavelength']  # convert to nanometers, eventually
        dLambda   = self.config['wavelength_spacing'] 

        # We need to correct for non-linearities in the spectrometer.
        corr_2    = self.config['second_order_correction']
        corr_3    = self.config['third_order_correction']
        corr_4    = self.config['fourth_order_correction']

        # The number of pixels on the line-scan camera:
        self.numPixels = int(self.config['line_length'])

        #Let's make an array of pixel values: 
        self.pixelArray = np.arange(1,self.numPixels+1)

        # Correcting for non-linearities in the spectrometer:
        lambdas = lambda0 + dLambda*self.pixelArray + corr_2*self.pixelArray**2 + corr_3*self.pixelArray**3 + corr_4*self.pixelArray**4 

        # Taking the linear lambdas and converting them to k-space:
        kVect = (2*np.pi)/lambdas

        # Determine a k-step using the kVect: 
        deltaK = np.abs(kVect[0]-kVect[-1])/self.numPixels

        # This allows us to generate a vector of k-values
        # that are spaced linearly by the value deltaK. 
        kLinearized = np.flip(kVect[-1] + (self.pixelArray*deltaK))

        # And we store this vector as a member of the class, in case we need it later. 
        self.kLinearized = kLinearized

        # Next, we define a interpolator object to convert from pixel space to k-space. 
        interpolator = interp1d(kVect,self.pixelArray,kind='cubic',fill_value="extrapolate")
        
        # We then feed the linearized k-space vector into the interpolator, to generated linearized indices.
        # These are not wavenumber values, but, rather are indices that correspond to (fractional) pixel values.
        self.linearizedIndices = interpolator(kLinearized)
    
    # We can upsample our pixel array for better results: 
    def generateUpsampledPixels(self): 

        # We initalize an array of zeros that is upFactor times the number of pixels, 
        upsampledPixels = np.zeros(self.numPixels*self.upFactor)

        # and fill it via nested for-loop: 
        for i in range(self.numPixels): 
            for j in range(self.upFactor): 
                upsampledPixels[self.upFactor*i+j] = self.pixelArray[i] + j/self.upFactor

        # And once filled we make it a member of the class, since we'll need it 
        # when resampling the spectra. 
        self.upsampledPixels = upsampledPixels

    # To set dispersion: 
    def setDispersionCoeffs(self,c2,c3): 

        # We store the coefficients as members of the config dictionary:
        self.config['c2'] = c2
        self.config['c3'] = c3

        # And we generate a dispersion vector using these coefficients (see below) 
        self.generateDispersionVector()

    # To get dispersion coeffs (function not currently used, but may be useful in the future)
    def getDispersioCoeffs(self): 
        return self.config['c2'],self.config['c3']
    
    # To generate dispersion compenstion vector: 
    def generateDispersionVector(self,c2=None,c3=None): 

        # We can use stored coefficients, if we have them. Else, we can input them here. 
        if c2 is None and c3 is None: 
            c2 = self.config['c2']
            c3 = self.config['c3']
        
        # Defining a central pixel, since dispersion can be modeled as 'spreading' about a central point.
        self.k0 = self.numPixels/2

        # We then generate a dispersion vector, which is a complex exponential that we can multiply by the raw spectra.
        # It is this multiplication that ultimately performs the dispersion compensation.
        self.Gc = np.exp(1j*((c2*(self.pixelArray-self.k0)**2) + (c3*(self.pixelArray-self.k0)**3)))

    # Averages a bunch of raw spectra and retunrs the the average (i.e. reference spectrum)
    def getReferenceSpectrum(self,ocuData): 
        # We iterate through all scan data to make an average, 
        # we are working with a list of scan. 
        for index,scan in enumerate(ocuData): 
            # grab the scan volume: 
            vol = np.squeeze(scan.volume)
            # determine axes to compute average: 
            if vol.ndim == 2: 
                axes = 0
            elif vol.ndim == 3: 
                axes = (0,1)
            # if no average has been created, create one 
            if index == 0: 
                reference = np.average(vol,axis=axes)
            # else we accumulate the average into a running variable.
            else: 
                reference = (reference + np.average(vol,axis=axes))/2

        # Let's make the reference spectrum a mermber of this parent class. 
        self.referenceSpectrum = reference

        # And low pass filter it. (see below for usage of this filtered spectrum)
        self.lowPassFilterSpectrum(self.referenceSpectrum)

        # Lastly, we return the reference spectrum for plotting, visualiztion , etc. 
        return self.referenceSpectrum

    # To low pass filter the reference spectrum:
    def lowPassFilterSpectrum(self,spectrum):

        # Define a low pass filter object: 
        sos = signal.butter(4, 0.08, 'lp', output='sos')

        # Keep track of the length of the spectrum original spectrum, since we will pad it.
        spectrumLength = spectrum.size

        # Pad the spectrum: 
        padLength = 128
        spectrum = np.pad(spectrum,(padLength,padLength),mode='minimum',stat_length=(3,3))

        # And filter the padded spectrum. 
        lpSpectrum = signal.sosfilt(sos, spectrum)

        # After filtering, we remove the padding: 
        lpSpectrum = lpSpectrum[padLength:spectrumLength+padLength]

        # Again, we store the low pass filtered spectrum as a member of the class.
        self.lowPassReference = lpSpectrum

        # And we return it, in case we need it (e.g. for plotting)
        return self.lowPassReference

    # To subtract reference: 
    def subtractReference(self,spectra): 
        
        # The goal here is to distinguish the spectral interferences fringes from the 'carrier' spectrum.
        spectra = ((spectra - self.referenceSpectrum)/self.referenceSpectrum) * self.lowPassReference

        # And we return the result.
        # We don't store this result as a member of the class, since we will be making a processing pipeline,
        # and we don't want to store a bunch of intermediate results.
        return spectra
    
    # Function to resample the spectra
    def resampleSpectra(self,spectra): 
        # Upsampling or not? Either way, we generate an interpolation curve.  
        if self.upsampledPixels is None: 
            interpolator = interp1d(self.pixelArray,spectra,axis=1,kind='cubic')
        else: 
            # We upsampled via fft: 
            fftSpectra = np.fft.fft(spectra,axis=-1)
            # Zero-pad the result: 
            zeroBlock = np.zeros((spectra.shape[0],spectra.shape[1]*(self.upFactor-1)))
            fftSpectra = np.concatenate((fftSpectra[:,:int(spectra.shape[1]/2)],zeroBlock,fftSpectra[:,int(spectra.shape[1]/2):]),axis=1)
            # Take ifft: 
            upsampledSpectra = self.upFactor * np.real(np.fft.ifft(fftSpectra,axis=-1))
            # And use the result to make an interpolator object: 
            interpolator = interp1d(self.upsampledPixels,upsampledSpectra,axis=1,kind='cubic')
        # Which we use to resample our spectra in linear-k: 
        resampledSpectra = interpolator(self.linearizedIndices)
        # and return the result as a complex array. 
        return resampledSpectra.astype('complex128')
    
    # To apply dispersion compensation: 
    def applyDispersionCompensation(self,spectra): 
        compensatedSpectra = spectra*self.Gc
        return compensatedSpectra
    
    # To window the spectra. More window types (e.g. Gaussian) can be added here.
    def windowSpectra(self,spectra,windowType): 

        # determining the window function:
        if windowType == 'hann': 
            windowFunc = signal.windows.hann(spectra.shape[1]).astype('complex128')
        elif windowType == 'hamming': 
            windowFunc = signal.windows.hamming(spectra.shape[1]).astype('complex128')
        elif windowType == 'blackman':
            windowFunc = signal.windows.blackman(spectra.shape[1]).astype('complex128')
        elif windowType == 'blackmanharris':
            windowFunc = signal.windows.blackmanharris(spectra.shape[1]).astype('complex128')
        elif windowType == 'gaussian':
            windowFunc = signal.windows.gaussian(spectra.shape[1]).astype('complex128')
        
        # Applying the window function via multiplication: 
        windowedSpectra = spectra * windowFunc

        # And returning the result:
        return windowedSpectra

    # A function to pad the spectra (e.g. padding prior to fft)
    def padSpectra(self,spectra):

        # We pad the spectra with zeros, on block on each side: 
        padBlock = np.zeros((spectra.shape[0],int(spectra.shape[1]/2)),dtype='complex128')

        # And concatenate to position the zero-block accordingly:
        spectra = np.concatenate((padBlock,spectra,padBlock),axis=1)

        # Return the padded spectra: 
        return spectra
    
    def transformSpectra(self,spectra): 

        # Do the fft: 
        transformedSpectra = np.fft.fft(spectra,axis=-1)

        # Truncate the result and return it: 
        return transformedSpectra[:,:1024]