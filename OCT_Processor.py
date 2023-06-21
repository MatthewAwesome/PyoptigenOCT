"""
This file is contain a class that serves to process OCT images. 

A class will help house variables needed to reconstruct and 
process OCT data. 

This will keep the code neat. 
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
    # We initialize config 
    def __init__(self,configPath=None,octPath=None):
        if configPath:
            self.configPath = configPath
            self.config = ConfigParser(configPath)
            self.generateLinearizedIndices()
            self.upsampledPixels = None
            self.upFactor = 4
        if octPath: 
            self.octPath = octPath
    
    # A function to generate inearized indices
    def generateLinearizedIndices(self):
        # Pry into the config dictionary: 
        lambda0   = self.config['start_wavelength']  # convert to nanometers, eventually
        dLambda   = self.config['wavelength_spacing'] 
        corr_2    = self.config['second_order_correction']
        corr_3    = self.config['third_order_correction']
        corr_4    = self.config['fourth_order_correction']
        self.numPixels = int(self.config['line_length'])

        #Let's make an array of pixel values: 
        self.pixelArray = np.arange(1,self.numPixels+1)

        # and an array of corresponding wavelengths: 
        lambdas = lambda0 + dLambda*self.pixelArray + corr_2*self.pixelArray**2 + corr_3*self.pixelArray**3 + corr_4*self.pixelArray**4 

        # which convert to wave-numbers, or k-values: 
        kVect = (2*np.pi)/lambdas

        # Determine a k-step using the kVect: 
        deltaK = np.abs(kVect[0]-kVect[-1])/self.numPixels

        # This allows us to generate a vector of k-values
        # that are spaced linearly by the value deltaK. 
        kLinearized = np.flip(kVect[-1] + (self.pixelArray*deltaK))
        self.kLinearized = kLinearized

        # Okay, we have a linear mapping of pixel index to wavelengths. 
        # And we converted this linear mapping to a kVector, which is 
        # not spaced linearly in k-space. 

        # So we generate a curve using kVect and our pixel array, 
        interpolator = interp1d(kVect,self.pixelArray,kind='cubic',fill_value="extrapolate")
        
        # and sample the curve at linearized increments in k-space. 
        self.linearizedIndices = interpolator(kLinearized)
    
    # We can upsample our pixel array for better results: 
    def generateUpsampledPixels(self): 
        # We initalize an array of zeros: 
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
        self.config['c2'] = c2
        self.config['c3'] = c3
        # And with this in hand, why dont we compute the correction vector: 
        self.generateDispersionVector()

    # To get dispersion: 
    def getDispersioCoeffs(self): 
        return self.config['c2'],self.config['c3']
    
    # To generate dispersion compenstion vector: 
    def generateDispersionVector(self,c2=None,c3=None): 
        # We can use stored coefficients, if we have them. Else, we can input them here. 
        if c2 is None and c3 is None: 
            c2 = self.config['c2']
            c3 = self.config['c3']
        self.k0 = self.numPixels/2
        self.Gc = np.exp(1j*((c2*(self.pixelArray-self.k0)**2) + (c3*(self.pixelArray-self.k0)**3)))

    # To subtract reference: 
    def subtractReference(self,spectra): 
        spectra = ((spectra - self.referenceSpectrum)/self.referenceSpectrum) * self.lowPassReference
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
    
    # To window the spectra: 
    def windowSpectra(self,spectra,windowType): 
        if windowType == 'hann': 
            windowFunc = signal.windows.hann(spectra.shape[1]).astype('complex128')
        elif windowType == 'hamming': 
            windowFunc = signal.windows.hamming(spectra.shape[1]).astype('complex128')
        windowedSpectra = spectra * windowFunc
        return windowedSpectra

    # A function to pad the spectra: 
    def padSpectra(self,spectra): 
        padBlock = np.zeros((spectra.shape[0],int(spectra.shape[1]/2)),dtype='complex128')
        spectra = np.concatenate((padBlock,spectra,padBlock),axis=1)
        return spectra
    
    # A function to 
    def prepareSpectra(self,spectra): 
        # DC-removal: 

        # Resampling: 

        # Dispersion Compensation: 

        # Windowing: 

        # Padding: 

        self.spectra = spectra

    def transformSpectra(self,spectra): 
        # Do the fft: 
        transformedSpectra = np.fft.fft(spectra,axis=-1)
        # Truncate the result and return it: 
        return transformedSpectra[:,:1024]
        
    # Loads a bunch of raw spectrum data and returns it: 
    def loadOcu(self,path): 
        self.octPath = path
        ocuLoader = BOCT(path)
        ocuData = ocuLoader.read_oct_volume(diskbuffered=True)
        return ocuData
    
    # Averages a bunch of raw spectra and retunrs the the average (i.e. reverence spectrum)
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
            # if not average has been created, create on. 
            if index == 0: 
                reference = np.average(vol,axis=axes)
            # else we accumulate the average into a running variable.
            else: 
                reference = (reference + np.average(vol,axis=axes))/2
        # Let's make the reference a class variable: 
        self.referenceSpectrum = reference
        # And low pass filter it: 
        self.lowPassFilterSpectrum(self.referenceSpectrum)
        return self.referenceSpectrum
        # alteratively , we can use self.reference.. 

    def lowPassFilterSpectrum(self,spectrum):
        sos = signal.butter(4, 0.08, 'lp', output='sos')
        spectrumLength = spectrum.size
        # pad the spectrum: 
        padLength = 128
        spectrum = np.pad(spectrum,(padLength,padLength),mode='minimum',stat_length=(3,3))
        # And filter the spectrum: 
        lpSpectrum = signal.sosfilt(sos, spectrum)
        # remove the padding: 
        lpSpectrum = lpSpectrum[padLength:spectrumLength+padLength]
        self.lowPassReference = lpSpectrum
        return self.lowPassReference


    # To reconstruct an OCT image from OCT
    def reconstructOCT(self,ocuPath,logScaled=False,complexOutput=False): 
        # Set the path, we might want to access later.  
        self.octPath = ocuPath
        # Load the OCU data to be processed: 
        ocuLoader = BOCT(ocuPath)
        ocuData = ocuLoader.read_oct_volume(diskbuffered=True)
        # Next we go through the data, scan by scan, 
        # and process the data. So we make functions to help here

