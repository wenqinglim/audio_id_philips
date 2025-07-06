import librosa
import math
import numpy as np
import os
import pickle
from tqdm import tqdm

class fingerprintBuilder():
    """
    Fingerprinting as defined in Philips algorithm
    """
    def __init__(self,  print_logs=False):
        self.print_logs = print_logs

        # Define stft params
        self.overlap_factor = 64
        self.window_size = 0.37

        self.song_db = {}

        print(f"Initialized fingerprint builder with overlap_factor {self.overlap_factor} and window_size {self.window_size}")


    def __call__(self, db_path, fingerprint_path):

        for input_file in tqdm(os.listdir(db_path)):
            melspec = self.get_melspec(f"{db_path}/{input_file}")
            fingerprint =  self.get_fingerprint(melspec)
            self.song_db[input_file] = fingerprint

        with open(f'{fingerprint_path}/fingerprints_{self.overlap_factor}.pkl', 'wb') as handle:
            pickle.dump(self.song_db, handle, protocol=pickle.HIGHEST_PROTOCOL)


        
    def get_melspec(self, file_path):
    
        orig_x, orig_sr = librosa.load(f"{file_path}")
        
        """
        Since the algorithm only takes into account frequencies
        below 2kHz the received audio is first down sampled to a mono
        audio stream with a sampling rate of 5kHz
        """
        sr = 5000
        x = librosa.resample(orig_x, orig_sr=orig_sr, target_sr=sr)
    
        """
        The audio signal is first segmented into overlapping frames. The
        overlapping frames have a length of 0.37 seconds and are
        weighted by a Hanning window with an overlap factor of 31/32.
        """
        hop_size = self.window_size/self.overlap_factor
        n_fft = math.ceil(self.window_size * sr)
        hop_len = math.ceil(hop_size * sr)
        
        """
        In order to extract a 32-bit sub-fingerprint value for every frame,
        33 non-overlapping frequency bands are selected. These bands lie
        in the range from 300Hz to 2000Hz (the most relevant spectral
        range for the HAS) and have a logarithmic spacing. 
        """
        n_mels = 33
        f_min = 300
        f_max = 2000
        
        melspec = librosa.feature.melspectrogram(y=x,
                                                 sr=sr, 
                                                 n_fft=n_fft, 
                                                 hop_length=hop_len, 
                                                 power=1, 
                                                 n_mels = n_mels, 
                                                 fmin = f_min, 
                                                 fmax = f_max,
                                                )
        return melspec



    def sub_fp_bit(self, n_frame, m_bit, melspec, encode=True):
        """
        Calculate sub-fingerprint bit value by taking energy band differences.
    
        If encode=True, return bit value in 1s and 0s.
        If not, simply return the raw value of the energy band differences.
        """
        # Transpose melspec so that x-axis corresponds to number of frames, and y_axis corresponds to number of bits
        energy_band = melspec.T
        
        # Calculate energy band differences based on formula described in paper (Equation (1) in paper)
        diff = energy_band[n_frame, m_bit] - energy_band[n_frame, m_bit+1] - (energy_band[n_frame-1, m_bit] - energy_band[n_frame-1, m_bit+1])
    
        if encode:
            return int(diff > 0)
        else:
            return diff
        
    def get_fingerprint(self, melspec, encode=True):
        """
        Get fingerprint representation from a mel-spectrogram.
    
        Returns all the sub-fingerprints for each frame in the mel-spetrogram.
        """
        fingerprints = []
        # Iterate through every frame to get a 32-bit sub-fingerprint for each frame
        for frame_idx in range(melspec.shape[1]):
            sub_fingerprint = []
            # fill in each bit in sub-fingerprint
            for bit_idx in range(32):
                sub_fingerprint.append(self.sub_fp_bit(frame_idx, bit_idx, melspec, encode))
            fingerprints.append(sub_fingerprint)
        return np.array(fingerprints)
    
    
