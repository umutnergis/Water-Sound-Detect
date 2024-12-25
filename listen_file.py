import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import numpy as np
from scipy.fft import fft
from scipy import signal
import wave
import os
from datetime import datetime

class WaterSoundComparer:
    def __init__(self):
        # Initialize I2C bus
        self.i2c = busio.I2C(board.SCL, board.SDA)
        
        # Initialize ADS1115
        self.ads = ADS.ADS1115(self.i2c)
        self.ads.gain = 1
        self.ads.data_rate = 860
        
        # Connect MAX9814 output to A0 on ADS1115
        self.chan = AnalogIn(self.ads, ADS.P0)
        
        # Audio parameters
        self.RATE = 860
        self.CHANNELS = 1
        self.WINDOW_SIZE = 860  # 1 second of samples
        
        # Directory containing saved water sounds
        self.samples_dir = "water_samples"
        
        # Load reference features
        self.reference_features = {}
        self.load_reference_sounds()

    def load_reference_sounds(self):
        """Load all saved water sounds and extract their features"""
        if not os.path.exists(self.samples_dir):
            print("No reference sounds found! Please record some sounds first.")
            return

        wav_files = [f for f in os.listdir(self.samples_dir) if f.endswith('.wav')]
        if not wav_files:
            print("No .wav files found in water_samples directory!")
            return

        print("Loading reference sounds...")
        for wav_file in wav_files:
            filepath = os.path.join(self.samples_dir, wav_file)
            with wave.open(filepath, 'rb') as wf:
                # Read the wave file
                n_frames = wf.getnframes()
                audio_data = wf.readframes(n_frames)
                samples = np.frombuffer(audio_data, dtype=np.int16)
                
                # Extract features
                features = self.extract_features(samples)
                if features is not None:
                    self.reference_features[wav_file] = features
                    print(f"Loaded features from {wav_file}")

    def extract_features(self, samples):
        """Extract features from audio samples"""
        if len(samples) == 0:
            return None
            
        # Convert to numpy array if needed
        samples = np.array(samples)
        
        # Apply Hanning window
        windowed = samples * signal.windows.hann(len(samples))
        
        # Compute FFT
        fft_data = fft(windowed)
        fft_freq = np.fft.fftfreq(len(samples), 1/self.RATE)
        
        # Get positive frequencies only
        positive_freq_mask = fft_freq > 0
        fft_data = np.abs(fft_data[positive_freq_mask])
        fft_freq = fft_freq[positive_freq_mask]
        
        # Focus on frequency ranges typical for water sounds (100-2000 Hz)
        water_freq_mask = (fft_freq >= 100) & (fft_freq <= 2000)
        water_fft = fft_data[water_freq_mask]
        
        if len(water_fft) == 0:
            return None
            
        # Extract features
        features = {
            'mean_power': np.mean(water_fft),
            'std_power': np.std(water_fft),
            'spectral_centroid': np.average(fft_freq[water_freq_mask], weights=water_fft),
            'spectral_flatness': self._spectral_flatness(water_fft),
            'temporal_consistency': np.std(samples) / np.mean(np.abs(samples))
        }
        
        return features

    def _spectral_flatness(self, spectrum):
        """Calculate spectral flatness (Wiener entropy)"""
        spectrum = spectrum + 1e-10  # Avoid log(0)
        geometric_mean = np.exp(np.mean(np.log(spectrum)))
        arithmetic_mean = np.mean(spectrum)
        return geometric_mean / arithmetic_mean

    def compare_with_references(self, features, threshold=0.7):
        """Compare current features with all reference features"""
        if not self.reference_features:
            return False, None
            
        best_match = None
        best_similarity = 0
        
        for filename, ref_features in self.reference_features.items():
            # Calculate similarity scores
            scores = []
            for key in features.keys():
                if key in ref_features:
                    ref_val = ref_features[key]
                    curr_val = features[key]
                    # Calculate normalized difference
                    similarity = 1 - min(abs(ref_val - curr_val) / max(abs(ref_val), 1e-10), 1)
                    scores.append(similarity)
            
            average_similarity = np.mean(scores)
            if average_similarity > best_similarity:
                best_similarity = average_similarity
                best_match = filename
        
        return best_similarity > threshold, best_match

    def monitor_realtime(self):
        """Monitor real-time audio and compare with saved sounds"""
        if not self.reference_features:
            print("No reference sounds loaded! Please record some sounds first.")
            return
            
        print("\nMonitoring for water sounds... (Press Ctrl+C to stop)")
        buffer = []
        
        try:
            while True:
                # Read sample
                voltage = self.chan.voltage
                sample = int((voltage / 3.3) * 32767)
                buffer.append(sample)
                
                # When buffer is full
                if len(buffer) >= self.WINDOW_SIZE:
                    # Extract features
                    features = self.extract_features(buffer)
                    
                    # Compare with references
                    if features:
                        is_match, matching_file = self.compare_with_references(features)
                        if is_match:
                            print(f"Match found! Similar to {matching_file}")
                    
                    # Clear buffer with overlap
                    buffer = buffer[self.WINDOW_SIZE//2:]
                
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\nStopping monitoring...")

def main():
    comparer = WaterSoundComparer()
    comparer.monitor_realtime()

if __name__ == "__main__":
    main()
