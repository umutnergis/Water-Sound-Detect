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
        
        # Connect MAX9814 output to A0 and LDR to A1 on ADS1115
        self.sound_chan = AnalogIn(self.ads, ADS.P0)
        self.ldr_chan = AnalogIn(self.ads, ADS.P1)
        
        # Audio parameters
        self.RATE = 860
        self.CHANNELS = 1
        self.WINDOW_SIZE = 1720  # 2 seconds for better analysis
        
        # Sound detection parameters
        self.MIN_AMPLITUDE = 0.1
        self.consecutive_matches = 0
        self.last_match_time = None
        self.MATCH_TIMEOUT = 10  # seconds
        self.REQUIRED_MATCHES = 20

        # LDR parameters
        self.light_start_time = None
        self.LIGHT_THRESHOLD = 2.4
        self.LIGHT_ALERT_THRESHOLD = 30  # seconds
        
        # Directory and features
        self.samples_dir = "water_samples"
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
            try:
                with wave.open(filepath, 'rb') as wf:
                    n_frames = wf.getnframes()
                    audio_data = wf.readframes(n_frames)
                    samples = np.frombuffer(audio_data, dtype=np.int16)
                    
                    features = self.extract_features(samples)
                    if features is not None:
                        self.reference_features[wav_file] = features
                        print(f"Loaded features from {wav_file}")
            except Exception as e:
                print(f"Error loading {wav_file}: {str(e)}")

    def _spectral_flatness(self, spectrum):
        """Calculate spectral flatness (Wiener entropy)"""
        spectrum = spectrum + 1e-10  # Avoid log(0)
        geometric_mean = np.exp(np.mean(np.log(spectrum)))
        arithmetic_mean = np.mean(spectrum)
        return geometric_mean / arithmetic_mean

    def _temporal_consistency(self, samples):
        """Calculate temporal consistency using overlapping windows"""
        window_size = len(samples) // 4
        windows = np.array([samples[i:i+window_size] for i in range(0, len(samples)-window_size, window_size//2)])
        rms_values = np.sqrt(np.mean(windows**2, axis=1))
        return np.std(rms_values) / (np.mean(rms_values) + 1e-10)

    def _spectral_rolloff(self, spectrum, freqs, percentile=0.85):
        """Calculate frequency below which percentile of the spectrum's energy is contained"""
        cumsum = np.cumsum(spectrum)
        threshold = percentile * cumsum[-1]
        rolloff_index = np.where(cumsum >= threshold)[0][0]
        return freqs[rolloff_index]

    def _spectral_bandwidth(self, spectrum, freqs):
        """Calculate the bandwidth of the spectrum"""
        centroid = np.average(freqs, weights=spectrum)
        bandwidth = np.sqrt(np.average((freqs - centroid)**2, weights=spectrum))
        return bandwidth

    def extract_features(self, samples):
        """Extract features from audio samples"""
        if len(samples) == 0:
            return None
            
        # Convert to numpy array and normalize
        samples = np.array(samples, dtype=float)
        max_abs = np.max(np.abs(samples))
        if max_abs > 0:
            samples = samples / max_abs
        
        # Check if amplitude is too low
        if np.max(np.abs(samples)) < self.MIN_AMPLITUDE:
            return None
            
        # Apply Hanning window
        windowed = samples * signal.windows.hann(len(samples))
        
        # Compute FFT
        fft_data = fft(windowed)
        fft_freq = np.fft.fftfreq(len(samples), 1/self.RATE)
        
        # Get positive frequencies
        positive_freq_mask = fft_freq > 0
        fft_data = np.abs(fft_data[positive_freq_mask])
        fft_freq = fft_freq[positive_freq_mask]
        
        # Frequency band analysis
        low_freq = (fft_freq >= 100) & (fft_freq <= 500)
        mid_freq = (fft_freq > 500) & (fft_freq <= 1000)
        high_freq = (fft_freq > 1000) & (fft_freq <= 2000)
        
        # Extract features with error handling
        try:
            features = {
                'low_freq_power': np.mean(fft_data[low_freq]) if any(low_freq) else 0,
                'mid_freq_power': np.mean(fft_data[mid_freq]) if any(mid_freq) else 0,
                'high_freq_power': np.mean(fft_data[high_freq]) if any(high_freq) else 0,
                'freq_ratio_low_mid': (np.mean(fft_data[low_freq]) / (np.mean(fft_data[mid_freq]) + 1e-10)) if any(low_freq) and any(mid_freq) else 0,
                'freq_ratio_mid_high': (np.mean(fft_data[mid_freq]) / (np.mean(fft_data[high_freq]) + 1e-10)) if any(mid_freq) and any(high_freq) else 0,
                'spectral_centroid': np.average(fft_freq, weights=np.abs(fft_data)),
                'spectral_flatness': self._spectral_flatness(fft_data),
                'temporal_consistency': self._temporal_consistency(samples),
                'zero_crossing_rate': np.mean(np.abs(np.diff(np.signbit(samples)))),
                'spectral_bandwidth': self._spectral_bandwidth(fft_data, fft_freq)
            }
            return features
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None

    def compare_with_references(self, features, threshold=0.8):
        """Compare current features with reference features"""
        if not self.reference_features or not features:
            return False, None
            
        feature_weights = {
            'low_freq_power': 1.5,
            'mid_freq_power': 1.2,
            'high_freq_power': 1.0,
            'freq_ratio_low_mid': 1.8,
            'freq_ratio_mid_high': 1.8,
            'spectral_centroid': 1.3,
            'spectral_flatness': 1.4,
            'temporal_consistency': 1.6,
            'zero_crossing_rate': 1.2,
            'spectral_bandwidth': 1.2
        }
        
        best_match = None
        best_similarity = 0
        
        try:
            for filename, ref_features in self.reference_features.items():
                weighted_scores = []
                total_weight = 0
                
                for key, weight in feature_weights.items():
                    if key in features and key in ref_features:
                        ref_val = ref_features[key]
                        curr_val = features[key]
                        similarity = 1 - min(abs(ref_val - curr_val) / (max(abs(ref_val), 1e-10)), 1)
                        weighted_scores.append(similarity * weight)
                        total_weight += weight
                
                if total_weight > 0:
                    average_similarity = sum(weighted_scores) / total_weight
                    if average_similarity > best_similarity:
                        best_similarity = average_similarity
                        best_match = filename
            
            current_time = time.time()
            is_match = best_similarity > threshold
            
            if is_match:
                self.consecutive_matches += 1
                if self.last_match_time is not None and current_time - self.last_match_time > self.MATCH_TIMEOUT:
                    self.consecutive_matches = 1
            else:
                self.consecutive_matches = 0
            
            self.last_match_time = current_time
            
            final_match = self.consecutive_matches >= self.REQUIRED_MATCHES
            if final_match:
                self.consecutive_matches = 0
            
            return final_match, best_match
            
        except Exception as e:
            print(f"Error in comparison: {str(e)}")
            return False, None

    def check_light_status(self):
        """Monitor LDR and check if light has been on too long"""
        current_time = time.time()
        light_level = self.ldr_chan.voltage
        #print("light level " + str(light_level))

        if light_level > self.LIGHT_THRESHOLD:
            if self.light_start_time is None:
                self.light_start_time = current_time
            elif current_time - self.light_start_time > self.LIGHT_ALERT_THRESHOLD:
                print("Light is open")
                self.light_start_time = current_time
        else:
            self.light_start_time = None

    def monitor_realtime(self):
        """Monitor real-time audio and light with improved detection"""
        if not self.reference_features:
            print("No reference sounds found! Please record some sounds first.")
            return
            
        print("\nMonitoring for water sounds and light... (Press Ctrl+C to stop)")
        buffer = []
        
        try:
            while True:
                # Read sound sample
                voltage = self.sound_chan.voltage
                sample = int((voltage / 3.3) * 32767)
                buffer.append(sample)
                
                # Check light status
                self.check_light_status()
                
                # Process sound when buffer is full
                if len(buffer) >= self.WINDOW_SIZE:
                    features = self.extract_features(buffer)
                    
                    if features:
                        is_match, matching_file = self.compare_with_references(features)
                        if is_match:
                            print(f"Water sound detected! Similar to {matching_file}")
                    
                    # Use 75% overlap for smooth detection
                    buffer = buffer[self.WINDOW_SIZE//4:]
                
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\nStopping monitoring...")
        except Exception as e:
            print(f"Error in monitoring: {str(e)}")

def main():
    try:
        comparer = WaterSoundComparer()
        comparer.monitor_realtime()
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()

