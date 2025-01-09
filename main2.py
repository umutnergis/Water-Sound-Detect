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
        # I2C veri yolunu başlat
        self.i2c = busio.I2C(board.SCL, board.SDA)
        
        # ADS1115 analog-dijital dönüştürücüyü başlat
        self.ads = ADS.ADS1115(self.i2c)
        self.ads.gain = 1
        self.ads.data_rate = 860
        
        # Sensör kanalları
        self.sound_chan = AnalogIn(self.ads, ADS.P0)
        self.ldr_chan = AnalogIn(self.ads, ADS.P1)
        
        # Ses parametreleri
        self.RATE = 860
        self.CHANNELS = 1
        self.WINDOW_SIZE = 1720
        
        # Geliştirilmiş ses algılama parametreleri
        self.MIN_AMPLITUDE = 0.15  # Arttırıldı
        self.consecutive_matches = 0
        self.last_match_time = None
        self.MATCH_TIMEOUT = 8  # Azaltıldı
        self.REQUIRED_MATCHES = 30  # Arttırıldı

        # LDR parametreleri
        self.light_start_time = None
        self.LIGHT_THRESHOLD = 2.4
        self.LIGHT_ALERT_THRESHOLD = 30
        
        # Referans ses yönetimi
        self.samples_dir = "water_samples"
        self.reference_features = {}
        self.load_reference_sounds()

        # Su sesi karakteristik frekans bantları
        self.WATER_FREQ_BANDS = {
            'low': (200, 400),
            'mid': (400, 800),
            'high': (800, 1600)
        }

    def load_reference_sounds(self):
        """Referans su seslerini yükle"""
        if not os.path.exists(self.samples_dir):
            print("Referans ses dizini bulunamadı!")
            return

        wav_files = [f for f in os.listdir(self.samples_dir) if f.endswith('.wav')]
        if not wav_files:
            print("Su sesi örnekleri bulunamadı!")
            return

        print("Referans su sesleri yükleniyor...")
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
                        print(f"{wav_file} özellikleri yüklendi")
            except Exception as e:
                print(f"Hata: {wav_file} yüklenemedi - {str(e)}")

    def _spectral_flatness(self, spectrum):
        """Spektral düzlük hesaplama (Wiener entropi)"""
        spectrum = spectrum + 1e-10
        geometric_mean = np.exp(np.mean(np.log(spectrum)))
        arithmetic_mean = np.mean(spectrum)
        return geometric_mean / arithmetic_mean

    def _temporal_consistency(self, samples):
        """Zamansal tutarlılık analizi"""
        window_size = len(samples) // 4
        windows = np.array([samples[i:i+window_size] for i in range(0, len(samples)-window_size, window_size//2)])
        rms_values = np.sqrt(np.mean(windows**2, axis=1))
        return np.std(rms_values) / (np.mean(rms_values) + 1e-10)

    def _water_sound_signature(self, spectrum, freqs):
        """Su sesi karakteristiklerini kontrol et"""
        # Spektral yumuşaklık
        spectral_smoothness = np.mean(np.abs(np.diff(spectrum)))
        
        # Frekans bantlarındaki enerji
        energy_200_800 = np.mean(spectrum[(freqs >= 200) & (freqs <= 800)])
        energy_above_1000 = np.mean(spectrum[freqs > 1000])
        
        # Su sesi enerji oranı
        energy_ratio = energy_200_800 / (energy_above_1000 + 1e-10)
        
        return spectral_smoothness, energy_ratio

    def _check_amplitude_consistency(self, samples):
        """Genlik tutarlılığını kontrol et"""
        segments = np.array_split(samples, 8)
        rms_values = [np.sqrt(np.mean(segment**2)) for segment in segments]
        variation = np.std(rms_values) / (np.mean(rms_values) + 1e-10)
        return variation < 0.5

    def _calculate_band_energy(self, spectrum, freqs, band):
        """Belirli frekans bandındaki enerjiyi hesapla"""
        mask = (freqs >= band[0]) & (freqs <= band[1])
        return np.mean(spectrum[mask]) if any(mask) else 0

    def extract_features(self, samples):
        """Geliştirilmiş özellik çıkarma"""
        if len(samples) == 0:
            return None
            
        # Normalize et
        samples = np.array(samples, dtype=float)
        max_abs = np.max(np.abs(samples))
        if max_abs > 0:
            samples = samples / max_abs
        
        # Minimum genlik kontrolü
        if np.max(np.abs(samples)) < self.MIN_AMPLITUDE:
            return None
            
        # Pencere uygula ve FFT hesapla
        windowed = samples * signal.windows.hann(len(samples))
        fft_data = fft(windowed)
        fft_freq = np.fft.fftfreq(len(samples), 1/self.RATE)
        
        # Pozitif frekansları al
        positive_freq_mask = fft_freq > 0
        fft_data = np.abs(fft_data[positive_freq_mask])
        fft_freq = fft_freq[positive_freq_mask]
        
        try:
            # Su sesi karakteristiklerini hesapla
            smoothness, energy_ratio = self._water_sound_signature(fft_data, fft_freq)
            amplitude_consistent = self._check_amplitude_consistency(samples)
            
            # Frekans bantlarını analiz et
            features = {
                'low_freq_power': self._calculate_band_energy(fft_data, fft_freq, self.WATER_FREQ_BANDS['low']),
                'mid_freq_power': self._calculate_band_energy(fft_data, fft_freq, self.WATER_FREQ_BANDS['mid']),
                'high_freq_power': self._calculate_band_energy(fft_data, fft_freq, self.WATER_FREQ_BANDS['high']),
                'freq_ratio_low_mid': (self._calculate_band_energy(fft_data, fft_freq, self.WATER_FREQ_BANDS['low']) /
                                     (self._calculate_band_energy(fft_data, fft_freq, self.WATER_FREQ_BANDS['mid']) + 1e-10)),
                'freq_ratio_mid_high': (self._calculate_band_energy(fft_data, fft_freq, self.WATER_FREQ_BANDS['mid']) /
                                      (self._calculate_band_energy(fft_data, fft_freq, self.WATER_FREQ_BANDS['high']) + 1e-10)),
                'spectral_flatness': self._spectral_flatness(fft_data),
                'temporal_consistency': self._temporal_consistency(samples),
                'spectral_smoothness': smoothness,
                'water_energy_ratio': energy_ratio,
                'amplitude_consistency': 1.0 if amplitude_consistent else 0.0
            }
            return features
            
        except Exception as e:
            print(f"Özellik çıkarma hatası: {str(e)}")
            return None

    def compare_with_references(self, features, threshold=0.9):  # Eşik değeri arttırıldı
        """Geliştirilmiş özellik karşılaştırma"""
        if not self.reference_features or not features:
            return False, None
            
        # Su sesine özel ağırlıklar
        feature_weights = {
            'low_freq_power': 2.0,
            'mid_freq_power': 1.8,
            'high_freq_power': 0.8,
            'freq_ratio_low_mid': 2.0,
            'freq_ratio_mid_high': 1.5,
            'spectral_flatness': 1.8,
            'temporal_consistency': 2.0,
            'spectral_smoothness': 1.8,
            'water_energy_ratio': 2.0,
            'amplitude_consistency': 1.5
        }
        
        try:
            best_match = None
            best_similarity = 0
            
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
                if self.last_match_time is not None and current_time - self.last_match_time > self.MATCH_TIMEOUT:
                    self.consecutive_matches = 1
                else:
                    self.consecutive_matches += 1
            else:
                self.consecutive_matches = 0
            
            self.last_match_time = current_time
            
            final_match = self.consecutive_matches >= self.REQUIRED_MATCHES
            if final_match:
                self.consecutive_matches = 0
            
            return final_match, best_match
            
        except Exception as e:
            print(f"Karşılaştırma hatası: {str(e)}")
            return False, None

    def check_light_status(self):
        """Işık durumu kontrolü"""
        current_time = time.time()
        light_level = self.ldr_chan.voltage

        if light_level > self.LIGHT_THRESHOLD:
            if self.light_start_time is None:
                self.light_start_time = current_time
            elif current_time - self.light_start_time > self.LIGHT_ALERT_THRESHOLD:
                print("Işık uzun süredir açık!")
                self.light_start_time = current_time
        else:
            self.light_start_time = None

    def monitor_realtime(self):
        """Gerçek zamanlı izleme"""
        if not self.reference_features:
            print("Önce referans su sesleri yüklenmelidir!")
            return
            
        print("\nSu sesleri ve ışık izleniyor... (Ctrl+C ile durdurun)")
        buffer = []
        
        try:
            while True:
                voltage = self.sound_chan.voltage
                sample = int((voltage / 3.3) * 32767)
                buffer.append(sample)
                
                self.check_light_status()
                
                if len(buffer) >= self.WINDOW_SIZE:
                    features = self.extract_features(buffer)
                    
                    if features:
                        is_match, matching_file = self.compare_with_references(features)
                        if is_match:
                            print(f"Su sesi algılandı! Eşleşen dosya: {matching_file}")
                    
                    buffer = buffer[self.WINDOW_SIZE//4:]
                
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\nİzleme durduruluyor...")
        except Exception as e:
            print(f"İzleme hatası: {str(e)}")

def main():
    try:
        comparer = WaterSoundComparer()
        comparer.monitor_realtime()
    except Exception as e:
        print(f"Program hatası: {str(e)}")

if __name__ == "__main__":
    main()