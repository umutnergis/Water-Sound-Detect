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

# Su sesi karşılaştırma sınıfı
class WaterSoundComparer:
    def __init__(self):
        # I2C veri yolunu başlat
        self.i2c = busio.I2C(board.SCL, board.SDA)
        
        # ADS1115 analog-dijital dönüştürücüyü başlat
        self.ads = ADS.ADS1115(self.i2c)
        self.ads.gain = 1
        self.ads.data_rate = 860
        
        # MAX9814 ses sensörünü A0'a ve LDR'yi A1'e bağla
        self.sound_chan = AnalogIn(self.ads, ADS.P0)
        self.ldr_chan = AnalogIn(self.ads, ADS.P1)
        
        # Ses parametreleri
        self.RATE = 860
        self.CHANNELS = 1
        self.WINDOW_SIZE = 1720  # Daha iyi analiz için 2 saniyelik pencere
        
        # Ses algılama parametreleri
        self.MIN_AMPLITUDE = 0.1
        self.consecutive_matches = 0
        self.last_match_time = None
        self.MATCH_TIMEOUT = 10  # saniye
        self.REQUIRED_MATCHES = 20

        # LDR (ışık) parametreleri
        self.light_start_time = None
        self.LIGHT_THRESHOLD = 2.4
        self.LIGHT_ALERT_THRESHOLD = 30  # saniye
        
        # Dizin ve özellikler
        self.samples_dir = "water_samples"
        self.reference_features = {}
        self.load_reference_sounds()

    def load_reference_sounds(self):
        """Kaydedilmiş su seslerini yükle ve özelliklerini çıkar"""
        if not os.path.exists(self.samples_dir):
            print("Referans ses bulunamadı! Lütfen önce bazı sesler kaydedin.")
            return

        wav_files = [f for f in os.listdir(self.samples_dir) if f.endswith('.wav')]
        if not wav_files:
            print("water_samples dizininde .wav dosyası bulunamadı!")
            return

        print("Referans sesler yükleniyor...")
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
                        print(f"{wav_file} dosyasından özellikler yüklendi")
            except Exception as e:
                print(f"{wav_file} yüklenirken hata oluştu: {str(e)}")

    def _spectral_flatness(self, spectrum):
        """Spektral düzlüğü hesapla (Wiener entropi)"""
        spectrum = spectrum + 1e-10  # log(0)'dan kaçın
        geometric_mean = np.exp(np.mean(np.log(spectrum)))
        arithmetic_mean = np.mean(spectrum)
        return geometric_mean / arithmetic_mean

    def _temporal_consistency(self, samples):
        """Örtüşen pencereler kullanarak zamansal tutarlılığı hesapla"""
        window_size = len(samples) // 4
        windows = np.array([samples[i:i+window_size] for i in range(0, len(samples)-window_size, window_size//2)])
        rms_values = np.sqrt(np.mean(windows**2, axis=1))
        return np.std(rms_values) / (np.mean(rms_values) + 1e-10)

    def _spectral_rolloff(self, spectrum, freqs, percentile=0.85):
        """Spektrumun enerjisinin yüzdelik diliminin altında kaldığı frekansı hesapla"""
        cumsum = np.cumsum(spectrum)
        threshold = percentile * cumsum[-1]
        rolloff_index = np.where(cumsum >= threshold)[0][0]
        return freqs[rolloff_index]

    def _spectral_bandwidth(self, spectrum, freqs):
        """Spektrumun bant genişliğini hesapla"""
        centroid = np.average(freqs, weights=spectrum)
        bandwidth = np.sqrt(np.average((freqs - centroid)**2, weights=spectrum))
        return bandwidth

    def extract_features(self, samples):
        """Ses örneklerinden özellikleri çıkar"""
        if len(samples) == 0:
            return None
            
        # Numpy dizisine dönüştür ve normalize et
        samples = np.array(samples, dtype=float)
        max_abs = np.max(np.abs(samples))
        if max_abs > 0:
            samples = samples / max_abs
        
        # Genlik çok düşükse kontrol et
        if np.max(np.abs(samples)) < self.MIN_AMPLITUDE:
            return None
            
        # Hanning penceresi uygula
        windowed = samples * signal.windows.hann(len(samples))
        
        # FFT hesapla
        fft_data = fft(windowed)
        fft_freq = np.fft.fftfreq(len(samples), 1/self.RATE)
        
        # Pozitif frekansları al
        positive_freq_mask = fft_freq > 0
        fft_data = np.abs(fft_data[positive_freq_mask])
        fft_freq = fft_freq[positive_freq_mask]
        
        # Frekans bandı analizi
        low_freq = (fft_freq >= 100) & (fft_freq <= 500)
        mid_freq = (fft_freq > 500) & (fft_freq <= 1000)
        high_freq = (fft_freq > 1000) & (fft_freq <= 2000)
        
        # Özellikleri çıkar ve hata kontrolü yap
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
            print(f"Özellik çıkarma hatası: {str(e)}")
            return None

    def compare_with_references(self, features, threshold=0.8):
        """Mevcut özellikleri referans özelliklerle karşılaştır"""
        if not self.reference_features or not features:
            return False, None
            
        # Özellik ağırlıkları
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
            print(f"Karşılaştırma hatası: {str(e)}")
            return False, None

    def check_light_status(self):
        """LDR'yi izle ve ışığın çok uzun süre açık kalıp kalmadığını kontrol et"""
        current_time = time.time()
        light_level = self.ldr_chan.voltage

        if light_level > self.LIGHT_THRESHOLD:
            if self.light_start_time is None:
                self.light_start_time = current_time
            elif current_time - self.light_start_time > self.LIGHT_ALERT_THRESHOLD:
                print("Işık açık")
                self.light_start_time = current_time
        else:
            self.light_start_time = None

    def monitor_realtime(self):
        """Gerçek zamanlı ses ve ışık izleme"""
        if not self.reference_features:
            print("Referans ses bulunamadı! Lütfen önce bazı sesler kaydedin.")
            return
            
        print("\nSu sesleri ve ışık izleniyor... (Durdurmak için Ctrl+C'ye basın)")
        buffer = []
        
        try:
            while True:
                # Ses örneği oku
                voltage = self.sound_chan.voltage
                sample = int((voltage / 3.3) * 32767)
                buffer.append(sample)
                
                # Işık durumunu kontrol et
                self.check_light_status()
                
                # Tampon dolduğunda sesi işle
                if len(buffer) >= self.WINDOW_SIZE:
                    features = self.extract_features(buffer)
                    
                    if features:
                        is_match, matching_file = self.compare_with_references(features)
                        if is_match:
                            print(f"Su sesi tespit edildi! {matching_file} ile benzer")
                    
                    # %75 örtüşme ile pürüzsüz algılama
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
        print(f"Ana program hatası: {str(e)}")

if __name__ == "__main__":
    main()