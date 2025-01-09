import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import numpy as np
import wave
import os
from datetime import datetime

class WaterSoundRecorder:
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
        self.SAMPLE_WIDTH = 2
        
        # Create directory for saving samples
        self.samples_dir = "water_samples"
        if not os.path.exists(self.samples_dir):
            os.makedirs(self.samples_dir)

    def record_water_sound(self, duration=5):
        """Su sesini dinle"""
        print(f"Recording water sound for 5 seconds...")
        samples = []
        start_time = time.time()
        
        num_samples = int(duration * self.RATE)
        for _ in range(num_samples):
            voltage = self.chan.voltage
            sample = int((voltage / 3.3) * 32767)
            samples.append(sample)
            
            elapsed = time.time() - start_time
            target_time = (len(samples) / self.RATE)
            if elapsed < target_time:
                time.sleep(target_time - elapsed)
        
        # Save the recording
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"water_sound_{timestamp}.wav"
        self.save_wav(samples, filename)
        return filename

    def save_wav(self, samples, filename):
        """Wav olarak kaydet"""
        filepath = os.path.join(self.samples_dir, filename)
        
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.SAMPLE_WIDTH)
            wf.setframerate(self.RATE)
            sample_data = np.array(samples, dtype=np.int16).tobytes()
            wf.writeframes(sample_data)
            
        print(f"Saved audio to {filepath}")

def main():
    recorder = WaterSoundRecorder()
    
    while True:
        input("\nPress Enter to record water sound (or Ctrl+C to exit)...")
        recorder.record_water_sound(duration=5)

if __name__ == "__main__":
    main()


