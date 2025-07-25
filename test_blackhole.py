#!/usr/bin/env python3
"""
Test script to verify BlackHole is receiving system audio.
This will show the audio levels being received by BlackHole.
"""

import pyaudio
import numpy as np
import time
import sys


def test_blackhole_audio():
    """Test if BlackHole is receiving audio from the system."""
    # Audio configuration
    chunk_size = 1024
    sample_rate = 44100
    channels = 2
    audio_format = pyaudio.paInt16
    
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    
    # Find BlackHole device
    blackhole_index = None
    print("Looking for BlackHole device...")
    
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        if 'blackhole' in device_info['name'].lower() and device_info['maxInputChannels'] > 0:
            blackhole_index = i
            print(f"Found BlackHole: {device_info['name']}")
            break
    
    if blackhole_index is None:
        print("❌ BlackHole device not found!")
        print("Make sure BlackHole is installed and configured.")
        audio.terminate()
        return False
    
    try:
        # Open audio stream
        stream = audio.open(
            format=audio_format,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=blackhole_index,
            frames_per_buffer=chunk_size
        )
        
        print(f"\n🎧 Monitoring BlackHole audio levels...")
        print("Play some audio on your Mac to see if it's being captured.")
        print("Press Ctrl+C to stop monitoring.\n")
        
        # Monitor audio levels
        max_level = 0
        while True:
            try:
                # Read audio data
                data = stream.read(chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Calculate audio level
                level = np.max(np.abs(audio_data)) / 32768.0
                max_level = max(max_level, level)
                
                # Visual level indicator
                bar_length = 50
                filled_length = int(bar_length * level)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                # Status indicator
                if level > 0.01:
                    status = "🔊 AUDIO DETECTED"
                    color = "\033[92m"  # Green
                elif level > 0.001:
                    status = "🔉 Low audio"
                    color = "\033[93m"  # Yellow
                else:
                    status = "🔇 Silent"
                    color = "\033[91m"  # Red
                
                # Print level
                print(f"\r{color}[{bar}] {level:.3f} - {status} (max: {max_level:.3f})\033[0m", end="", flush=True)
                
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                break
                
    except Exception as e:
        print(f"❌ Error accessing BlackHole: {e}")
        return False
        
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print(f"\n\n📊 Test completed. Maximum level detected: {max_level:.3f}")
        
        if max_level > 0.01:
            print("✅ BlackHole is receiving audio from your system!")
            print("You can now run the transcriber to capture system audio.")
        elif max_level > 0.001:
            print("⚠️  Very low audio detected. Check your volume levels.")
        else:
            print("❌ No audio detected. Check your BlackHole configuration:")
            print("1. Make sure you created a Multi-Output Device")
            print("2. Set it as your system audio output")
            print("3. Play some audio to test")
        
        return max_level > 0.01


if __name__ == "__main__":
    test_blackhole_audio()
