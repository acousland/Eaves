#!/usr/bin/env python3
"""
Test script to verify the audio setup and list available devices.
"""

import pyaudio
import sys


def list_audio_devices():
    """List all available audio input devices."""
    audio = pyaudio.PyAudio()
    
    print("Available Audio Devices:")
    print("=" * 50)
    
    blackhole_devices = []
    input_devices = []
    
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        
        if device_info['maxInputChannels'] > 0:
            device_name = device_info['name']
            input_channels = device_info['maxInputChannels']
            sample_rate = int(device_info['defaultSampleRate'])
            
            print(f"  {i}: {device_name}")
            print(f"      Input Channels: {input_channels}")
            print(f"      Sample Rate: {sample_rate} Hz")
            print()
            
            input_devices.append((i, device_name))
            
            if 'blackhole' in device_name.lower():
                blackhole_devices.append((i, device_name))
    
    audio.terminate()
    
    print("\nSummary:")
    print(f"Found {len(input_devices)} input devices")
    
    if blackhole_devices:
        print(f"Found {len(blackhole_devices)} BlackHole devices:")
        for idx, name in blackhole_devices:
            print(f"  - Device {idx}: {name}")
        print("\nTo use BlackHole, run:")
        device_name = blackhole_devices[0][1]
        print(f'  python audio_transcriber.py --device "{device_name}"')
    else:
        print("No BlackHole devices found.")
        print("Make sure BlackHole is installed and configured.")
        print("You can still use other input devices:")
        for idx, name in input_devices[:3]:  # Show first 3 devices
            print(f'  python audio_transcriber.py --device "{name}"')


def test_whisper():
    """Test if Whisper can be imported and loaded."""
    try:
        import whisper
        print("\nTesting Whisper model loading...")
        model = whisper.load_model("tiny")
        print("✓ Whisper 'tiny' model loaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Error loading Whisper: {e}")
        return False


def main():
    print("Eaves Audio Setup Test")
    print("=" * 30)
    
    try:
        list_audio_devices()
        whisper_ok = test_whisper()
        
        print("\nSetup Status:")
        print("✓ PyAudio working")
        print("✓ Audio devices detected" if True else "✗ No audio devices found")
        print("✓ Whisper working" if whisper_ok else "✗ Whisper not working")
        
        if whisper_ok:
            print("\n🎉 Setup appears to be working! You can now run the main transcriber.")
        else:
            print("\n⚠️  There may be issues with the Whisper installation.")
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install required packages with: pip install -r requirements.txt")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
