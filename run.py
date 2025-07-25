#!/usr/bin/env python3
"""
Simple launcher for the Eaves audio transcriber.
"""

import subprocess
import sys
import os


def main():
    """Launch the audio transcriber with a user-friendly interface."""
    print("🎧 Eaves - Real-time Audio Transcription")
    print("=" * 40)
    
    # Check if we're in a virtual environment
    venv_python = "/Users/acousland/Documents/Code/Eaves/.venv/bin/python"
    if os.path.exists(venv_python):
        python_cmd = venv_python
        print("✓ Using virtual environment")
    else:
        python_cmd = sys.executable
        print("⚠️  Virtual environment not found, using system Python")
    
    print("\nChoose an option:")
    print("1. Test setup and list audio devices")
    print("2. Test BlackHole system audio capture")
    print("3. Test GPU performance and model speed")
    print("4. Start transcription with BlackHole 2ch (default)")
    print("5. Start transcription with speaker diarization (advanced)")
    print("6. Start transcription with simple speaker diarization")
    print("7. Start transcription with custom settings")
    print("8. Exit")
    
    try:
        choice = input("\nEnter choice (1-8): ").strip()
        
        if choice == "1":
            print("\nRunning setup test...")
            subprocess.run([python_cmd, "test_setup.py"])
            
        elif choice == "2":
            print("\nTesting BlackHole system audio capture...")
            print("Make sure you've configured BlackHole as system output!")
            subprocess.run([python_cmd, "test_blackhole.py"])
            
        elif choice == "3":
            print("\nTesting GPU performance and model speed...")
            subprocess.run([python_cmd, "test_gpu.py"])
            
        elif choice == "4":
            print("\nStarting transcription with BlackHole 2ch...")
            print("Using optimized settings for real-time performance")
            print("Press Ctrl+C to stop transcription")
            subprocess.run([python_cmd, "audio_transcriber.py", "--model", "tiny", "--chunk-duration", "1.0"])
            
        elif choice == "5":
            print("\nStarting transcription with speaker diarization (advanced)...")
            print("👥 Different speakers will be shown in different colors")
            print("⚠️  Note: Advanced diarization may be slower than regular transcription")
            print("Press Ctrl+C to stop transcription")
            subprocess.run([python_cmd, "audio_transcriber_diarization.py", "--model", "tiny", "--chunk-duration", "1.5"])
            
        elif choice == "6":
            print("\nStarting transcription with simple speaker diarization...")
            print("👥 Different speakers will be shown in different colors")
            print("🚀 Uses lightweight clustering for speaker identification")
            print("Press Ctrl+C to stop transcription")
            subprocess.run([python_cmd, "audio_transcriber_simple_diarization.py", "--model", "tiny"])
            
        elif choice == "7":
            device = input("Enter audio device name (or press Enter for BlackHole 2ch): ").strip()
            if not device:
                device = "BlackHole 2ch"
                
            model = input("Enter model size (tiny/base/small/medium/large) [tiny]: ").strip()
            if not model:
                model = "tiny"  # Changed default to tiny for real-time
                
            duration = input("Enter chunk duration in seconds [1.0]: ").strip()
            if not duration:
                duration = "1.0"  # Reduced default for faster response
            
            print(f"\nStarting transcription...")
            print(f"Device: {device}")
            print(f"Model: {model}")
            print(f"Chunk duration: {duration}s")
            print("Press Ctrl+C to stop transcription")
            
            cmd = [
                python_cmd, "audio_transcriber.py",
                "--device", device,
                "--model", model,
                "--chunk-duration", duration
            ]
            subprocess.run(cmd)
            
        elif choice == "8":
            print("Goodbye!")
            return
            
        else:
            print("Invalid choice. Please enter 1-8.")
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
