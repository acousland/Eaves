#!/usr/bin/env python3
"""
Real-time Audio Transcription from BlackHole
============================================

This application captures audio from BlackHole (or any audio device) and 
transcribes it in real-time using OpenAI's Whisper model with GPU acceleration.

Requirements:
- BlackHole installed on macOS
- Python packages: pyaudio, speech_recognition, openai-whisper, torch
- GPU support (Metal on macOS, CUDA on other platforms)
"""

import os
import sys
from contextlib import contextmanager

# Suppress progress bars and verbose output
os.environ['WHISPER_SUPPRESS_PROGRESS'] = '1'

import pyaudio
import speech_recognition as sr
import whisper
import numpy as np
import threading
import queue
import time
from typing import Optional
import torch
from collections import deque


@contextmanager
def suppress_output():
    """Context manager to suppress stdout during model operations."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class AudioTranscriber:
    def __init__(self, device_name: str = "BlackHole 2ch", model_size: str = "tiny"):
        """
        Initialize the audio transcriber with GPU acceleration.
        
        Args:
            device_name: Name of the audio device to capture from
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.device_name = device_name
        self.device_index = None
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recognizer = sr.Recognizer()
        
        # Check for GPU availability
        self.device = self._get_best_device()
        print(f"Using device: {self.device}")
        
        # Load Whisper model with GPU support
        print(f"Loading Whisper model '{model_size}' on {self.device}...")
        self.whisper_model = whisper.load_model(model_size, device=self.device)
        print("Model loaded successfully!")
        
        # Audio configuration - optimized for real-time
        self.chunk_size = 512  # Smaller chunks for lower latency
        self.sample_rate = 16000  # Match Whisper's expected rate
        self.channels = 1  # Mono for faster processing
        self.audio_format = pyaudio.paInt16
        
        # Streaming buffer for overlapping audio
        self.audio_buffer = deque(maxlen=int(self.sample_rate * 5))  # 5-second rolling buffer
        self.last_transcription_time = 0
        self.min_speech_duration = 0.5  # Minimum speech duration to process
        self.silence_threshold = 0.01
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self._find_audio_device()
    
    def _get_best_device(self):
        """Determine the best device for Whisper processing."""
        if torch.backends.mps.is_available():
            # Test if MPS works with Whisper (there are known compatibility issues)
            try:
                # Quick test to see if MPS works with current Whisper version
                test_model = whisper.load_model("tiny", device="mps")
                del test_model
                torch.mps.empty_cache()
                return "mps"
            except Exception as e:
                print(f"⚠️  MPS detected but incompatible with Whisper: {type(e).__name__}")
                print("   Falling back to CPU (still very fast with tiny model)")
                return "cpu"
        elif torch.cuda.is_available():
            # NVIDIA CUDA
            return "cuda"
        else:
            # Fallback to CPU
            return "cpu"
    
    def _find_audio_device(self):
        """Find the specified audio device."""
        print("Available audio devices:")
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            print(f"  {i}: {device_info['name']} (inputs: {device_info['maxInputChannels']})")
            
            if self.device_name.lower() in device_info['name'].lower():
                if device_info['maxInputChannels'] > 0:
                    self.device_index = i
                    print(f"Selected device: {device_info['name']}")
                    break
        
        if self.device_index is None:
            print(f"Warning: Could not find device '{self.device_name}'. Using default input device.")
            self.device_index = self.audio.get_default_input_device_info()['index']
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream."""
        # Convert to numpy array and add to buffer
        audio_chunk = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Convert stereo to mono if needed
        if len(audio_chunk) > frame_count:
            audio_chunk = audio_chunk.reshape(-1, 2).mean(axis=1)
        
        # Add to rolling buffer
        self.audio_buffer.extend(audio_chunk)
        
        return (in_data, pyaudio.paContinue)
    
    def start_recording(self):
        """Start recording audio from the specified device."""
        try:
            self.stream = self.audio.open(
                format=self.audio_format,
                channels=2,  # Input as stereo, convert to mono in callback
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            self.stream.start_stream()
            print(f"Started recording from device index {self.device_index}")
            
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            return False
        
        return True
    
    def stop_recording(self):
        """Stop recording audio."""
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("Stopped recording")
    
    def _detect_voice_activity(self, audio_data: np.ndarray, threshold: float = 0.01) -> bool:
        """Simple voice activity detection based on energy."""
        if len(audio_data) == 0:
            return False
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms > threshold
    
    def _get_audio_for_transcription(self, duration: float = 2.0) -> Optional[np.ndarray]:
        """Get the most recent audio data for transcription."""
        if len(self.audio_buffer) < int(self.sample_rate * self.min_speech_duration):
            return None
        
        # Get the most recent duration seconds of audio
        sample_count = int(self.sample_rate * duration)
        audio_data = np.array(list(self.audio_buffer)[-sample_count:])
        
        return audio_data
    
    def _collect_audio_chunk(self, duration: float = 3.0) -> Optional[np.ndarray]:
        """
        Collect audio data for a specified duration.
        
        Args:
            duration: Duration in seconds to collect audio
            
        Returns:
            Numpy array of audio samples or None if no audio collected
        """
        audio_data = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                audio_data.append(chunk)
            except queue.Empty:
                continue
        
        if not audio_data:
            return None
        
        # Convert to numpy array
        audio_bytes = b''.join(audio_data)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Convert to float32 and normalize
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        # If stereo, convert to mono by averaging channels
        if self.channels == 2:
            audio_float = audio_float.reshape(-1, 2).mean(axis=1)
        
        return audio_float
    
    def transcribe_chunk(self, audio_data: np.ndarray) -> str:
        """
        Transcribe a chunk of audio data using Whisper with GPU acceleration.
        
        Args:
            audio_data: Numpy array of audio samples
            
        Returns:
            Transcribed text
        """
        try:
            if len(audio_data) == 0:
                return ""
            
            # Pad or trim to expected length for consistency
            expected_length = int(self.sample_rate * 2.0)  # 2 seconds
            if len(audio_data) < expected_length:
                audio_data = np.pad(audio_data, (0, expected_length - len(audio_data)))
            elif len(audio_data) > expected_length:
                audio_data = audio_data[-expected_length:]
            
            # Transcribe with Whisper (completely suppress output)
            with suppress_output():
                result = self.whisper_model.transcribe(
                    audio_data, 
                    fp16=self.device != "cpu",  # Use fp16 for GPU acceleration
                    language='en',  # Specify language for faster processing
                    task='transcribe',
                    verbose=None,  # Completely disable verbose output
                    beam_size=1,  # Faster inference with greedy decoding
                    temperature=0,  # Deterministic output for faster processing
                    no_speech_threshold=0.6,  # Skip processing obvious non-speech
                    word_timestamps=False,  # Disable word-level timestamps for speed
                    condition_on_previous_text=False  # Don't use previous context for speed
                )
            return result["text"].strip()
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""
    
    def run_transcription(self, chunk_duration: float = 1.0):
        """
        Run the main transcription loop with streaming processing.
        
        Args:
            chunk_duration: How often to attempt transcription (seconds)
        """
        print(f"Starting real-time transcription...")
        print(f"Processing every {chunk_duration} seconds")
        print(f"Using {self.sample_rate}Hz, mono audio on {self.device} for optimal performance")
        print("Press Ctrl+C to stop\n")
        
        last_text = ""
        transcription_interval = chunk_duration
        
        try:
            while self.is_recording:
                current_time = time.time()
                
                # Only transcribe at intervals to avoid overwhelming the system
                if current_time - self.last_transcription_time >= transcription_interval:
                    # Get recent audio data
                    audio_chunk = self._get_audio_for_transcription(2.0)  # Use 2-second chunks
                    
                    if audio_chunk is not None:
                        # Check for voice activity
                        if self._detect_voice_activity(audio_chunk, self.silence_threshold):
                            print("🎤", end=" ", flush=True)
                            
                            # Transcribe the chunk
                            transcription = self.transcribe_chunk(audio_chunk)
                            
                            if transcription and transcription != last_text:
                                timestamp = time.strftime("%H:%M:%S")
                                print(f"\r[{timestamp}] {transcription}")
                                last_text = transcription
                            else:
                                print("\r", end="")
                        else:
                            print(".", end="", flush=True)  # Activity indicator for silence
                    
                    self.last_transcription_time = current_time
                
                time.sleep(0.05)  # Small delay to prevent excessive CPU usage
                
        except KeyboardInterrupt:
            print("\nStopping transcription...")
        except Exception as e:
            print(f"Error in transcription loop: {e}")


def main():
    """Main function to run the audio transcriber."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time audio transcription from BlackHole")
    parser.add_argument("--device", default="BlackHole 2ch", 
                       help="Audio device name to capture from")
    parser.add_argument("--model", default="tiny", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size (tiny recommended for real-time)")
    parser.add_argument("--chunk-duration", type=float, default=1.0,
                       help="How often to attempt transcription (seconds)")
    
    args = parser.parse_args()
    
    # Warn about non-optimal settings
    if args.model != "tiny":
        print(f"⚠️  Warning: '{args.model}' model may be too slow for real-time transcription.")
        print("   Consider using 'tiny' for best real-time performance.")
    
    # Create and configure transcriber
    transcriber = AudioTranscriber(device_name=args.device, model_size=args.model)
    
    # Start recording
    if not transcriber.start_recording():
        print("Failed to start recording. Exiting.")
        return
    
    try:
        # Run transcription
        transcriber.run_transcription(chunk_duration=args.chunk_duration)
    finally:
        # Clean up
        transcriber.stop_recording()


if __name__ == "__main__":
    main()
