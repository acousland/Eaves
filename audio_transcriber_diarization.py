#!/usr/bin/env python3
"""
Enhanced Audio Transcriber with Speaker Diarization
==================================================

This enhanced version includes speaker diarization to identify different speakers
in addition to transcribing their speech.

Features:
- Real-time speech transcription
- Speaker diarization (who is speaking when)
- Color-coded output for different speakers
- Speaker change detection
"""

import os
import sys
from contextlib import contextmanager
import tempfile
import wave

# Suppress progress bars and verbose output
os.environ['WHISPER_SUPPRESS_PROGRESS'] = '1'

import pyaudio
import speech_recognition as sr
import whisper
import numpy as np
import threading
import queue
import time
from typing import Optional, Dict, List, Tuple
import torch
from collections import deque
from pyannote.audio import Pipeline
import warnings

# Suppress pyannote warnings
warnings.filterwarnings("ignore", category=UserWarning)


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


class EnhancedAudioTranscriber:
    def __init__(self, device_name: str = "BlackHole 2ch", model_size: str = "tiny", enable_diarization: bool = True):
        """
        Initialize the enhanced audio transcriber with diarization.
        
        Args:
            device_name: Name of the audio device to capture from
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            enable_diarization: Whether to enable speaker diarization
        """
        self.device_name = device_name
        self.device_index = None
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recognizer = sr.Recognizer()
        self.enable_diarization = enable_diarization
        
        # Check for GPU availability
        self.device = self._get_best_device()
        print(f"Using device: {self.device}")
        
        # Load Whisper model with GPU support
        print(f"Loading Whisper model '{model_size}' on {self.device}...")
        self.whisper_model = whisper.load_model(model_size, device=self.device)
        print("✅ Whisper model loaded successfully!")
        
        # Initialize diarization pipeline
        self.diarization_pipeline = None
        if enable_diarization:
            self._init_diarization()
        
        # Audio configuration - optimized for real-time
        self.chunk_size = 512
        self.sample_rate = 16000
        self.channels = 1
        self.audio_format = pyaudio.paInt16
        
        # Streaming buffer for overlapping audio
        self.audio_buffer = deque(maxlen=int(self.sample_rate * 8))  # 8-second rolling buffer
        self.last_transcription_time = 0
        self.min_speech_duration = 0.5
        self.silence_threshold = 0.01
        
        # Speaker tracking
        self.current_speaker = None
        self.speaker_colors = {}
        self.available_colors = [
            '\033[91m',  # Red
            '\033[92m',  # Green
            '\033[94m',  # Blue
            '\033[95m',  # Magenta
            '\033[96m',  # Cyan
            '\033[93m',  # Yellow
        ]
        self.color_index = 0
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self._find_audio_device()
    
    def _get_best_device(self):
        """Determine the best device for Whisper processing."""
        if torch.backends.mps.is_available():
            try:
                test_model = whisper.load_model("tiny", device="mps")
                del test_model
                torch.mps.empty_cache()
                return "mps"
            except Exception as e:
                print(f"⚠️  MPS detected but incompatible with Whisper: {type(e).__name__}")
                print("   Falling back to CPU (still very fast with tiny model)")
                return "cpu"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _init_diarization(self):
        """Initialize the speaker diarization pipeline."""
        try:
            print("Loading speaker diarization model...")
            # Use the pre-trained diarization pipeline
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=None  # You may need a HuggingFace token for some models
            )
            
            if self.device == "cuda":
                self.diarization_pipeline = self.diarization_pipeline.to(torch.device("cuda"))
            
            print("✅ Diarization model loaded successfully!")
        except Exception as e:
            print(f"⚠️  Could not load diarization model: {e}")
            print("   Continuing without speaker diarization...")
            self.enable_diarization = False
    
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
        
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms > threshold
    
    def _get_audio_for_transcription(self, duration: float = 3.0) -> Optional[np.ndarray]:
        """Get the most recent audio data for transcription."""
        if len(self.audio_buffer) < int(self.sample_rate * self.min_speech_duration):
            return None
        
        sample_count = int(self.sample_rate * duration)
        audio_data = np.array(list(self.audio_buffer)[-sample_count:])
        
        return audio_data
    
    def _save_audio_to_temp_file(self, audio_data: np.ndarray) -> str:
        """Save audio data to a temporary WAV file for diarization."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        
        # Convert to 16-bit PCM
        audio_16bit = (audio_data * 32767).astype(np.int16)
        
        with wave.open(temp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_16bit.tobytes())
        
        return temp_file.name
    
    def _perform_diarization(self, audio_file: str) -> List[Tuple[float, float, str]]:
        """Perform speaker diarization on audio file."""
        try:
            # Apply diarization
            diarization = self.diarization_pipeline(audio_file)
            
            # Extract speaker segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append((turn.start, turn.end, speaker))
            
            return segments
        except Exception as e:
            print(f"Diarization error: {e}")
            return []
    
    def _get_speaker_color(self, speaker: str) -> str:
        """Get a consistent color for a speaker."""
        if speaker not in self.speaker_colors:
            if self.color_index < len(self.available_colors):
                self.speaker_colors[speaker] = self.available_colors[self.color_index]
                self.color_index += 1
            else:
                # Cycle through colors if we have more speakers than colors
                self.speaker_colors[speaker] = self.available_colors[self.color_index % len(self.available_colors)]
                self.color_index += 1
        
        return self.speaker_colors[speaker]
    
    def _format_speaker_output(self, speaker: str, text: str, timestamp: str) -> str:
        """Format output with speaker information and colors."""
        if not self.enable_diarization or not speaker:
            return f"[{timestamp}] {text}"
        
        color = self._get_speaker_color(speaker)
        reset_color = '\033[0m'
        
        return f"{color}[{timestamp}] {speaker}:{reset_color} {text}"
    
    def transcribe_chunk(self, audio_data: np.ndarray) -> Tuple[str, Optional[str]]:
        """
        Transcribe a chunk of audio data and identify the speaker.
        
        Returns:
            Tuple of (transcribed_text, speaker_id)
        """
        try:
            if len(audio_data) == 0:
                return "", None
            
            # Pad or trim to expected length
            expected_length = int(self.sample_rate * 3.0)  # 3 seconds for diarization
            if len(audio_data) < expected_length:
                audio_data = np.pad(audio_data, (0, expected_length - len(audio_data)))
            elif len(audio_data) > expected_length:
                audio_data = audio_data[-expected_length:]
            
            speaker = None
            
            # Perform diarization if enabled
            if self.enable_diarization and self.diarization_pipeline:
                temp_file = self._save_audio_to_temp_file(audio_data)
                try:
                    segments = self._perform_diarization(temp_file)
                    if segments:
                        # Get the dominant speaker in the chunk
                        speaker_durations = {}
                        for start, end, spk in segments:
                            duration = end - start
                            speaker_durations[spk] = speaker_durations.get(spk, 0) + duration
                        
                        if speaker_durations:
                            speaker = max(speaker_durations, key=speaker_durations.get)
                finally:
                    os.unlink(temp_file)  # Clean up temp file
            
            # Transcribe with Whisper
            with suppress_output():
                result = self.whisper_model.transcribe(
                    audio_data,
                    fp16=self.device != "cpu",
                    language='en',
                    task='transcribe',
                    verbose=None,
                    beam_size=1,
                    temperature=0,
                    no_speech_threshold=0.6,
                    word_timestamps=False,
                    condition_on_previous_text=False
                )
            
            return result["text"].strip(), speaker
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return "", None
    
    def run_transcription(self, chunk_duration: float = 1.5):
        """
        Run the main transcription loop with speaker diarization.
        
        Args:
            chunk_duration: How often to attempt transcription (seconds)
        """
        diarization_status = "✅ enabled" if self.enable_diarization else "❌ disabled"
        
        print(f"Starting real-time transcription with speaker diarization...")
        print(f"Processing every {chunk_duration} seconds")
        print(f"Using {self.sample_rate}Hz, mono audio on {self.device}")
        print(f"Speaker diarization: {diarization_status}")
        if self.enable_diarization:
            print("👥 Different speakers will be shown in different colors")
        print("Press Ctrl+C to stop\n")
        
        last_text = ""
        last_speaker = None
        transcription_interval = chunk_duration
        
        try:
            while self.is_recording:
                current_time = time.time()
                
                if current_time - self.last_transcription_time >= transcription_interval:
                    # Get recent audio data (longer for better diarization)
                    audio_chunk = self._get_audio_for_transcription(3.0)
                    
                    if audio_chunk is not None:
                        if self._detect_voice_activity(audio_chunk, self.silence_threshold):
                            print("🎤", end=" ", flush=True)
                            
                            # Transcribe and identify speaker
                            transcription, speaker = self.transcribe_chunk(audio_chunk)
                            
                            if transcription and (transcription != last_text or speaker != last_speaker):
                                timestamp = time.strftime("%H:%M:%S")
                                formatted_output = self._format_speaker_output(speaker, transcription, timestamp)
                                print(f"\r{formatted_output}")
                                
                                last_text = transcription
                                last_speaker = speaker
                            else:
                                print("\r", end="")
                        else:
                            print(".", end="", flush=True)
                    
                    self.last_transcription_time = current_time
                
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\nStopping transcription...")
        except Exception as e:
            print(f"Error in transcription loop: {e}")


def main():
    """Main function to run the enhanced audio transcriber."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced real-time audio transcription with speaker diarization")
    parser.add_argument("--device", default="BlackHole 2ch", 
                       help="Audio device name to capture from")
    parser.add_argument("--model", default="tiny", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size (tiny recommended for real-time)")
    parser.add_argument("--chunk-duration", type=float, default=1.5,
                       help="How often to attempt transcription (seconds)")
    parser.add_argument("--no-diarization", action="store_true",
                       help="Disable speaker diarization")
    
    args = parser.parse_args()
    
    # Warn about performance
    if args.model != "tiny":
        print(f"⚠️  Warning: '{args.model}' model may be too slow for real-time transcription.")
    
    enable_diarization = not args.no_diarization
    
    # Create and configure transcriber
    transcriber = EnhancedAudioTranscriber(
        device_name=args.device, 
        model_size=args.model,
        enable_diarization=enable_diarization
    )
    
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
