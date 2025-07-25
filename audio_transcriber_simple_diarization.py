#!/usr/bin/env python3
"""
Enhanced Audio Transcriber with Simple Speaker Diarization
Captures and transcribes audio from BlackHole in real-time with speaker identification.
Uses simple clustering techniques instead of complex ML models.
"""

import pyaudio
import whisper
import torch
import numpy as np
import threading
import queue
import time
import contextlib
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import librosa

# Color codes for different speakers
SPEAKER_COLORS = [
    '\033[94m',  # Blue
    '\033[92m',  # Green
    '\033[93m',  # Yellow
    '\033[95m',  # Magenta
    '\033[96m',  # Cyan
    '\033[91m',  # Red
    '\033[97m',  # White
    '\033[90m',  # Dark Gray
]
RESET_COLOR = '\033[0m'

@dataclass
class AudioSegment:
    """Represents a segment of audio with features"""
    start_time: float
    end_time: float
    audio_data: np.ndarray
    features: Optional[np.ndarray] = None
    speaker_id: Optional[int] = None
    transcription: str = ""

class SimpleAudioFeatures:
    """Extract simple audio features for speaker identification"""
    
    @staticmethod
    def extract_mfcc_features(audio_data: np.ndarray, sample_rate: int = 16000, n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCC features from audio data"""
        try:
            # Ensure audio has enough samples
            if len(audio_data) < 512:
                return np.zeros(n_mfcc)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio_data.astype(np.float32),
                sr=sample_rate,
                n_mfcc=n_mfcc,
                n_fft=512,
                hop_length=256
            )
            
            # Return mean of MFCC coefficients
            return np.mean(mfccs, axis=1)
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.zeros(n_mfcc)
    
    @staticmethod
    def extract_spectral_features(audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract spectral features from audio data"""
        try:
            if len(audio_data) < 512:
                return np.zeros(4)
            
            # Convert to float
            audio_float = audio_data.astype(np.float32)
            
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_float, sr=sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_float, sr=sample_rate)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_float)[0]
            
            # Calculate statistics
            features = np.array([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.mean(spectral_rolloff),
                np.mean(zero_crossing_rate)
            ])
            
            return features
        except Exception as e:
            print(f"Error extracting spectral features: {e}")
            return np.zeros(4)

class SimpleSpeakerDiarizer:
    """Simple speaker diarization using clustering"""
    
    def __init__(self, eps: float = 0.5, min_samples: int = 2):
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.feature_extractor = SimpleAudioFeatures()
        self.segments: List[AudioSegment] = []
        self.is_fitted = False
        
    def add_segment(self, audio_data: np.ndarray, start_time: float, end_time: float, transcription: str = "") -> AudioSegment:
        """Add a new audio segment"""
        # Extract features
        mfcc_features = self.feature_extractor.extract_mfcc_features(audio_data)
        spectral_features = self.feature_extractor.extract_spectral_features(audio_data)
        
        # Combine features
        combined_features = np.concatenate([mfcc_features, spectral_features])
        
        # Create segment
        segment = AudioSegment(
            start_time=start_time,
            end_time=end_time,
            audio_data=audio_data,
            features=combined_features,
            transcription=transcription
        )
        
        self.segments.append(segment)
        return segment
    
    def cluster_speakers(self) -> Dict[int, List[AudioSegment]]:
        """Cluster segments by speaker"""
        if len(self.segments) < 2:
            # Not enough segments to cluster
            for i, segment in enumerate(self.segments):
                segment.speaker_id = 0
            return {0: self.segments}
        
        # Extract features from all segments
        features = np.array([segment.features for segment in self.segments if segment.features is not None])
        
        if len(features) == 0:
            return {0: self.segments}
        
        # Normalize features
        if not self.is_fitted:
            features_normalized = self.scaler.fit_transform(features)
            self.is_fitted = True
        else:
            features_normalized = self.scaler.transform(features)
        
        # Perform clustering
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        cluster_labels = clustering.fit_predict(features_normalized)
        
        # Assign speaker IDs
        speakers = {}
        for i, label in enumerate(cluster_labels):
            if i < len(self.segments):
                # Handle noise points (label = -1)
                speaker_id = max(0, label)
                self.segments[i].speaker_id = speaker_id
                
                if speaker_id not in speakers:
                    speakers[speaker_id] = []
                speakers[speaker_id].append(self.segments[i])
        
        return speakers

@contextlib.contextmanager
def suppress_output():
    """Suppress stdout and stderr"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

class EnhancedAudioTranscriber:
    def __init__(self, model_name="tiny", device=None):
        self.model_name = model_name
        self.device = self._get_best_device() if device is None else device
        
        print(f"Initializing Whisper model '{model_name}' on {self.device}...")
        with suppress_output():
            self.model = whisper.load_model(model_name, device=self.device)
        
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.chunk_duration = 1.0  # seconds
        self.sample_rate = 16000
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # Speaker diarization
        self.diarizer = SimpleSpeakerDiarizer(eps=0.8, min_samples=1)
        self.speaker_history = []
        self.current_speaker = 0
        
    def _get_best_device(self):
        """Determine the best available device for processing"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS has issues with Whisper, fall back to CPU with warning
            print("⚠️  MPS detected but using CPU due to compatibility issues")
            return "cpu"
        else:
            return "cpu"
    
    def _find_blackhole_device(self):
        """Find BlackHole audio device"""
        p = pyaudio.PyAudio()
        blackhole_device = None
        
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if "BlackHole" in device_info['name'] and device_info['maxInputChannels'] > 0:
                blackhole_device = i
                print(f"Found BlackHole device: {device_info['name']} (ID: {i})")
                break
        
        p.terminate()
        
        if blackhole_device is None:
            raise RuntimeError("BlackHole audio device not found. Please install BlackHole.")
        
        return blackhole_device
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Add to queue for processing
        self.audio_queue.put((audio_data.copy(), time.time()))
        
        return (in_data, pyaudio.paContinue)
    
    def _get_speaker_color(self, speaker_id: int) -> str:
        """Get color for speaker"""
        return SPEAKER_COLORS[speaker_id % len(SPEAKER_COLORS)]
    
    def _format_speaker_output(self, speaker_id: int, text: str) -> str:
        """Format output with speaker colors"""
        color = self._get_speaker_color(speaker_id)
        return f"{color}[Speaker {speaker_id + 1}]{RESET_COLOR} {text}"
    
    def _perform_diarization(self, audio_data: np.ndarray, start_time: float, transcription: str) -> int:
        """Perform speaker diarization on audio segment"""
        if len(transcription.strip()) == 0:
            return self.current_speaker
        
        # Add segment to diarizer
        end_time = start_time + len(audio_data) / self.sample_rate
        segment = self.diarizer.add_segment(audio_data, start_time, end_time, transcription)
        
        # Perform clustering if we have enough segments
        if len(self.diarizer.segments) >= 2:
            speakers = self.diarizer.cluster_speakers()
            
            # Update current speaker based on latest segment
            if segment.speaker_id is not None:
                self.current_speaker = segment.speaker_id
        
        return self.current_speaker
    
    def transcribe_chunk(self, audio_data):
        """Transcribe a chunk of audio data"""
        try:
            # Ensure audio is the right shape and type
            if len(audio_data) == 0:
                return ""
            
            # Convert to float32 and normalize
            audio_data = audio_data.astype(np.float32)
            
            # Pad or trim to expected length
            if len(audio_data) < self.chunk_size:
                audio_data = np.pad(audio_data, (0, self.chunk_size - len(audio_data)))
            else:
                audio_data = audio_data[:self.chunk_size]
            
            # Suppress Whisper output
            with suppress_output():
                result = self.model.transcribe(
                    audio_data,
                    fp16=False,
                    language=None,
                    task="transcribe",
                    verbose=False
                )
            
            text = result["text"].strip()
            return text
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def start_transcription(self):
        """Start real-time transcription"""
        try:
            device_id = self._find_blackhole_device()
            
            p = pyaudio.PyAudio()
            
            print(f"Starting transcription with {self.model_name} model on {self.device}")
            print("Listening for audio from BlackHole...")
            print("Press Ctrl+C to stop\n")
            
            # Open audio stream
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_id,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            stream.start_stream()
            self.is_running = True
            
            # Process audio chunks
            audio_buffer = []
            buffer_duration = 0.0
            last_process_time = time.time()
            
            while self.is_running:
                try:
                    # Get audio data from queue (with timeout)
                    audio_data, timestamp = self.audio_queue.get(timeout=0.1)
                    
                    # Add to buffer
                    audio_buffer.extend(audio_data)
                    buffer_duration += len(audio_data) / self.sample_rate
                    
                    # Process when buffer reaches target duration
                    if buffer_duration >= self.chunk_duration:
                        # Convert buffer to numpy array
                        buffer_array = np.array(audio_buffer, dtype=np.float32)
                        
                        # Check if there's actual audio (not just silence)
                        if np.max(np.abs(buffer_array)) > 0.001:
                            # Transcribe
                            start_time = time.time()
                            text = self.transcribe_chunk(buffer_array)
                            
                            if text:
                                # Perform diarization
                                speaker_id = self._perform_diarization(buffer_array, timestamp, text)
                                
                                # Print with speaker identification
                                formatted_output = self._format_speaker_output(speaker_id, text)
                                print(formatted_output)
                                
                                # Calculate and display performance
                                processing_time = time.time() - start_time
                                real_time_factor = buffer_duration / processing_time
                                if real_time_factor < 1.0:
                                    print(f"⚠️  Processing slower than real-time: {real_time_factor:.1f}x")
                        
                        # Clear buffer
                        audio_buffer = []
                        buffer_duration = 0.0
                    
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    break
            
            # Cleanup
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            print(f"Error during transcription: {e}")
        finally:
            self.is_running = False
            print("\nTranscription stopped.")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time audio transcription with speaker diarization")
    parser.add_argument("--model", default="tiny", choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model to use (default: tiny)")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], 
                       help="Device to use for processing")
    
    args = parser.parse_args()
    
    try:
        transcriber = EnhancedAudioTranscriber(model_name=args.model, device=args.device)
        transcriber.start_transcription()
    except KeyboardInterrupt:
        print("\nTranscription interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
