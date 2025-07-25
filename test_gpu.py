#!/usr/bin/env python3
"""
Test GPU performance and model loading speed for different Whisper models.
"""

import os
import sys
from contextlib import contextmanager

# Suppress progress bars
os.environ['WHISPER_SUPPRESS_PROGRESS'] = '1'

import time
import torch
import whisper
import numpy as np


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


def test_gpu_whisper():
    """Test Whisper model loading and inference speed on different devices."""
    print("🧪 Testing Whisper GPU Performance")
    print("=" * 40)
    
    # Check available devices
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
        print("✅ Metal Performance Shaders (MPS) available")
    if torch.cuda.is_available():
        devices.append("cuda")
        print("✅ CUDA available")
    
    print(f"Available devices: {devices}")
    print()
    
    # Test models (start with tiny for speed)
    models = ["tiny", "base"]
    
    # Create test audio (2 seconds of random audio)
    sample_rate = 16000
    test_audio = np.random.randn(sample_rate * 2).astype(np.float32) * 0.1
    
    results = {}
    
    for model_name in models:
        print(f"Testing {model_name} model:")
        
        for device in devices:
            try:
                print(f"  Loading on {device}...", end=" ")
                start_time = time.time()
                
                model = whisper.load_model(model_name, device=device)
                load_time = time.time() - start_time
                
                print(f"loaded in {load_time:.2f}s")
                
                # Test inference speed
                print(f"  Testing inference...", end=" ")
                start_time = time.time()
                
                # Do a few inferences to get average (suppress output)
                with suppress_output():
                    for _ in range(3):
                        result = model.transcribe(
                            test_audio, 
                            fp16=(device != "cpu"),
                            verbose=None,
                            beam_size=1,
                            temperature=0
                        )
                
                inference_time = (time.time() - start_time) / 3
                print(f"avg {inference_time:.2f}s per 2s audio")
                
                results[f"{model_name}_{device}"] = {
                    "load_time": load_time,
                    "inference_time": inference_time,
                    "realtime_factor": 2.0 / inference_time  # How much faster than real-time
                }
                
                # Clean up GPU memory
                del model
                if device in ["mps", "cuda"]:
                    torch.mps.empty_cache() if device == "mps" else torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"error: {e}")
        
        print()
    
    # Summary
    print("📊 Performance Summary:")
    print("-" * 50)
    for key, result in results.items():
        model, device = key.split("_")
        rt_factor = result["realtime_factor"]
        status = "🚀 Real-time capable" if rt_factor > 1.0 else "⏰ Slower than real-time"
        print(f"{model:>5} on {device:>4}: {result['inference_time']:.2f}s ({rt_factor:.1f}x) - {status}")
    
    # Recommendations
    print("\n💡 Recommendations:")
    best_realtime = max([r for r in results.values() if r["realtime_factor"] > 1.0], 
                       key=lambda x: x["realtime_factor"], default=None)
    
    if best_realtime:
        best_config = [k for k, v in results.items() if v == best_realtime][0]
        model, device = best_config.split("_")
        print(f"✅ Best real-time config: {model} model on {device}")
        print(f"   {best_realtime['realtime_factor']:.1f}x faster than real-time")
    else:
        print("⚠️  No configuration achieved real-time performance with test setup")
    
    print(f"\n🎯 For live transcription, use: --model tiny")
    if "mps" in devices:
        print("🔥 Your Mac GPU (MPS) will be used automatically for acceleration!")


if __name__ == "__main__":
    test_gpu_whisper()
