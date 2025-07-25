# Real-time Performance Optimizations

## 🚀 What's Been Optimized

### 1. **Audio Processing**
- **Reduced sample rate**: 16kHz (from 44.1kHz) - matches Whisper's native rate
- **Mono audio**: Converted from stereo for faster processing
- **Smaller chunk size**: 512 samples (from 1024) for lower latency
- **Streaming buffer**: Rolling 5-second buffer for continuous processing

### 2. **Whisper Model Optimizations**
- **Default model**: Changed to "tiny" (fastest available)
- **GPU acceleration**: Automatic detection and use of Mac GPU (MPS) when compatible
- **Optimized inference settings**:
  - `beam_size=1` (greedy decoding for speed)
  - `temperature=0` (deterministic output)
  - `language='en'` (skip language detection)
  - `fp16=True` (when using GPU)
  - `no_speech_threshold=0.6` (skip obvious non-speech)

### 3. **Processing Pipeline**
- **Faster chunk duration**: 1.0 seconds (from 3.0 seconds)
- **Voice Activity Detection**: Only process audio with speech
- **Overlapping chunks**: Continuous 2-second analysis windows
- **Reduced CPU overhead**: Optimized sleep intervals and buffer management

## 📊 Performance Results

From our testing:
- **Tiny model on CPU**: 1.4x faster than real-time (✅ Real-time capable)
- **Base model on CPU**: 0.7x real-time (⏰ Too slow for real-time)

## 🔧 GPU Acceleration Status

### Current Status
- **MPS (Mac GPU)**: Detected but incompatible with current Whisper version
- **Fallback**: Using optimized CPU processing (still real-time capable)
- **Future**: Will automatically use GPU when compatibility is fixed

### Why CPU is Still Great
- The "tiny" model is highly optimized for CPU
- 1.4x real-time performance means minimal latency
- Lower power consumption than GPU processing
- More stable and reliable

## 🎯 Usage Recommendations

### For Best Real-time Performance:
```bash
python audio_transcriber.py --model tiny --chunk-duration 1.0
```

### Performance vs Accuracy Trade-offs:
- **tiny**: Fastest, good accuracy for clear speech
- **base**: 2x slower, better accuracy for difficult audio
- **small/medium/large**: Too slow for real-time

### Audio Quality Tips:
1. **Use clean audio sources** - tiny model works best with clear speech
2. **Set appropriate volume** - not too loud (distortion) or too quiet
3. **Minimize background noise** - helps with voice activity detection

## 🔍 Monitoring Performance

Use the built-in indicators:
- `🎤` - Currently processing speech
- `.` - Silence detected (not processing)
- Timestamps show when transcription completes

## 🛠️ Advanced Configuration

### For Lower Latency (Trade-off: May miss words):
```bash
python audio_transcriber.py --chunk-duration 0.5
```

### For Better Accuracy (Trade-off: Higher latency):
```bash
python audio_transcriber.py --model base --chunk-duration 2.0
```

## 🚨 Troubleshooting Performance

### If transcription is laggy:
1. Check system resources (Activity Monitor)
2. Close other audio applications
3. Use tiny model
4. Increase chunk duration

### If words are being cut off:
1. Increase chunk duration to 1.5-2.0 seconds
2. Check microphone/BlackHole levels
3. Verify BlackHole configuration

## 📈 Future Improvements

### When GPU support is fixed:
- Expect 2-3x speed improvement
- Ability to use larger models in real-time
- Lower CPU usage

### Potential optimizations:
- WebRTC VAD for better voice detection
- Streaming inference (process audio as it arrives)
- Custom Whisper compilation for macOS optimization
