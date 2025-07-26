# Eaves - Real-time Audio Transcription

Eaves is a Python application that captures audio from BlackHole (or any audio device) and transcribes it in real-time using OpenAI's Whisper model.

## Features

- Real-time audio capture from BlackHole or any audio input device
- Speech-to-text transcription using OpenAI Whisper
- **NEW: Speaker diarization** - identify different speakers with color-coded output
- Support for multiple Whisper model sizes (tiny, base, small, medium, large)
- Configurable audio chunk processing duration
- Silence detection to avoid processing empty audio
- Timestamped transcription output
- Clean output without loading bars or progress indicators

## Prerequisites

### BlackHole Installation

1. Download and install BlackHole from: https://existential.audio/blackhole/
2. Configure BlackHole as an audio device in your macOS Audio MIDI Setup
3. Route the audio you want to transcribe through BlackHole

### System Requirements

- macOS (tested on macOS with BlackHole)
- Python 3.8 or higher
- Microphone or audio input device access

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd Eaves
   ```

2. **Easy setup with shell script**:
   ```bash
   chmod +x run.sh
   ./run.sh install
   ```

3. **Manual setup** (alternative):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   pip install -r requirements.txt
   ```

4. **Setup authentication for advanced diarization** (optional):
   ```bash
   cp config_private.ini.template config_private.ini
   # Edit config_private.ini and add your Hugging Face token
   ```

## Usage

### **Quick Start with Shell Script** (Recommended)

```bash
# Interactive menu
./run.sh

# Quick start with optimal settings
./run.sh start

# Start with speaker diarization (identifies different speakers)
./run.sh start-simple-diarization

# Test your setup
./run.sh test

# See all options
./run.sh help
```

### **Speaker Diarization** 🎯

Eaves can identify different speakers and display their transcriptions in different colors:

```bash
# Simple speaker diarization (recommended)
./run.sh start-simple-diarization

# Via Python menu
python run.py  # Choose option 6
```

**Output example:**
```
[Speaker 1] Hello, how are you today?
[Speaker 2] I'm doing great, thanks for asking!
[Speaker 1] That's wonderful to hear.
```

See [DIARIZATION.md](DIARIZATION.md) for detailed information about speaker identification features.

### **Direct Python Usage**

```bash
# Basic usage
python audio_transcriber.py

# Advanced options
python audio_transcriber.py --device "BlackHole 2ch" --model tiny --chunk-duration 1.0
```

### **Shell Script Commands**

The `run.sh` script provides convenient commands:

```bash
./run.sh start                                    # Quick start with defaults
./run.sh start --model base                       # Use base model
./run.sh start --device "MacBook Pro Microphone"  # Use different device
./run.sh test                                     # Run setup tests
./run.sh test-blackhole                          # Test BlackHole audio
./run.sh test-gpu                                # Test GPU performance
```

### Command Line Options

- `--device`: Audio device name to capture from (default: "BlackHole 2ch")
- `--model`: Whisper model size - tiny, base, small, medium, large (default: "base")
- `--chunk-duration`: Duration of audio chunks to process in seconds (default: 3.0)

### Finding Your Audio Device

The application will list all available audio devices when it starts. Look for your BlackHole device in the list, or use any other input device you prefer.

## BlackHole Setup for System Audio

To transcribe system audio (like video calls, music, etc.):

1. Open **Audio MIDI Setup** (Applications > Utilities)
2. Create a **Multi-Output Device**:
   - Include your speakers/headphones AND BlackHole
   - This allows you to hear audio while also capturing it
3. Set the Multi-Output Device as your system output in System Preferences > Sound
4. Run Eaves targeting BlackHole as the input device

## Performance Notes

- **Model Size vs Speed**: 
  - `tiny`: Fastest, 1.4x real-time performance ⚡ (recommended)
  - `base`: Good balance, slower than real-time ⚠️
  - `small`: Better accuracy, much slower ❌
  - `medium`: High accuracy, very slow ❌
  - `large`: Best accuracy, extremely slow ❌

- **Chunk Duration**: Shorter chunks (1.0s) give faster response but may cut off words. Longer chunks (2-3s) are more accurate but have higher latency.

- **GPU Acceleration**: Automatic detection of Mac GPU (MPS). Falls back to optimized CPU processing if incompatible.

- **Real-time Performance**: The `tiny` model achieves 1.4x real-time speed, meaning minimal latency for live transcription.

## Troubleshooting

### Audio Device Issues

If you can't find BlackHole:
1. Ensure BlackHole is properly installed
2. Check Audio MIDI Setup to verify BlackHole appears
3. Try running with `--device "BlackHole"` (without channel specification)

### Permission Issues

If you get microphone permission errors:
1. Go to System Preferences > Security & Privacy > Privacy > Microphone
2. Add Terminal or your Python IDE to the allowed applications

### Performance Issues

- Start with the `tiny` model for testing
- Increase chunk duration if transcription is choppy
- Close other audio applications that might interfere

## Output

The application will display:
- Available audio devices on startup
- Real-time transcription with timestamps
- Activity indicators during processing
- Silence detection (shows dots for quiet periods)

Example output:
```
[14:23:15] Hello, this is a test of the transcription system.
[14:23:18] The audio quality is quite good through BlackHole.
....
[14:23:25] Now I'm speaking again after a pause.
```

## License

This project is licensed under the terms specified in the LICENSE file.