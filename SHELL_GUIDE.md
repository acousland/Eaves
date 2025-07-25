# Shell Script Guide - run.sh

## 🚀 **Quick Start**

The `run.sh` script provides a convenient way to run Eaves without remembering Python commands or virtual environment paths.

### **Make it executable** (one-time setup):
```bash
chmod +x run.sh
```

### **Basic usage**:
```bash
./run.sh                    # Interactive menu
./run.sh start             # Quick start with optimal settings
./run.sh help              # Show all options
```

## 📋 **Available Commands**

### **Interactive Mode**
```bash
./run.sh                   # Default: shows the interactive menu
./run.sh menu              # Explicitly show menu
```

### **Direct Transcription**
```bash
./run.sh start                                    # Optimal defaults
./run.sh start --model base                       # Use base model
./run.sh start --device "MacBook Pro Microphone"  # Different device
./run.sh start --duration 2.0                     # 2-second chunks
```

### **Testing Commands**
```bash
./run.sh test              # Run setup tests
./run.sh test-blackhole    # Test BlackHole audio capture
./run.sh test-gpu          # Test GPU performance
```

### **Setup & Installation**
```bash
./run.sh install          # Install dependencies and setup virtual environment
```

## ⚙️ **Command Options**

### **Transcription Parameters**
- `--device DEVICE`: Audio device name (default: "BlackHole 2ch")
- `--model MODEL`: Whisper model size (tiny/base/small/medium/large, default: tiny)
- `--duration SECONDS`: Audio chunk duration (default: 1.0)

### **Examples**
```bash
# High accuracy, slower processing
./run.sh start --model base --duration 2.0

# Use microphone instead of BlackHole
./run.sh start --device "MacBook Pro Microphone"

# Maximum speed settings
./run.sh start --model tiny --duration 0.5
```

## 🎨 **Features**

### **Colored Output**
- 🔴 **Red**: Errors and warnings
- 🟢 **Green**: Success messages
- 🟡 **Yellow**: Progress and setup
- 🔵 **Blue**: Information
- 🟣 **Purple**: Special notices
- 🔵 **Cyan**: Headers and titles

### **Smart Environment Detection**
- Automatically finds and uses the virtual environment
- Checks for required dependencies
- Provides helpful error messages if setup is incomplete

### **Cross-platform Support**
- Works on macOS (primary target)
- Detects macOS and offers Homebrew integration
- Graceful fallbacks for other platforms

## 🛠️ **Setup Process**

### **First-time setup**:
```bash
./run.sh install
```

This will:
1. Create Python virtual environment (`.venv/`)
2. Install all required packages from `requirements.txt`
3. Install PortAudio via Homebrew (macOS)
4. Provide next-step guidance

### **Manual setup** (if you prefer):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install portaudio  # macOS only
```

## 🔍 **Troubleshooting**

### **Common Issues**

1. **"Virtual environment not found"**:
   ```bash
   ./run.sh install
   ```

2. **"Permission denied"**:
   ```bash
   chmod +x run.sh
   ```

3. **Audio device issues**:
   ```bash
   ./run.sh test              # Check available devices
   ./run.sh test-blackhole    # Test BlackHole specifically
   ```

4. **Performance issues**:
   ```bash
   ./run.sh test-gpu          # Check GPU/CPU performance
   ```

### **Verbose Testing**
```bash
./run.sh test              # All tests
./run.sh test-blackhole    # Audio capture test
./run.sh test-gpu          # Performance benchmark
```

## 📁 **File Structure**

The script expects this project structure:
```
Eaves/
├── run.sh                 # This shell script
├── run.py                 # Python launcher
├── audio_transcriber.py   # Main transcriber
├── test_setup.py          # Setup tests
├── test_blackhole.py      # BlackHole tests
├── test_gpu.py           # GPU performance tests
├── requirements.txt       # Python dependencies
└── .venv/                # Virtual environment (auto-created)
```

## 🚀 **Pro Tips**

### **Quick Commands**
```bash
# Add to your shell profile for global access
echo 'alias eaves="cd /path/to/Eaves && ./run.sh"' >> ~/.zshrc

# Then use anywhere:
eaves start
eaves test
```

### **Background Transcription**
```bash
# Run in background (not recommended for real-time viewing)
./run.sh start &

# Better: use screen or tmux for persistent sessions
screen -S eaves ./run.sh start
```

### **Logging Output**
```bash
# Save transcription to file
./run.sh start 2>&1 | tee transcription.log

# Only save transcription text (filter out system messages)
./run.sh start 2>/dev/null | grep "^\[" > transcription.txt
```

## 🎯 **Best Practices**

1. **Always test first**: `./run.sh test` before production use
2. **Use optimal settings**: Default `tiny` model with 1.0s chunks for real-time
3. **Monitor performance**: Check `./run.sh test-gpu` occasionally
4. **Keep updated**: Re-run `./run.sh install` after updates

The shell script makes Eaves much more accessible and provides a professional command-line interface for all operations!
