#!/bin/bash

# Eaves - Real-time Audio Transcription Runner
# A convenient shell script to run the Eaves transcriber

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

# Function to print colored output
print_color() {
    printf "${1}${2}${NC}\n"
}

# Function to check if virtual environment exists
check_venv() {
    if [[ -f "$VENV_PYTHON" ]]; then
        return 0
    else
        return 1
    fi
}

# Function to show help
show_help() {
    print_color $CYAN "🎧 Eaves - Real-time Audio Transcription"
    echo "=========================================="
    echo ""
    echo "Usage: ./run.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  menu                     Show interactive menu (default)"
    echo "  start                   Start transcription with default settings"
    echo "  start-diarization       Start transcription with advanced speaker diarization"
    echo "  start-simple-diarization Start transcription with simple speaker diarization"
    echo "  test                    Run setup tests"
    echo "  test-blackhole          Test BlackHole audio capture"
    echo "  test-gpu                Test GPU performance"
    echo "  install                 Install/setup the project"
    echo "  help                    Show this help message"
    echo ""
    echo "Transcription Options:"
    echo "  --device DEVICE     Audio device name (default: 'BlackHole 2ch')"
    echo "  --model MODEL       Whisper model (tiny/base/small/medium/large, default: tiny)"
    echo "  --duration SECONDS  Chunk duration in seconds (default: 1.0)"
    echo ""
    echo "Examples:"
    echo "  ./run.sh                                    # Show interactive menu"
    echo "  ./run.sh start                             # Quick start with defaults"
    echo "  ./run.sh start-diarization                 # Start with advanced speaker ID"
    echo "  ./run.sh start-simple-diarization          # Start with simple speaker ID"
    echo "  ./run.sh start --model base                # Use base model"
    echo "  ./run.sh start --device 'MacBook Pro Microphone'"
    echo "  ./run.sh test                              # Run all setup tests"
    echo ""
}

# Function to install/setup the project
install_project() {
    print_color $YELLOW "🔧 Setting up Eaves project..."
    
    # Check if Python 3 is available
    if ! command -v python3 &> /dev/null; then
        print_color $RED "❌ Python 3 is required but not installed."
        exit 1
    fi
    
    # Create virtual environment if it doesn't exist
    if ! check_venv; then
        print_color $YELLOW "📦 Creating virtual environment..."
        python3 -m venv "$PROJECT_DIR/.venv"
    fi
    
    # Install dependencies
    print_color $YELLOW "📥 Installing dependencies..."
    "$VENV_PYTHON" -m pip install --upgrade pip
    "$VENV_PYTHON" -m pip install -r "$PROJECT_DIR/requirements.txt"
    
    # Install PortAudio if on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            print_color $YELLOW "🍺 Installing PortAudio via Homebrew..."
            brew install portaudio || true
        else
            print_color $YELLOW "⚠️  Consider installing Homebrew and PortAudio for better audio support"
        fi
    fi
    
    print_color $GREEN "✅ Installation complete!"
    echo ""
    print_color $CYAN "Next steps:"
    echo "1. Configure BlackHole for system audio (see README.md)"
    echo "2. Run: ./run.sh test"
    echo "3. Run: ./run.sh start"
}

# Function to run the interactive menu
run_menu() {
    if ! check_venv; then
        print_color $RED "❌ Virtual environment not found. Run: ./run.sh install"
        exit 1
    fi
    
    cd "$PROJECT_DIR"
    "$VENV_PYTHON" run.py
}

# Function to start transcription
start_transcription() {
    if ! check_venv; then
        print_color $RED "❌ Virtual environment not found. Run: ./run.sh install"
        exit 1
    fi
    
    # Default values
    device="BlackHole 2ch"
    model="tiny"
    duration="1.0"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --device)
                device="$2"
                shift 2
                ;;
            --model)
                model="$2"
                shift 2
                ;;
            --duration)
                duration="$2"
                shift 2
                ;;
            *)
                print_color $RED "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    print_color $GREEN "🎤 Starting Eaves transcription..."
    print_color $BLUE "Device: $device"
    print_color $BLUE "Model: $model"
    print_color $BLUE "Duration: ${duration}s"
    echo ""
    print_color $YELLOW "Press Ctrl+C to stop transcription"
    echo ""
    
    cd "$PROJECT_DIR"
    "$VENV_PYTHON" audio_transcriber.py --device "$device" --model "$model" --chunk-duration "$duration"
}

# Function to start transcription with diarization
start_diarization() {
    if ! check_venv; then
        print_color $RED "❌ Virtual environment not found. Run: ./run.sh install"
        exit 1
    fi
    
    # Default values
    device="BlackHole 2ch"
    model="tiny"
    duration="1.5"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --device)
                device="$2"
                shift 2
                ;;
            --model)
                model="$2"
                shift 2
                ;;
            --duration)
                duration="$2"
                shift 2
                ;;
            *)
                print_color $RED "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    print_color $GREEN "🎤 Starting Eaves transcription with speaker diarization..."
    print_color $PURPLE "👥 Different speakers will be shown in different colors"
    print_color $BLUE "Device: $device"
    print_color $BLUE "Model: $model"
    print_color $BLUE "Duration: ${duration}s"
    print_color $YELLOW "⚠️  Note: Diarization may be slower than regular transcription"
    echo ""
    print_color $YELLOW "Press Ctrl+C to stop transcription"
    echo ""
    
    cd "$PROJECT_DIR"
    "$VENV_PYTHON" audio_transcriber_diarization.py --device "$device" --model "$model" --chunk-duration "$duration"
}

# Function to start transcription with simple diarization
start_simple_diarization() {
    if ! check_venv; then
        print_color $RED "❌ Virtual environment not found. Run: ./run.sh install"
        exit 1
    fi
    
    # Default values
    model="tiny"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                model="$2"
                shift 2
                ;;
            *)
                print_color $RED "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    print_color $GREEN "🎤 Starting Eaves transcription with simple speaker diarization..."
    print_color $PURPLE "👥 Different speakers will be shown in different colors"
    print_color $CYAN "🚀 Uses lightweight clustering for speaker identification"
    print_color $BLUE "Model: $model"
    echo ""
    print_color $YELLOW "Press Ctrl+C to stop transcription"
    echo ""
    
    cd "$PROJECT_DIR"
    "$VENV_PYTHON" audio_transcriber_simple_diarization.py --model "$model"
}

# Function to run tests
run_test() {
    if ! check_venv; then
        print_color $RED "❌ Virtual environment not found. Run: ./run.sh install"
        exit 1
    fi
    
    cd "$PROJECT_DIR"
    
    case "${1:-all}" in
        "all")
            print_color $CYAN "🧪 Running setup test..."
            "$VENV_PYTHON" test_setup.py
            ;;
        "blackhole")
            print_color $CYAN "🔊 Testing BlackHole audio capture..."
            print_color $YELLOW "Make sure BlackHole is configured as system output!"
            "$VENV_PYTHON" test_blackhole.py
            ;;
        "gpu")
            print_color $CYAN "🚀 Testing GPU performance..."
            "$VENV_PYTHON" test_gpu.py
            ;;
        *)
            print_color $RED "Unknown test: $1"
            echo "Available tests: all, blackhole, gpu"
            exit 1
            ;;
    esac
}

# Main script logic
main() {
    cd "$PROJECT_DIR"
    
    # If no arguments, show menu
    if [[ $# -eq 0 ]]; then
        run_menu
        return
    fi
    
    # Parse main command
    case "$1" in
        "help" | "-h" | "--help")
            show_help
            ;;
        "install")
            install_project
            ;;
        "menu")
            run_menu
            ;;
        "start")
            shift
            start_transcription "$@"
            ;;
        "start-diarization")
            shift
            start_diarization "$@"
            ;;
        "start-simple-diarization")
            shift
            start_simple_diarization "$@"
            ;;
        "test")
            run_test "${2:-all}"
            ;;
        "test-blackhole")
            run_test "blackhole"
            ;;
        "test-gpu")
            run_test "gpu"
            ;;
        *)
            print_color $RED "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
