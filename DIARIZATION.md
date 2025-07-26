# Speaker Diarization in Eaves

Eaves now supports speaker diarization - the ability to identify different speakers in audio and display their transcriptions in different colors.

## Available Diarization Methods

### 1. Simple Speaker Diarization (Recommended)

**File:** `audio_transcriber_simple_diarization.py`

**Features:**
- ✅ Lightweight clustering using MFCC and spectral features
- ✅ Real-time performance
- ✅ Color-coded speaker output
- ✅ No complex dependencies
- ✅ Works out of the box

**How it works:**
- Extracts MFCC (Mel-frequency cepstral coefficients) and spectral features from audio
- Uses DBSCAN clustering to group similar voice characteristics
- Assigns different colors to different speakers
- Updates speaker identification in real-time

**Usage:**
```bash
# Via Python launcher
python run.py  # Choose option 6

# Via shell script
./run.sh start-simple-diarization

# Direct execution
python audio_transcriber_simple_diarization.py --model tiny
```

### 2. Advanced Speaker Diarization (Experimental)

**File:** `audio_transcriber_diarization.py`

**Features:**
- 🔬 Uses pyannote.audio for advanced speaker identification
- ⚠️ Requires additional dependencies (pytorch-lightning, torch-audiomentations, etc.)
- ⚠️ May be slower than simple diarization
- ⚠️ Complex setup requirements

**Status:** Now working but requires Hugging Face authentication. If you get authentication errors, use simple diarization instead.

**Authentication Setup (if you want to use advanced diarization):**
1. Visit https://hf.co/pyannote/speaker-diarization-3.1
2. Accept the user conditions  
3. Create a token at https://hf.co/settings/tokens
4. Copy `config_private.ini.template` to `config_private.ini`
5. Replace `YOUR_HF_TOKEN_HERE` with your actual token

## Color Coding

Different speakers are displayed in different colors:

- **Speaker 1:** Blue
- **Speaker 2:** Green  
- **Speaker 3:** Yellow
- **Speaker 4:** Magenta
- **Speaker 5:** Cyan
- **Speaker 6:** Red
- **Speaker 7:** White
- **Speaker 8:** Dark Gray

## Performance Notes

### Simple Diarization
- **Real-time factor:** ~1.4x (faster than real-time)
- **Latency:** ~1.0 seconds
- **Memory usage:** Low
- **CPU usage:** Moderate

### Best Practices

1. **Use the tiny model** for real-time performance
2. **Ensure good audio quality** - clear separation between speakers helps
3. **Minimize background noise** for better speaker identification
4. **Allow a few seconds** for initial speaker clustering to stabilize

## Technical Details

### Feature Extraction
- **MFCC Features:** 13 coefficients capturing vocal tract characteristics
- **Spectral Features:** Centroid, rolloff, and zero-crossing rate
- **Normalization:** StandardScaler for consistent clustering

### Clustering Algorithm
- **Algorithm:** DBSCAN (Density-Based Spatial Clustering)
- **Parameters:** eps=0.8, min_samples=1
- **Advantages:** Automatically determines number of speakers, handles noise

### Audio Processing
- **Sample Rate:** 16kHz (optimized for Whisper)
- **Chunk Duration:** 1.0 seconds
- **Format:** 32-bit float, mono

## Troubleshooting

### Common Issues

1. **Authentication errors with advanced diarization**
   - The pyannote models are gated and require Hugging Face authentication
   - **Solution**: Use simple diarization (option 6) - no authentication required
   - **Alternative**: Set up HF authentication as described above

2. **All speakers show as Speaker 1**
   - Audio might be too similar between speakers
   - Try adjusting clustering parameters
   - Ensure speakers have distinct vocal characteristics

2. **Speakers change colors frequently**
   - Normal during initial clustering phase
   - Should stabilize after 10-20 seconds
   - Consider increasing chunk duration for more stable clustering

3. **Poor speaker separation**
   - Check audio quality and microphone placement
   - Reduce background noise
   - Ensure speakers speak clearly and distinctly

### Configuration

You can adjust clustering sensitivity by modifying the SimpleSpeakerDiarizer parameters:

```python
# More sensitive (more speakers detected)
diarizer = SimpleSpeakerDiarizer(eps=0.5, min_samples=1)

# Less sensitive (fewer speakers detected)  
diarizer = SimpleSpeakerDiarizer(eps=1.0, min_samples=2)
```

## Future Improvements

- [ ] Voice activity detection for better segmentation
- [ ] Speaker enrollment and recognition
- [ ] Confidence scores for speaker assignments
- [ ] Export speaker-labeled transcripts
- [ ] Real-time speaker visualization
