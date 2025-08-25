# CPU Virtual Machine Processing Folder

## Purpose
This folder serves as a workspace for processing video transcriptions using OpenAI's Whisper model. It's specifically designed to handle CPU-based transcription tasks in a structured and resumable manner.

### Folder Structure
```
CPU_VM_folder/
├── Non_Transcribed_Videos/     # Videos pending transcription
├── Cleanup.ipynb              # Script for removing processed videos
├── Transcription_Pipeline.ipynb # Main transcription processing script
└── about.md                   # This documentation file
```

### Process Flow

1. **Video Management**
   - The `Non_Transcribed_Videos` folder maintains a queue of videos awaiting transcription
   - Videos are moved here from the main download directory
   - Only videos with confirmed language detection (confidence > 0.98) are included

2. **Transcription Process**
   - Videos are processed sequentially using Whisper (Transcription_Pipeline.ipynb)
   - Each successful transcription is saved to `Transcription.csv`
   - Progress is tracked through the pipeline

3. **State Management**
   - Cleanup.ipynb handles removal of processed videos
   - Progress tracking through CSV file comparison
   - Prevents duplicate transcriptions through file checks

### Maintenance Rules
1. Only add videos that:
   - Have passed language detection
   - Have confidence scores > 0.98
   - Are not already transcribed
2. Run Cleanup.ipynb regularly to remove processed videos
3. Keep folder synchronized with main processing pipeline
4. Monitor `Transcription.csv` for completion status

### Note
This folder is temporary and should be emptied once all videos have been successfully transcribed and validated.