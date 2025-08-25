# YouTube Video Processing Pipelines

This repository provides a comprehensive suite of reproducible analysis pipelines for processing and extracting features from YouTube videos for downstream scientific research. This repository includes modules for audio, visual, motion, and linguistic analysis.

This is part of a larger project, "YouTube Video Clasification", headed by [David Wegmann](https://orcid.org/0000-0002-7372-9850), under the ARTS Social Media Influence project of DATALAB – Center for Digital Social Research, Aarhus University. The processed YouTube videos are from participants in the [Data donation as a method for investigating trends and challenges in digital media landscapes at national scale](https://norden.diva-portal.org/smash/record.jsf?pid=diva2%3A1954799&dswid=9605) Project. The project investigates how digital platforms influence public discourse and develops ethical, legally compliant methods for collecting and processing user-contributed data.

## Project Structure

```
YouTube-Video-Processing-Pipelines/
├── Audio/
│   ├── Get_1st_minute.ipynb         # Extract first minute of audio
│   └── Audio_Analysis.ipynb         # Audio feature extraction
├── Visual/
│   ├── visual_features_csv_combiner.ipynb
│   └── video_analysis_utils.py      # Visual analysis utilities
├── Motion/
│   └── Merge_Sub-DataFrames.ipynb   # Motion data consolidation
├── Linguistic/
│   ├── CPU_VM_folder/               # Transcription workspace
│   ├── Language_Detection_Script.ipynb
│   ├── Text_Descriptors.ipynb
│   └── Validation_of_Transcription.ipynb
└── Metadata/
    └── create_video_metadata_df.ipynb
```

## Features

### Audio Processing
- First-minute extraction from videos
- Audio feature analysis including:
  - Volume contour analysis
  - Frequency characteristics
  - Temporal features
  - Advanced audio metrics (MFCC, ZCR)

### Visual Analysis
- Color analysis
- Texture features
- Composition metrics
- Object detection
- Face and person detection

### Motion Analysis
- Distributed processing support
- Optical flow analysis
- Scene detection
- Motion direction statistics
- Shot analysis

### Linguistic Processing
- Language detection (24 languages)
- Video transcription using Whisper
- Text feature extraction
- Multi-language support
- Validation tools

### Metadata Management
- Video metadata extraction
- Structured DataFrame creation
- Data validation
- CSV export functionality

## Prerequisites

Required packages include:
- Jupyter
- OpenAI Whisper
- OpenCV
- spaCy (with language models)
- FFmpeg
- textdescriptives
- pandas
- numpy
- scikit-image
- librosa

## Usage

Each pipeline is contained in its respective Jupyter notebook. Follow these steps:

1. **Metadata Extraction**
```python
jupyter notebook Metadata/create_video_metadata_df.ipynb
```

2. **Language Detection**
```python
jupyter notebook Linguistic/Language_Detection_Script.ipynb
```

3. **Transcription Processing**
Navigate to CPU_VM_folder and run the transcription pipeline.

4. **Feature Extraction**
Run the respective notebooks for audio, visual, and motion analysis.

## Data Flow

1. Start with video metadata extraction
2. Detect languages and filter for high confidence
3. Process transcriptions
4. Extract audio/visual/motion features
5. Combine results in final analysis


# AU-DATALAB

DATALAB – Center for Digital Social Research is an interdisciplinary research center at the School of Communication and Culture. The center is based on the vision that technology and data systems should maintain a focus on people and society, supporting the principles of democracy, human rights and ethics.


All research and activities of the center is focusing on three contemporary challenges facing the digital society, that is the challenge of 1) preserving conditions for privacy, autonomy and trust among individuals and groups; 2) sustaining the provision of and access to high-quality content online to safeguard democracy; and 3) maintaining a suitable and meaningful balance between algorithmic and human control in connection with automation.

<p align="center">
  <img width="460" src="https://github.com/AU-DATALAB/AU-DATALAB/blob/main/images/Datalab_logo_blue_transparent.png">
</p>

For more information, visit [DATALAB's website](https://datalab.au.dk/).