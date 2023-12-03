# Project Title

Tugstugi whisper infer w/ & w/o punctuations

## Overview

This project includes scripts to work with the Tugstugi Whisper dataset for Bengali automatic speech recognition (ASR). It features two main functionalities: downloading the dataset, and performing ASR inference with or without punctuation models.

## Scripts

### 1. Download Tugstugi Checkpoint

**File**: `stt_all/tugtushi_whisper/download_tugtushi_ckpt_kaggle.py`

**Purpose**: This script is used to download the checkpoint from the Tugstugi Kaggle dataset using the `opendatasets` library.

**Usage**:
```python
import opendatasets as od
od.download("https://www.kaggle.com/datasets/tugstugi/bengali-ai-asr-submission")
```

### 2. ASR Inference with Punctuation
**File**: `stt_all/tugtushi_whisper/infer_tugstugi_asr_punctuation.py`

**Purpose**: Performs ASR inference using the Tugstugi-trained Whisper medium model. It also utilizes Tugstugi punctuation models to add punctuation to the sentences post-inference.

**Features**:

- Support for batch inference.
- Ability to fix repetition problems in the transcribed text.
- Ability to punctuate the sentence. 
- Outputs the results as a CSV file.
- Output File: `./output/infer_w_punctuation.csv`
- CSV Format: Each row contains an id and the corresponding sentence with punctuation.

### 3. ASR Inference without Punctuation
**File**: `stt_all/tugtushi_whisper/infer_tugstugi_asr.py`

**Purpose**: Performs ASR inference using the Tugstugi-trained Whisper medium model, but without using Tugstugi punctuation models.

**Features**:

- Batch inference capability.
- Repetition correction in the transcribed text.
- Outputs the results as a CSV file.
- Output File: infer.csv
- CSV Format: Each row consists of an id and the sentence without punctuation.

### Installation & Run
- Install packages from `requrement.txt`
- Run `download_tugtushi_ckpt_kaggle.py` to download the weights 
- Run `infer_tugstugi_asr.py` to infer tugstugi model
- Run `infer_tugstugi_asr_punctuation.py` to infer tugstugi model with punctuations
