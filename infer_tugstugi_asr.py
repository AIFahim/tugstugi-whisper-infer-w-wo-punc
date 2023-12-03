import os
import csv
import time
import glob

MODEL = './bengali-ai-asr-submission/bengali-whisper-medium/'

CHUNK_LENGTH_S = 20.1
ENABLE_BEAM = True

if ENABLE_BEAM:
    BATCH_SIZE = 4
else:
    BATCH_SIZE = 8

if len(glob.glob("./test_audios/*.wav")) > 10:
    EVAL = False
    DATASET_PATH = './test_audios/'
else:
    EVAL = True
    DATASET_PATH = './test_audios/'
    
import csv
import glob
import shutil
import librosa
import argparse
import warnings
from pathlib import Path
import transformers
print(transformers.__version__)
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

import warnings

warnings.filterwarnings("ignore")

files = list(glob.glob(DATASET_PATH + '/' + '*.wav'))
files.sort()

pipe = pipeline(task="automatic-speech-recognition",
                model=MODEL,
                tokenizer=MODEL,
                chunk_length_s=CHUNK_LENGTH_S, device=0, batch_size=BATCH_SIZE)

pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language="bn", task="transcribe")

print("model loaded!")


def fix_repetition(text, max_count):
    uniq_word_counter = {}
    words = text.split()
    for word in text.split():
        if word not in uniq_word_counter:
            uniq_word_counter[word] = 1
        else:
            uniq_word_counter[word] += 1

    for word, count in uniq_word_counter.items():
        if count > max_count:
            words = [w for w in words if w != word]
    text = " ".join(words)
    return text


if ENABLE_BEAM:
    texts = pipe(files, generate_kwargs={"max_length": 260, "num_beams": 4})
else:
    texts = pipe(files)

del pipe

predictions = []
with open("./output/infer.csv", 'wt', encoding="utf8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'sentence'])
    for f, text in zip(files, texts):
        file_id = Path(f).stem
        pred = text['text'].strip()
        pred = fix_repetition(pred, max_count=8)
        if pred[-1] not in ['।', '?', ',']:
            pred = pred + '।'
        # print(i, file_id, pred)
        prediction = [file_id, pred]
        writer.writerow(prediction)
        predictions.append(prediction)

print("inference finished!")