import os
import csv
import time
import glob

MODEL = './bengali-ai-asr-submission/bengali-whisper-medium/'

CHUNK_LENGTH_S = 20.1
ENABLE_BEAM = True

PUNCT_MODELS = [
    './bengali-ai-asr-submission/punct-model-6layers/',
    './bengali-ai-asr-submission/punct-model-8layers',
    './bengali-ai-asr-submission/punct-model-11layers/',
    './bengali-ai-asr-submission/punct-model-12layers/'
]

PUNCT_WEIGHTS = [[1.0, 1.4, 1.0, 0.8]]

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
# Need to remove device=0 part if need infer with cpu
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
import torch
models = [
    AutoModelForTokenClassification.from_pretrained(f).eval().cuda() for f in PUNCT_MODELS
]
tokenizer = AutoTokenizer.from_pretrained(PUNCT_MODELS[0])
def punctuate(text):
    input_ids = tokenizer(text).input_ids
    with torch.no_grad():
        model = models[0]
        logits = torch.nn.functional.softmax(
            model(input_ids=torch.LongTensor([input_ids]).cuda()).logits[0, 1:-1],
            dim=1).cpu()
        for model in models[1:]:
            logits += torch.nn.functional.softmax(
                model(input_ids=torch.LongTensor([input_ids]).cuda()).logits[0, 1:-1],
                dim=1).cpu()
        logits = logits / len(models)
        logits *= torch.FloatTensor(PUNCT_WEIGHTS)
        label_ids = torch.argmax(logits, dim=-1)

        tokens = tokenizer(text, add_special_tokens=False).input_ids
        punct_text = ""
        for index, token in enumerate(tokens):
            token_str = tokenizer.decode(token)
            if '##' not in token_str:
                punct_text += " " + token_str
            else:
                punct_text += token_str[2:]
            punct_text += ['', '।', ',', '?'][label_ids[index].item()]

    punct_text = punct_text.strip()
    return punct_text

predictions = []
with open("infer_w_punctuation.csv", 'wt', encoding="utf8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'sentence'])
    for f, text in zip(files, texts):
        file_id = Path(f).stem
        pred = text['text'].strip()
        pred = fix_repetition(pred, max_count=8)
        pred = punctuate(pred)
        if pred[-1] not in ['।', '?', ',']:
            pred = pred + '।'
        # print(i, file_id, pred)
        prediction = [file_id, pred]
        writer.writerow(prediction)
        predictions.append(prediction)

print("inference finished!")
