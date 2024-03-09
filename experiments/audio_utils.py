import os
import pandas as pd
import speech_recognition as sr
from tqdm import tqdm
tqdm.pandas()
import whisper
import wave
import librosa
from scipy.io import wavfile
import soundfile as sf


model = whisper.load_model("base")
selected_keys = {'ru', 'tr', 'ar'}

def detect_language(audio_file):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    dummy, probs = model.detect_language(mel)

    selected_values = {key: probs[key] for key in selected_keys}

    pred_lang = max(selected_values, key=probs.get)
    return pred_lang

def resample(audio_file, sample_rate, out_path):
    x, sr = librosa.load(audio_file)
    sf.write(out_path, x, samplerate=sample_rate)

def preprocess(audio_file):
    full_path = os.path.join('data/wav_files_subset', audio_file)
    resample_path = os.path.join('data/resampled', audio_file)

    with wave.open(full_path, "rb") as wav_file:
        # Get the sample rate
        sample_rate = wav_file.getframerate()
        nchannels = wav_file.getnchannels()

    pred_lang = detect_language(full_path)
    resample(full_path, 16_000, resample_path)

    return sample_rate, nchannels, pred_lang


def pre_processing(df):
    df[
        'sample_rate',
        'nchannels',
        'pred_lang'
    ] = df['file'].progress_apply(preprocess)

    return df


def calculate_accuracy(df):
    correct_predictions = (df['lang'] == df['pred_lang']).sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions * 100
    return accuracy


def main():
    langdict = {
        'russian': 'ru',
        'arabic': 'ar',
        'turkish': 'tr',
    }
    # Read CSV file
    csv_file = "data/hackathon_train_subset.csv"  # Replace with your CSV file path
    df = pd.read_csv(csv_file)
    df['lang'] = df['language'].map(langdict)
    df = df.head(10)

    # Predict language
    pre_processing_df = pre_processing(df)

    # Calculate accuracy
    accuracy = calculate_accuracy(pre_processing_df)
    print("Accuracy: {:.2f}%".format(accuracy))


if __name__ == "__main__":
    main()
