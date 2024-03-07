import os
import pandas as pd
import speech_recognition as sr
from tqdm import tqdm
tqdm.pandas()
import whisper


model = whisper.load_model("medium")
selected_keys = {'ru', 'tr', 'ar'}

def detect_language(audio_file):
    full_path = os.path.join('data/wav_files_subset', audio_file)

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(full_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)

    selected_values = {key: probs[key] for key in selected_keys}

    pred_lang = max(selected_values, key=probs.get)
    return pred_lang


def predict_language(df):
    df['pred_lang'] = df['file'].progress_apply(detect_language)
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
    # df = df.head(10)

    # Predict language
    df_with_pred_lang = predict_language(df)

    # Calculate accuracy
    accuracy = calculate_accuracy(df_with_pred_lang)
    print("Accuracy: {:.2f}%".format(accuracy))


if __name__ == "__main__":
    main()
