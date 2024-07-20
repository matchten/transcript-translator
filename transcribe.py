import whisper
import yt_dlp as youtube_dl
import librosa
import torch
import os
import numpy as np
from tqdm import tqdm

# Initialize Whisper model (consider using "base" or "small" for better accuracy)
model = whisper.load_model("tiny")

def download_youtube_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': 'downloaded_audio.%(ext)s',
        'quiet': True,
        'progress_hooks': [progress_hook],
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return 'downloaded_audio.wav'
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None

def progress_hook(d):
    if d['status'] == 'downloading':
        total = d.get('total_bytes')
        downloaded = d.get('downloaded_bytes', 0)
        if total:
            percentage = (downloaded / total) * 100
            print(f'\rDownloading: {percentage:.1f}%', end='', flush=True)
    elif d['status'] == 'finished':
        print('\nDownload completed.')

def load_audio_file(file_path):
    try:
        # Load audio file
        audio_data, sample_rate = librosa.load(file_path, sr=16000, mono=True)
        
        # Ensure audio is float32 numpy array
        audio_data = audio_data.astype(np.float32)
        
        return audio_data, sample_rate
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None, None

def transcribe_audio(audio_data, sample_rate):
    try:
        # Ensure audio_data is a numpy array
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.squeeze().numpy()
        
        # Ensure audio is float32 numpy array
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.frombuffer(audio_data, np.int16).flatten().astype(np.float32) / 32768.0
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Ensure audio is in the correct shape (1-dimensional)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.flatten()
        
        # Check if audio data is empty
        if len(audio_data) == 0:
            raise ValueError("Audio data is empty")
        
        # Calculate total duration
        total_duration = len(audio_data) / sample_rate
        
        # Split audio into 30-second chunks
        chunk_length = 30 * sample_rate
        chunks = [audio_data[i:i+chunk_length] for i in range(0, len(audio_data), chunk_length)]
        
        # Transcribe with progress bar
        transcription = []
        with tqdm(total=total_duration, unit='sec', desc='Transcribing') as pbar:
            for chunk in chunks:
                result = model.transcribe(chunk)
                transcription.append(result['text'])
                pbar.update(len(chunk) / sample_rate)
        
        return ' '.join(transcription)
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def get_transcription(youtube_url):
    # Download and load audio from YouTube
    print("Downloading audio...")
    audio_file_path = download_youtube_audio(youtube_url)
    if audio_file_path:
        print("Audio downloaded. Loading file...")
        audio_data, sample_rate = load_audio_file(audio_file_path)
        if audio_data is not None:
            print(f"Audio loaded. Duration: {len(audio_data) / sample_rate:.2f} seconds")
            transcription = transcribe_audio(audio_data, sample_rate)
            if transcription:
                print("Transcription complete:")
                print(transcription)
                return transcription
            
            # Clean up the temporary audio file
            os.remove(audio_file_path)
        else:
            print("Failed to load audio file.")
    else:
        print("Failed to download audio.")
    
    return None