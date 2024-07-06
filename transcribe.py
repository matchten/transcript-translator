import whisper
from pyannote.audio import Pipeline
import yt_dlp as youtube_dl
import librosa
import torch
from dotenv import load_dotenv
import os

# Initialize Whisper model
model = whisper.load_model("tiny")

# Get hf access token
load_dotenv()
hf_token = os.getenv("HUGGING_FACE_TOKEN")

# Initialize pyAnnote pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)

def download_youtube_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': 'downloaded_audio.%(ext)s',
        'quiet': True
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return 'downloaded_audio.wav'

def load_audio_file(file_path):
    data, sample_rate = librosa.load(file_path, sr=16000, mono=True)
    # Convert to 2D Tensor (1, time)
    tensor_data = torch.tensor(data).unsqueeze(0)
    return tensor_data, sample_rate

# Input YouTube URL
youtube_url = input("Enter the YouTube URL: ")

# Download and load audio from YouTube
audio_file_path = download_youtube_audio(youtube_url)
audio_data, sample_rate = load_audio_file(audio_file_path)

# Perform speaker diarization
diarization = pipeline({'uri': 'audio', 'waveform': audio_data, 'sample_rate': sample_rate})

# Transcribe each segment
transcriptions = []
for segment, _, speaker in diarization.itertracks(yield_label=True):
    print(f"Speaker: {speaker}")
    print(segment)
    start, end = map(int, segment)  # Convert start and end to integers
    # Convert time in seconds to sample indices
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    segment_data = audio_data[:, start_sample:end_sample]  # Slice and add batch dimension
    result = model.transcribe(segment_data)
    print(result['text'])
    transcriptions.append((segment, result['text']))

# Merge and label transcriptions
final_transcription = ""
for segment, text in transcriptions:
    final_transcription += f"Speaker {segment['track']}: {text}\n"

# Output final transcription
print(final_transcription)
