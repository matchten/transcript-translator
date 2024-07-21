# ScribeGPT

Use ScribeGPT to summarize any YouTube video. Whether it be
a lecture video, news report, or podcast, ScribeGPT will provide
transcribe and summarize the video.

Built using whisper and LLama3. MIT License.

<div style="display: flex; justify-content: space-between;">
    <img src="images/home.png" alt="screenshot" style="width: 50%;">
    <img src="images/output.png" alt="screenshot" style="width: 50%;">
</div>

## Setup

1. Clone the repository.

```
git clone https://github.com/matchten/transcript-translator.git
```

2. Navigate to the project directory.
3. Create a virtual environment (python 3.9) and activate it.

```
python3.9 -m venv venv
source venv/bin/activate
```

4. Install the required packages

```
pip install -r requirements.txt
```

5. Build node modules.

```
npm install
```

6. Run the flask app.

```
python app.py
```

7. Navigate to `localhost:5000` to start summarizing your videos.
