from flask import Flask, request, render_template, jsonify
from summarize import summarize
from transcribe import get_transcription

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_video():
    youtube_url = request.form['youtube_url']
    try:
        transcript = get_transcription(youtube_url=youtube_url)
        final_summary = summarize(transcript)
        return jsonify({'summary': final_summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/dev/result')
def dev_result():
    dummy_summary = "This is a dummy summary for development purposes. It allows you to view and modify the CSS of the result page without having to resubmit a YouTube URL each time. You can make this dummy text as long or as short as you need to test various layout scenarios."
    return render_template('result.html', summary=dummy_summary)

if __name__ == '__main__':
    app.run(debug=True)