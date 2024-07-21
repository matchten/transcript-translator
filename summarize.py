from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from tqdm import tqdm
from transcribe import get_transcription
import requests
import os
import re
from dotenv import load_dotenv

# def summarize_text(text, max_length=150, model_name="facebook/bart-large-cnn"):
#     tokenizer = BartTokenizer.from_pretrained(model_name)
#     model = BartForConditionalGeneration.from_pretrained(model_name)

#     inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
#     summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=max_length, early_stopping=True)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#     return summary

# def chunk_summarize(text, chunk_size=1500, intermediate_length=80):
#     chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
#     intermediate_summaries = []
#     for chunk in tqdm(chunks, desc="Summarizing Chunks", unit="chunk"):
#         summary = summarize_text("Give a brief summary of the following text:" + chunk, intermediate_length)
#         intermediate_summaries.append(summary)

#     combined_summary = " ".join(intermediate_summaries)
#     return combined_summary

def truncate_text(text, max_tokens=7500):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    tokens = tokenizer.encode(text, max_length=max_tokens, truncation=True)
    truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
    return truncated_text

def summarize(text):
    # Truncate the text before summarizing
    truncated_text = truncate_text(text)
    
    load_dotenv()
    API_KEY = os.getenv("TOGETHER_API_KEY")
    if not API_KEY:
        raise ValueError("Please set the TOGETHER_API_KEY environment variable")

    url = "https://api.together.xyz/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        "messages": [
            {
                "role": "user",
                "content": f"[INST] Write exactly one summary of the following text: \n\n{truncated_text}\n\nSummary:[/INST]",
            },
        ],
        "max_tokens": 512,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.1,
        "stop": ["</s>", "Human:", "Assistant:", "[INST]"],
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            summary = result['choices'][0]['message']['content']
            if summary.startswith(truncated_text[:100]):  # Check if summary is just the input text
                raise Exception("Model returned input text instead of summary")
            # Truncate at ".assistant"
            summary = summary.split(".assistant")[0]
            # Remove any remaining [INST] tags and text after them
            summary = re.split(r'\[INST\]', summary)[0]
            # Remove introductory text
            summary = re.sub(r'^.*?Here\'s a summary of the text:\s*', '', summary, flags=re.DOTALL)
            # Remove any remaining numbering
            summary = re.sub(r"^\d+\.\s*", "", summary, flags=re.MULTILINE)
            summary += "."
            print("Summary Complete:")
            print(summary)
            return summary
        else:
            raise Exception("Unexpected API response format")
    else:
        raise Exception(f"Error in API call: {response.text}")


def main():
    # Example usage
    youtube_url = input("Enter the YouTube URL: ")
    transcript = get_transcription(youtube_url=youtube_url)

    final_summary = summarize(transcript)
    print("FINAL SUMMARY LENGTH", len(final_summary))
    print("\nFINAL SUMMARY:")
    print(final_summary)


if __name__ == "__main__":
    main()
