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


def summarize(text):
    load_dotenv()
    API_KEY = os.getenv("TOGETHER_API_KEY")
    if not API_KEY:
        raise ValueError("Please set the TOGETHER_API_KEY environment variable")

    url = "https://api.together.xyz/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "togethercomputer/Llama-2-7B-32K-Instruct",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Please provide a concise and varied summary of the following text, avoiding repetitive phrases. "
                    f"Focus on the main points and use diverse language:\n\n{text}\n\nSummary:"
                ),
            }
        ],
        "max_tokens": 512,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "stop": ["</s>", "Human:", "Assistant:", "Keywords:", "Instructions:"],
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        summary = response.json()["choices"][0]["message"]["content"].strip()
        # Remove any remaining numbering or [INST] tags
        summary = re.sub(r"^\d+\.\s*", "", summary, flags=re.MULTILINE)
        summary = re.sub(r"\[INST\].*$", "", summary, flags=re.DOTALL)
        return summary.strip()
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
