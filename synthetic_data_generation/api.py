# For each selected individual, prompt the LLM to generate a synthetic entry based on the provided prompt.

from typing import List
from google import genai
import time
import os
import random
import torch
from transformers import pipeline
from openai import OpenAI
from openai import APIStatusError
import transformers
import yaml


MAX_DELAY = 60
BASE_DELAY = 2
MAX_RETRIES = 10


def get_gemini_response(prompt, api_key):
    client = genai.Client(api_key=api_key)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Send the request
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            # print(response.text)
            return response.text

        except Exception as e:
            # Catch errors
            if "503" in str(e) or "overloaded" in str(e).lower():
                wait = min(MAX_DELAY, BASE_DELAY * (2 ** (attempt - 1)))
                # Add jitter to avoid stampeding the server
                wait += random.uniform(0, 1)
                print(
                    f"[Retry {attempt}/{MAX_RETRIES}] Model overloaded. Waiting {wait:.1f}s..."
                )
                time.sleep(wait)
            else:
                # ❌ If it's some other error, stop trying
                raise


def get_llama_response(prompt, api_key):
    model = pipeline(
        "text-generation", model="meta-llama/Llama-3.1-8B-Instruct"
    )
    out_text = ""
    chat = [
        {
            "role": "system",
            "content": "You are an AI Assistant that specializes in generating synthetic data. Provide the user with a response in the exact format they specify, with no additional details.",
        },
        {"role": "user", "content": prompt},
    ]
    response = model(chat)

    for r in response[0]["generated_text"]:
        if r["role"] == "assistant":
            out_text = r["content"]
    return out_text

# def get_llama_response_batch(model:transformers.pipeline, prompts: List):

#     chats = [
#         [
#             {"role": "system", "content": "You are an AI Assistant that specializes in generating synthetic data. Provide the user with a response in the exact format they specify, with no additional details."},
#             {"role": "user", "content": prompt},
#         ] for prompt in prompts
#     ]

#     model.tokenizer.pad_token_id = model.model.config.eos_token_id[0]
#     responses = model(chats,
#                       max_new_tokens=2048,
#                       batch_size=len(prompts))
#     out_texts = []
#     for response in responses:
#         for r in response[0]["generated_text"]:
#             if r["role"]=="assistant":
#                 out_text = r["content"]
#                 out_texts.append(out_text)
#     return out_text


def get_chatgpt_response(prompt, api_key):
    # Initialize the client

    client = OpenAI(api_key=api_key)
    # Send a prompt to ChatGPT
    # response = client.responses.create(
    #     model="gpt-4o",
    #     instructions="You are an AI Assistant that specializes in generating synthetic data. Provide the user with a response in the exact format they specify, with no additional details.",
    #     input=prompt,
    # )
    try:
        
        response = client.chat.completions.create(model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI Assistant that specializes in generating synthetic data. Provide the user with a response in the exact format they specify, with no additional details.",
                },
                {"role": "user", "content": prompt},
            ])
        # print("GPT response " + response.choices[0].message.content)
        return response.choices[0].message.content
    except APIStatusError as e:
        raise e


# Get response from LLM API to a prompt
def get_llm_response(prompt, api_key, llm):
    if llm == "gemini":
        return get_gemini_response(prompt, api_key)
    elif llm == "llama":
        return get_llama_response(prompt, api_key)
    elif llm == "chatgpt":
        return get_chatgpt_response(prompt, api_key)
