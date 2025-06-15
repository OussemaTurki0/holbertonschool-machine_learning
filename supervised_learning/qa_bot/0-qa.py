#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad'
)
model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')

def question_answer(question, reference):
    """Answers a question from a reference text using BERT"""
    # Tokenize
    inputs = tokenizer.encode_plus(
        question,
        reference,
        add_special_tokens=True,
        return_tensors='np'
    )
    input_ids = inputs["input_ids"].tolist()[0]

    # Run model
    outputs = model([inputs['input_ids'],
                     inputs['attention_mask'],
                     inputs['token_type_ids']])
    start_scores, end_scores = outputs

    # Get best start/end tokens
    start = np.argmax(start_scores)
    end = np.argmax(end_scores) + 1

    if start >= end:
        return None

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[start:end])
    )

    return answer.strip() if answer else None
