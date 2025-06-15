#!/usr/bin/env python3
import os
import tensorflow_hub as hub
import numpy as np

# Load USE
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def semantic_search(corpus_path, sentence):
    """Performs semantic search on a corpus of documents"""
    docs = []
    doc_texts = []

    for filename in sorted(os.listdir(corpus_path)):
        if not filename.endswith('.md'):
            continue
        with open(os.path.join(corpus_path, filename)) as f:
            text = f.read()
            docs.append(text)
            doc_texts.append(text)

    embeddings = embed([sentence] + doc_texts)
    question_vec = embeddings[0]
    doc_vecs = embeddings[1:]

    similarities = np.inner(question_vec, doc_vecs)
    idx = np.argmax(similarities)

    return docs[idx]
