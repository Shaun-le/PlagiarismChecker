import torch
import re
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
import random
from colorama import Fore, Style

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
def split_sentences(text):
    text = re.sub(r'\b\d+\.\d+(\.\d+)*\b', '', text)
    text = re.sub(r'\b(Hình\s*\d+\.\d+|\d+\.\d+\.\d+)\b', '', text)
    text = re.sub(r'\bBảng\s*\d+\.\d+\b|\b^\d+\b', '', text)
    text = re.sub(r'^\d+\s*', '', text)
    sentences = re.split(r'\.', text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences
'''def colorize_high_similarity(similarity_matrix, threshold):
    colored_sentences = []
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            if similarity_matrix[i, j] >= threshold and i != j:
                colored_sentences.append((i, j))
    return colored_sentences'''

def computed_cosine_similarity(sentences1, sentences2):
    # Encode the sentences using the tokenizer
    encoded_sentences1 = tokenizer(sentences1, padding=True, truncation=True, return_tensors="pt")
    encoded_sentences2 = tokenizer(sentences2, padding=True, truncation=True, return_tensors="pt")

    # Retrieve the input IDs and attention masks
    input_ids1 = encoded_sentences1["input_ids"]
    attention_mask1 = encoded_sentences1["attention_mask"]
    input_ids2 = encoded_sentences2["input_ids"]
    attention_mask2 = encoded_sentences2["attention_mask"]

    # Pad the input IDs and attention masks to have the same length
    max_length = max(input_ids1.size(1), input_ids2.size(1))
    input_ids1 = torch.nn.functional.pad(input_ids1, (0, max_length - input_ids1.size(1)))
    attention_mask1 = torch.nn.functional.pad(attention_mask1, (0, max_length - attention_mask1.size(1)))
    input_ids2 = torch.nn.functional.pad(input_ids2, (0, max_length - input_ids2.size(1)))
    attention_mask2 = torch.nn.functional.pad(attention_mask2, (0, max_length - attention_mask2.size(1)))

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(input_ids1, input_ids2)

    return similarity_matrix


def colorize_high_similarity(similarity_matrix, threshold):
    colored_sentences = []

    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            if similarity_matrix[i, j] >= threshold:
                color = f"#{random.randint(0, 0xFFFFFF):06x}"
                colored_sentences.append((i, j, color))

    return colored_sentences


def checker(sentences1, sentences2):
    sentences1 = split_sentences(sentences1)
    sentences2 = split_sentences(sentences2)
    cosine_similarity_matrix = computed_cosine_similarity(sentences1, sentences2)
    colored_sentences = colorize_high_similarity(cosine_similarity_matrix, threshold=0.7)

    colored_doc1 = []
    colored_doc2 = []

    for i in range(len(sentences1)):
        colored = next((colored for colored in colored_sentences if colored[0] == i), None)
        if colored:
            colored_sentence = f'<span style="color: {colored[2]};">{sentences1[i]}</span>'
            colored_doc1.append(colored_sentence)
        else:
            colored_doc1.append(sentences1[i])

    for j in range(len(sentences2)):
        colored = next((colored for colored in colored_sentences if colored[1] == j), None)
        if colored:
            colored_sentence = f'<span style="color: {colored[2]};">{sentences2[j]}</span>'
            colored_doc2.append(colored_sentence)
        else:
            colored_doc2.append(sentences2[j])

    s1 = '<br>'.join(colored_doc1)
    s2 = '<br>'.join(colored_doc2)

    return s1, s2