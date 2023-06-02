from transformers import AutoTokenizer
import math

def all_cosine_similarity(vector1, vector2):
    dot_product = sum(x * y for x, y in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(x ** 2 for x in vector1))
    magnitude2 = math.sqrt(sum(y ** 2 for y in vector2))
    return dot_product / (magnitude1 * magnitude2)
def cosine_similarity(sen1, sen2):
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    sen1_encode = tokenizer(sen1, truncation=True)
    sen2_encode = tokenizer(sen2, truncation=True)

    input_ids1 = sen1_encode['input_ids'][1:-1]
    input_ids2 = sen2_encode['input_ids'][1:-1]

    cosine_sim = all_cosine_similarity(input_ids1,input_ids2)

    return cosine_sim
