import tensorflow as tf
from transformers import TFGPT2LMHeadModel,GPT2Tokenizer
import re
import torch
from collections import Counter
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('./big.txt').read()))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def correctResult(query):
    print(query)
    text = ""
    word_list = query.split('%20')
    for word in word_list:
        text += correction(word)
        text += " "
    print(text)
    return text

def generate(
    model,
    tokenizer,
    prompt,
    entry_count=10,
    entry_length=15, #maximum number of words
    top_p=0.8,
    temperature=1.,
):
    # model.eval()
    generated_num = 0

    filter_value = -float("Inf")

    with torch.no_grad():

        for entry_idx in trange(entry_count):

            entry_finished = False
            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            for i in range(entry_length):
                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                if next_token in tokenizer.encode("<|endoftext|>"):
                    entry_finished = True

                if entry_finished:

                    generated_num = generated_num + 1

                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)
                    break
            
            if not entry_finished:
              output_list = list(generated.squeeze().numpy())
            #   output_text = f"{tokenizer.decode(output_list)}<|endoftext|>" 
              output_text = f"{tokenizer.decode(output_list)}" 
                
    return output_text

def tokenisation(data):

    # part = r'\w+'
    part = r"[a-zA-Z]+\'*[a-z]*."
    return re.findall(part,data)

#Function to generate multiple sentences. Test data should be a dataframe
def text_generation(test_data):
    generated_lyrics = {}
    model_path = "./wreckgar-4.pt"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    test_data = correctResult(test_data)
    x = generate(model, tokenizer, test_data, entry_count=1)
    texts = tokenisation(x)
    print(texts)
    stop_words = []
    pathst = './englishST.txt'
    with open(pathst) as f:
        lines = f.readlines()
    for line in lines:
        stop_words.append(line.replace('\n',''))
    f.close() 
    text = " ".join([word.lower() for word in texts if word not in stop_words])
    generated_lyrics[0] = text
    return generated_lyrics    

if __name__== '__main__':
    print(text_generation('apple'))
