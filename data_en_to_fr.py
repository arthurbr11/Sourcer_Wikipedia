import os
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import numpy as np
PATH = os.getcwd().rstrip('Sourcer_Wikipedia')
model_name = "Helsinki-NLP/opus-mt-tc-big-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

with open(f'{PATH}data/en_statements/fa - featured articles/en_wiki_subset_statements_all_citations_sample.txt', 'r',
          encoding="utf8") as f:
    text_en = f.readlines()
print(text_en[0].split("\t"))
nb_col = len(text_en[0])
nb_raw = len(text_en)
text_fr = [text_en[0]]
for n in tqdm(range(1, nb_raw)):
    list_n = text_en[n].split("\t")
    to_trad = [' '.join(text_en[1].split("\t")[3].split("_")),list_n[5],list_n[8]]
    translated = model.generate(**tokenizer(to_trad, return_tensors="pt", padding=True))
    trad=[]
    for t in translated:
        trad.append(tokenizer.decode(t, skip_special_tokens=True))
    list_n[3] = '_'.join(trad[0].split(" "))
    list_n[5] = trad[1]
    list_n[8] = trad[2]
    text_fr.append('\t'.join(list_n))
