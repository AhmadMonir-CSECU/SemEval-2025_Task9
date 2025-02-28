from tqdm import tqdm
import json
import csv
import pandas as pd
import torch
from transformers import AutoTokenizer
from bs4 import BeautifulSoup
import html
import re


def read_dataset(file_in, tokenizer, max_seq_length=1024):
    features=[]
    max_count=0
    c=0
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    haz_cat2id = json.load(open('/content/drive/MyDrive/Templates/Semeval2025Task9/Data/hazards.json', 'r'))
    prod_cat2id = json.load(open('/content/drive/MyDrive/Templates/Semeval2025Task9/Data/products.json', 'r'))

    data = pd.read_csv(file_in)
    for i in range(len(data)):
        sample = data.iloc[i]
        title = sample["title"]
        text = sample["text"]
        hazard = 0
        product = 0

        if "test" not in file_in:
          hazard = haz_cat2id[sample["hazard"]]
          product = prod_cat2id[sample["product"]]

        text = title+" # "+text
        tokens=tokenizer.tokenize(text)
        if(len(tokens)>max_seq_length):
          max_count+=1

        tokens = tokens[:max_seq_length - 2]
        ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids= tokenizer.build_inputs_with_special_tokens(ids)
        
        feature = {'input_ids': input_ids,
              'labels_h': hazard,
              'labels_p': product
              }

        features.append(feature)
        c+=1

        # if(c==40):
        #     break

    print("Total samples: ",c,"\n")
    print("Num seq that greater than max_len: ",max_count,"\n")
    return features


