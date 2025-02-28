import argparse
import os

import pandas as pd
import numpy as np
import torch
import json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_scheduler
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import *
from utils import set_seed, collate_fn
from prepro import read_dataset
from model import AIModel
from tqdm.auto import tqdm
from apex import amp
from sklearn.model_selection import train_test_split

def train(args, model, train_features, dev_features):
    def finetune(features, optimizer, num_epochs):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        num_training_steps = num_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps=num_training_steps) 
        print(num_training_steps)    
        progress_bar = tqdm(range(num_training_steps))

        for epoch in range(num_epochs):
            total_loss = 0
            total_batch = len(train_dataloader)
            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[args.label_pos],
                          }

                outputs = model(**inputs)
                loss = outputs["loss"]
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                progress_bar.update(1)
                total_loss += loss
            
            average_loss = total_loss / total_batch  
            print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")  
            torch.save(model.state_dict(), args.save_path)      
           
    new_layer = ["classifier"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs)


def report(args, model, features):
    haz_cat2id = json.load(open('/content/drive/MyDrive/Templates/Semeval2025Task9/Data/hazard_categories.json', 'r'))
    prod_cat2id = json.load(open('/content/drive/MyDrive/Templates/Semeval2025Task9/Data/product_categories.json', 'r'))
    haz_id2cat = {value: key for key, value in haz_cat2id.items()}
    prod_id2cat = {value: key for key, value in prod_cat2id.items()}

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  }

        with torch.no_grad():
            output = model(**inputs)
            pred = torch.argmax(output["logits"],dim=-1)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    labels = [float(item[args.label_name]) for item in features]
    labels = np.array(labels)

    prediction = []
    true_labels = []

    if args.label_pos==2:
      prediction = np.array([haz_id2cat[p] for p in preds])
      true_labels = np.array([haz_id2cat[p] for p in labels])

    else:
      prediction = np.array([prod_id2cat[p] for p in preds])
      true_labels = np.array([prod_id2cat[p] for p in labels])

    return true_labels, prediction  

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../Data", type=str)
    parser.add_argument("--model_name_or_path", type=str, default= "bert-base-uncased")

    parser.add_argument("--train_file", default="train_merge.csv", type=str)
    # parser.add_argument("--train_file", default="incidents_train.csv", type=str)
    parser.add_argument("--dev_file", default="incidents_valid_labeled.csv", type=str)
    parser.add_argument("--test_file", default="incidents_test.csv", type=str)

    parser.add_argument("--save_path", default="/content/drive/MyDrive/Templates/Semeval2025Task9/Data/checkpoint_product_cat", type=str)

    parser.add_argument("--load_path_h", default="", type=str)
    parser.add_argument("--load_path_p", default="", type=str)
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--test_batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--num_train_epochs", default=7, type=int)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--num_class", type=int, default=22)
    parser.add_argument("--label_pos", type=int, default=3) #3
    parser.add_argument("--label_name", type=str, default="labels_p")
 
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_class,
    )

    read = read_dataset

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)

    train_features = read(train_file, args.model_name_or_path, max_seq_length=args.max_seq_length)
    dev_features = []
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        config=config,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id

    set_seed(args)
    model = AIModel(config, model, num_class=args.num_class)
    model.to(args.device)    
 
    if args.load_path_h == "":  # Training
        train(args, model, train_features, dev_features)

    else:  # Testing
        test_df = pd.read_csv(os.path.join(args.data_dir, args.test_file))
        test_features = read(test_file, args.model_name_or_path, max_seq_length=args.max_seq_length)

        args.label_pos = 2
        args.label_name = "labels_h"
        args.num_class = 10
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=args.num_class,
        )
        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            config=config,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        config.cls_token_id = tokenizer.cls_token_id
        config.sep_token_id = tokenizer.sep_token_id       
        set_seed(args)
        model = AIModel(config, model, num_class=args.num_class)
        model.to(args.device)   
        model = amp.initialize(model, opt_level="O1", verbosity=0)       
        model.load_state_dict(torch.load(args.load_path_h))
        hazard_true, hazard_preds = report(args, model, test_features)
        test_df['hazard-category'] = hazard_preds


        args.label_pos = 3
        args.label_name = "labels_p"
        args.num_class = 22
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=args.num_class,
        )
        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            config=config,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        config.cls_token_id = tokenizer.cls_token_id
        config.sep_token_id = tokenizer.sep_token_id             
        set_seed(args)
        model = AIModel(config, model, num_class=args.num_class)
        model.to(args.device) 
        model = amp.initialize(model, opt_level="O1", verbosity=0)     
        model.load_state_dict(torch.load(args.load_path_p))
        product_true, product_preds = report(args, model, test_features)
        test_df['product-category'] = product_preds
        test_df[['hazard-category', 'product-category']].to_csv('submissionST1.csv')
        
if __name__ == "__main__":
    main()