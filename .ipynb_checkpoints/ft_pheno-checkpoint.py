import json
import pickle

import tqdm
from transformers import get_scheduler
from transformers import AutoTokenizer, BertForSequenceClassification, BertModel
from typing import List
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from datasets import load_dataset
from pheno_model import PhenoPredictor
from pheno_dataset import PhenoDataset

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 500
ce = torch.nn.CrossEntropyLoss()


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def parameters_to_finetune(model: nn.Module, mode: str) -> List:
    if mode == 'all':
        return list(model.parameters())
    elif mode == 'classifier':
        classifier_weights = model.get_submodule("classifier")
        return list(classifier_weights.parameters())
    elif mode == 'lastbert':
        classifier = model.get_submodule("classifier")
        last_encoder = model.get_submodule("bert.encoder.layer.11")
        return list(classifier.parameters()) + list(last_encoder.parameters())


def get_loss(logits: torch.tensor, targets: torch.tensor) -> torch.tensor:
    ce = torch.nn.CrossEntropyLoss()
    return ce(logits, targets)


def get_acc(logits, targets):
    print(logits, targets)
    logit_max = torch.argmax(logits, dim=-1)
    acc_mask = (logit_max == targets).type(torch.uint8)
    return torch.sum(acc_mask).item() / targets.size(dim=0)


def eval():
    all_accs = 0
    losses = 0
    with torch.inference_mode():
        for data, label in val_loader:
            data = {k: v.to(DEVICE) for k, v in data.items()}
            label = label.to(DEVICE)
            logits = model(data)
            losses += ce(logits, label)
            all_accs += get_acc(logits, label)
    all_accs = all_accs / len(val_loader)
    losses = losses / len(val_loader)
    return {"acc": all_accs, "loss": losses}


def ft_bert():
    model.train()
    num_training_steps = args.epochs * len(train_loader)

    optimizer = torch.optim.Adam(parameters_to_finetune(model, args.ft_mode), lr=1e-4)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    pbar = tqdm.tqdm(range(num_training_steps))
    metrics = {"tr_loss": [], "tr_accs": [], "eval_loss": [], "eval_accs": []}
    for epoch in range(args.epochs):
        tr_losses = 0
        tr_accs = 0
        for data, label in train_loader:
            data = {k: v.to(DEVICE) for k, v in data.items()}
            label = label.to(DEVICE)
            outputs = model(data)
            loss = ce(outputs, label)
            loss.backward()
            tr_losses += loss.detach().item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            pbar.update(1)
            with torch.inference_mode():
                total_acc = get_acc(model(data), label)
                tr_accs += total_acc
            pbar.set_description(f'Fine-tuning acc: {total_acc:.04f}, loss: {loss:.04f}')
        tr_losses = tr_losses / len(train_loader)
        tr_accs = tr_accs / len(train_loader)
        metrs = eval()
        metrics["tr_loss"].append(tr_losses)
        metrics["tr_accs"].append(tr_accs)
        metrics["eval_accs"].append(metrs["acc"])
        metrics["eval_loss"].append(metrs["loss"])
        pbar.set_description(f'Eval acc: {metrs["acc"]:.04f}, Eval loss: {metrs["loss"]:.04f}')
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_mode", default="classifier")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bert_name", default="bert-base-cased")
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    model = PhenoPredictor(14, args.bert_name)
    model.to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    # tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_ids_total = pickle.load(open("data_ids.p", "rb"))
    train_set = data_ids_total['train']
    val_set = data_ids_total['val']
    test_set = data_ids_total['test']

    train_data = PhenoDataset(train_set, tokenizer)
    val_data = PhenoDataset(val_set, tokenizer)
    test_data = PhenoDataset(test_set, tokenizer)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=args.batch_size)
    print("Begin training")
    mets = ft_bert()
    torch.save(model, "pheno_ckpt.pt")
    json.dump(mets, open("results.json", "w+"))