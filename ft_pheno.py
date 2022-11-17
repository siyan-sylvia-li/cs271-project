from collections import defaultdict
import json
import pickle
import os

from sklearn import metrics
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
import utils

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 500
ce = torch.nn.CrossEntropyLoss()
THRESHOLD = 0.5
RESULT_DIR = "results"


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
    binarized_logits = (logits > THRESHOLD).long()
    acc_mask = (binarized_logits == targets).type(torch.uint8)
    return torch.sum(acc_mask).item() / (targets.shape[0] * targets.shape[1])


def get_f1(logits, targets):
    preds = (logits > THRESHOLD).long()
    f1_macro = metrics.f1_score(
        targets.flatten(), preds.flatten(), average='macro')
    f1_micro = metrics.f1_score(
        targets.flatten(), preds.flatten(), average='micro')
    f1_weighted = metrics.f1_score(
        targets.flatten(), preds.flatten(), average='weighted')
    return {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
    }


def eval():
    all_accs = 0
    losses = 0
    f1s = defaultdict(float)
    with torch.inference_mode():
        for data, label in val_loader:
            data = {k: v.to(DEVICE) for k, v in data.items()}
            label = label.to(DEVICE)
            logits = model(data)
            losses += ce(logits, label)
            all_accs += get_acc(logits, label)
            for f1_type, f1 in get_f1(logits, label).items():
                f1s[f1_type] += f1
    all_accs = all_accs / len(val_loader)
    losses = losses / len(val_loader)
    for f1_type in f1s:
        f1s[f1_type] = f1s[f1_type] / len(val_loader)
    return {"acc": all_accs, "loss": losses.item(), **f1s}


def ft_bert():
    model.train()
    num_training_steps = args.epochs * len(train_loader)

    optimizer = torch.optim.Adam(parameters_to_finetune(model, args.ft_mode), lr=1e-4)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    early_stopping = utils.EarlyStopping()

    pbar = tqdm.tqdm(range(num_training_steps))
    metrics = defaultdict(list)
    for epoch in range(args.epochs):
        tr_losses = 0
        tr_accs = 0
        tr_f1s = defaultdict(float)
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
                logits = model(data)
                total_acc = get_acc(logits, label)
                tr_accs += total_acc
                f1s = get_f1(logits, label)
                for f1_type, f1 in f1s.items():
                    tr_f1s[f1_type] += f1
            pbar.set_description(f'Fine-tuning acc: {total_acc:.04f}, loss: {loss:.04f}, F1: {f1s["f1_weighted"]:.04f}')
        tr_losses = tr_losses / len(train_loader)
        tr_accs = tr_accs / len(train_loader)
        for f1_type in tr_f1s:
            tr_f1s[f1_type] = tr_f1s[f1_type] / len(train_loader)
        metrs = eval()
        metrics["tr_loss"].append(tr_losses)
        metrics["tr_accs"].append(tr_accs)
        metrics["tr_f1s_macro"].append(tr_f1s["f1_macro"])
        metrics["tr_f1s_micro"].append(tr_f1s["f1_micro"])
        metrics["tr_f1s_weighted"].append(tr_f1s["f1_weighted"])
        metrics["eval_accs"].append(metrs["acc"])
        metrics["eval_loss"].append(metrs["loss"])
        metrics["eval_f1s_macro"].append(metrs["f1_macro"])
        metrics["eval_f1s_micro"].append(metrs["f1_micro"])
        metrics["eval_f1s_weighted"].append(metrs["f1_weighted"])
        pbar.set_description(f'Eval acc: {metrs["acc"]:.04f}, Eval loss: {metrs["loss"]:.04f}, Eval F1: {metrs["f1_weighted"]:.04f}')

        early_stopping(-metrics['eval_f1s_weighted'][-1], model)
        if args.early_stopping and early_stopping.early_stop:
            print("Early stopping...")
            break
    return metrics, early_stopping.get_best_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="bert")
    parser.add_argument("--ft_mode", default="classifier")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bert_name", default="bert-base-cased")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--baseline_bert", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stopping", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    result_dir = os.path.join(RESULT_DIR, args.exp_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    model = PhenoPredictor(
        14, args.bert_name, use_pretrained=(not args.baseline_bert))
    model.to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    # tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_ids_total = pickle.load(open("data_ids.p", "rb"))
    train_set = data_ids_total['train'][:4]
    val_set = data_ids_total['val'][:4]
    test_set = data_ids_total['test'][:4]

    train_data = PhenoDataset(train_set, tokenizer)
    val_data = PhenoDataset(val_set, tokenizer)
    test_data = PhenoDataset(test_set, tokenizer)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=args.batch_size)
    print("Begin training")
    mets, best_model = ft_bert()

    arg_id = ''
    for k, v in args.__dict__.items():
        arg_id += f'{k}={v}__'
    with open(os.path.join(result_dir, f'{arg_id}.json'), 'w') as f:
        json.dump(mets, f)
    torch.save(
        best_model,
        os.path.join(result_dir, f"{arg_id}_best.pt"))
    torch.save(model, os.path.join(result_dir, f"{arg_id}_final.pt"))
