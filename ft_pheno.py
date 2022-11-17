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

import data_util
from pheno_model import PhenoPredictor
from pheno_dataset import PhenoDataset
import utils

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 500
be = torch.nn.BCELoss()
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
    return be(logits, targets)


def get_acc(logits, targets):
    binarized_logits = (logits > THRESHOLD).long()
    acc_mask = (binarized_logits == targets).type(torch.uint8)
    return torch.sum(acc_mask, dim=0) / targets.shape[0]


def get_f1(logits, targets):
    preds = (logits > THRESHOLD).long()
    results = defaultdict(dict)
    for i, p in enumerate(data_util.PHENOTYPE_NAMES):
        for f1_type in ['macro', 'micro', 'weighted']:
            f1 = metrics.f1_score(
                targets[:, i], preds[:, i], average=f1_type)
            results[p][f'f1_{f1_type}'] = f1
    return results


def eval():
    all_accs = torch.zeros(len(data_util.PHENOTYPE_NAMES))
    losses = 0
    eval_f1s = defaultdict(lambda: defaultdict(int))
    avg_f1s = 0
    with torch.inference_mode():
        for data, label in val_loader:
            data = {k: v.to(DEVICE) for k, v in data.items()}
            label = label.to(DEVICE)
            logits = model(data)
            losses += be(logits, label)
            total_acc = get_acc(logits.detach().cpu(), label.detach().cpu())
            all_accs += total_acc
            f1s = get_f1(logits.detach().cpu(), label.detach().cpu())
            weighted_f1s = 0
            for p, p_f1s in f1s.items():
                for f1_type, f1 in p_f1s.items():
                    eval_f1s[p][f1_type] += f1
                weighted_f1s += f1s[p]['f1_weighted']
            avg_weighted_f1 = weighted_f1s / len(f1s)
            avg_f1s += avg_weighted_f1
    all_accs = all_accs / len(val_loader)
    losses = losses / len(val_loader)
    results = dict()
    for p, p_f1s in eval_f1s.items():
        for f1_type, f1 in p_f1s.items():
            results[f"{p}_{f1_type}"] = eval_f1s[p][f1_type] / len(val_loader)
    avg_f1s = avg_f1s / len(val_loader)
    return {"acc": all_accs, "loss": losses.item(), "avg_f1_weighted": avg_f1s, **results}


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
        tr_accs = torch.zeros(len(data_util.PHENOTYPE_NAMES))
        tr_f1s = defaultdict(lambda: defaultdict(int))
        tr_avg_f1s = 0
        for data, label in train_loader:
            data = {k: v.to(DEVICE) for k, v in data.items()}
            label = label.to(DEVICE)
            outputs = model(data)
            loss = be(outputs, label)
            loss.backward()
            tr_losses += loss.detach().item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            pbar.update(1)
            with torch.inference_mode():
                logits = model(data)
                total_acc = get_acc(logits.detach().cpu(), label.detach().cpu())
                tr_accs += total_acc
                f1s = get_f1(logits.detach().cpu(), label.detach().cpu())
                weighted_f1s = 0
                for p, p_f1s in f1s.items():
                    for f1_type, f1 in p_f1s.items():
                        tr_f1s[p][f1_type] += f1
                    weighted_f1s += f1s[p]['f1_weighted']
                avg_weighted_f1 = weighted_f1s / len(f1s)
                tr_avg_f1s += avg_weighted_f1
            pbar.set_description(f'Fine-tuning acc: {total_acc.mean().item():.04f}, loss: {loss:.04f}, F1: {avg_weighted_f1:.04f}')
        tr_losses = tr_losses / len(train_loader)
        tr_accs = tr_accs / len(train_loader)
        for p, p_f1s in tr_f1s.items():
            for f1_type, f1 in p_f1s.items():
                tr_f1s[p][f1_type] = tr_f1s[p][f1_type] / len(train_loader)
                metrics[f"tr_{p}_{f1_type}"].append(tr_f1s[p][f1_type])
        tr_avg_f1s = tr_avg_f1s / len(train_loader)

        metrs = eval()
        for i, p in enumerate(data_util.PHENOTYPE_NAMES):
            metrics[f"tr_{p}_accs"].append(tr_accs[i].item())
            metrics[f"eval_{p}_accs"].append(metrs["acc"][i].item())
        metrics["tr_loss"].append(tr_losses)
        metrics["eval_loss"].append(metrs["loss"])
        metrics["tr_avg_f1s_weighted"].append(tr_avg_f1s)
        for p, p_f1s in tr_f1s.items():
            for f1_type, _ in p_f1s.items():
                metrics[f"eval_{p}_{f1_type}"].append(metrs[f"{p}_{f1_type}"])
        metrics["eval_avg_f1s_weighted"].append(metrs["avg_f1_weighted"])
        pbar.set_description(f'Eval acc: {metrs["acc"].mean().item():.04f}, Eval loss: {metrs["loss"]:.04f}, Eval F1: {metrs["avg_f1_weighted"]:.04f}')

        early_stopping(-metrics['eval_avg_f1s_weighted'][-1], model)
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
