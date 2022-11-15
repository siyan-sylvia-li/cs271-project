import tqdm
from transformers import get_scheduler
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from typing import List
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from datasets import load_dataset
from pheno_model import PhenoPredictor

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 500
dataset = load_dataset("yelp_review_full")


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
    logit_max = torch.argmax(logits, dim=-1)
    acc_mask = (logit_max == targets).type(torch.uint8)
    return torch.sum(acc_mask).item() / targets.size(dim=0)


def eval():
    all_accs = 0
    with torch.inference_mode():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            all_accs += get_acc(logits, batch['labels'])
    all_accs = all_accs / len(val_loader)
    return all_accs


def ft_bert():
    model.train()
    num_training_steps = args.epochs * len(train_loader)

    optimizer = torch.optim.Adam(parameters_to_finetune(model, args.ft_mode), lr=1e-4)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    pbar = tqdm.tqdm(range(num_training_steps))
    for step in pbar:
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if step % 10 == 0:
                with torch.inference_mode():
                    total_acc = get_acc(model(**batch).logits, batch['labels'])
                pbar.set_description(f'Fine-tuning acc: {total_acc:.04f}')
        acc_val = eval()
        pbar.set_description(f'Eval acc: {acc_val:.04f}')
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_mode", default="classifier")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    parameters_to_finetune(model, "lastbert")
    model.to(DEVICE)


    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    # tokenized_datasets = dataset.map(tokenize_function, batched=True)

    train_data = dataset["train"].shuffle(seed=42).select(range(10))
    train_data = train_data.map(tokenize_function, batched=True).rename_column("label", "labels").remove_columns("text")
    train_data.set_format("torch")
    val_data = dataset["test"].shuffle(seed=42).select(range(10))
    val_data = val_data.map(tokenize_function, batched=True).rename_column("label", "labels").remove_columns("text")
    val_data.set_format("torch")
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=args.batch_size)
    print("Begin training")
    ft_bert()
    torch.save(model, "bert_ckpt.pt")