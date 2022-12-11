import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_util import get_data
import pandas as pd
import os
from transformers import AutoTokenizer

DATA_DIR = 'data/'
DATASET_PATH = os.path.join(DATA_DIR, 'discharge_tokenized_dataset.csv')
NUM_CHUNKS = 15


# def process_text(text):
#     text = text.replace("\n", " ")
#     text_splts = text.split(" ")
#     text_splts = [x for x in text_splts if len(x)]
#     return " ".join(text_splts)

class PhenoDataset(Dataset):
    def __init__(self, data_ids, tokenizer, max_len=128):
        self.data_ids = data_ids
        self.tokenizer = tokenizer
        self.max_len = 128

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, item):
        notes, labels = get_data(
            pd.read_csv(DATASET_PATH),
            categories=['Discharge summary'],
            subject_id=self.data_ids[item][0],
            hadm_id=self.data_ids[item][1])
        notes = notes.tolist()

        final_note = self.tokenizer(
            notes,
            max_length=self.max_len,
            truncation=True,
            stride=self.max_len // 8,
            padding="max_length",
            return_overflowing_tokens=True,
            return_tensors='pt'
        )

        print(len(final_note['input_ids']), final_note['input_ids'].shape)

        to_pad = 60 - len(final_note['input_ids'])

        for k in final_note:
            if len(final_note[k].shape) == 2:
                padding = torch.zeros((to_pad, self.max_len), dtype=torch.long)
                final_note[k] = torch.cat([torch.LongTensor(final_note[k]), padding])
            else:
                padding = torch.zeros((to_pad,), dtype=torch.long)
                final_note[k] = torch.cat([final_note[k], padding])

        final_note.update({"orig_len": 60 - to_pad})

        # print("NOTES", len(notes), final_note['input_ids'].shape)
        # if len(notes) != len(final_note['input_ids']):
        #     print(len(notes), len(final_note['input_ids']))
            # print(final_note["input_ids"][:3])

        return final_note, torch.FloatTensor(labels[0])


#         if len(encoded_note['input_ids']) <= self.max_len:
#             item_note = copy.copy(encoded_note)
#             for k in item_note:
#                 print(k, item_note[k][:10])
#                 item_note[k] = item_note[:, k] + [0] * (self.max_len - len(item_note[k]))
#                 item_note[k] = torch.tensor(item_note[k])
#             final_note.append(item_note)
#         else:
#             ptr = 0
#             while len(encoded_note['input_ids'][ptr:]) > self.max_len:
#                 item_note = {
#                     'input_ids': torch.tensor(encoded_note['input_ids'][ptr: ptr + self.max_len]),
#                     'token_type_ids': torch.tensor(encoded_note['token_type_ids'][ptr: ptr + self.max_len]),
#                     'attention_mask': torch.tensor(encoded_note['attention_mask'][ptr: ptr + self.max_len])
#                 }
#                 final_note.append(item_note)
#                 ptr = ptr + self.max_len
#             fin_len = len(encoded_note['input_ids'][ptr:])
#             item_note = {
#                 'input_ids': torch.tensor(encoded_note['input_ids'][ptr:] + [0] * (self.max_len - fin_len)),
#                 'token_type_ids': torch.tensor(encoded_note['token_type_ids'][ptr:] + [0] * (self.max_len - fin_len)),
#                 'attention_mask': torch.tensor(encoded_note['attention_mask'][ptr:] + [0] * (self.max_len - fin_len))
#             }
#             final_note.append(item_note)

if __name__ == "__main__":
    # Dataset tester
    data_ids_total = pickle.load(open("data_ids.p", "rb"))
    val_set = data_ids_total['val']
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    for k in ['train', 'val', 'test']:
        set_data = data_ids_total[k]
        dset = PhenoDataset(set_data, tokenizer)
        loader = torch.utils.data.DataLoader(dset)
        for d in loader:
            print(d[1].shape)
    # dataset_val = PhenoDataset(val_set, tokenizer)
    # loader = torch.utils.data.DataLoader(dataset_val)
