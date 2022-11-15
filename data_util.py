"""Utilies for loading MIMIC-III text and phenotype annotations."""
import json
import os
from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import pickle


DATA_DIR = 'data/'
# Dataset created from joining NOTEEVENTS.csv and ACTdb102003.csv.
DATASET_PATH = os.path.join(DATA_DIR, 'dataset.csv')
PHENOTYPE_NAMES = [
    'ADVANCED.HEART.DISEASE', 'ADVANCED.LUNG.DISEASE', 'ALCOHOL.ABUSE',
    'CHRONIC.NEUROLOGICAL.DYSTROPHIES', 'CHRONIC.PAIN.FIBROMYALGIA',
    'DEMENTIA', 'DEPRESSION', 'DEVELOPMENTAL.DELAY.RETARDATION',
    'NON.ADHERENCE', 'NONE', 'OBESITY', 'OTHER.SUBSTANCE.ABUSE',
    'SCHIZOPHRENIA.AND.OTHER.PSYCHIATRIC.DISORDERS', 'UNSURE',
]
CATEGORIES = [
    'Discharge summary', 'ECG', 'Radiology', 'Nursing/other', 'Echo',
    'Physician ', 'Nursing', 'Respiratory ', 'Social Work', 'Rehab Services',
    'Nutrition', 'Case Management ', 'General', 'Pharmacy',
]


def check_class_imbalance(dataset_path: str) -> None:
  """Prints class statistics."""
  print('\nLabel statistics:')
  dataset = pd.read_csv(dataset_path)
  labels = dataset[PHENOTYPE_NAMES].to_numpy()
  for i, name in enumerate(PHENOTYPE_NAMES):
    class_labels = labels[:, i]
    ones = np.sum(class_labels)
    print(f'- {name} % of 1s: {ones / len(class_labels)}')


def check_dataset_statistics(dataset_path: str) -> None:
  """Prints dataset statistics."""
  print('\nDataset statistics:')
  dataset = pd.read_csv(dataset_path)
  print(f'Number of entries:', len(dataset))
  print(f'Notes category:', list(dataset['CATEGORY'].unique()))
  print('Approx. average note length:')
  for category, sub_df in dataset.groupby('CATEGORY'):
    notes = sub_df['TEXT'].to_numpy()
    avg_note_length = sum([len(note.split(' ')) for note in notes]) / len(notes)
    print(f'- {category}: {avg_note_length}')


def get_data(
    dataset: pd.DataFrame,
    categories: Sequence[str],
    subject_id: int,
    hadm_id: int
) -> Tuple[np.array, np.array]:
  """Loads unstructured notes and corresponding phenotype labels."""
  # print(subject_id, hadm_id)
  sub_dataset = dataset.loc[
      (dataset['CATEGORY'].isin(categories)) &
      (dataset['SUBJECT_ID'] == subject_id) &
      (dataset['HADM_ID'] == hadm_id)]
  # print(sub_dataset)
  notes = sub_dataset['TEXT'].to_numpy()
  labels = sub_dataset[PHENOTYPE_NAMES].to_numpy()
  return notes, labels


def split_dataset(
    dataset_path: str,
    train_split: float = 0.7,
    val_split: float = 0.1,
) -> Mapping[str, np.array]:
  """Splits (SUBJECT_ID, HADM_ID) pairs into train/val/test splits."""
  dataset = pd.read_csv(dataset_path)
  samples = dataset.drop_duplicates(
      subset=['SUBJECT_ID', 'HADM_ID'])[['SUBJECT_ID', 'HADM_ID']].to_numpy()
  num_train = int(len(samples) * train_split)
  num_val = int(len(samples) * val_split)
  print('\nDataset splits:'
      f'\n#Train: {num_train}'
      f'\n#Val: {num_val}'
      f'\n#Test: {len(samples) - num_train - num_val}')
  shuffled_indices = np.random.permutation(len(samples))
  shuffled_samples = samples[shuffled_indices]
  return {
      'train': shuffled_samples[: num_train],
      'val': shuffled_samples[num_train: num_train + num_val],
      'test': shuffled_samples[num_train + num_val:],
  }


if __name__ == '__main__':
  check_class_imbalance(DATASET_PATH)
  check_dataset_statistics(DATASET_PATH)
  data_ids = split_dataset(DATASET_PATH, train_split=0.7, val_split=0.1)

  pickle.dump(data_ids, open("data_ids.p", "wb+"))

  notes, labels = get_data(
      pd.read_csv(DATASET_PATH),
      categories=['Discharge summary'],
      subject_id=data_ids['train'][0][0],
      hadm_id=data_ids['train'][0][1])
  # print(notes.shape, labels.shape)
  # print(notes[0][:50], labels)
