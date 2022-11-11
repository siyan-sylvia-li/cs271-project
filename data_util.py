"""Utilies for loading MIMIC-III text and phenotype annotations."""

import os
from typing import Mapping, Tuple

import numpy as np
import pandas as pd


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


def check_class_imbalance(labels: np.array) -> None:
  """Prints class statistics."""
  print('\nLabel statistics:')
  for i, name in enumerate(PHENOTYPE_NAMES):
    class_labels = labels[:, i]
    ones = np.sum(class_labels)
    print(f'- {name} % of 1s: {ones / len(class_labels)}')


def extract_notes_labels(dataset_path: str) -> Tuple[np.array, np.array]:
  """Loads unstructured notes and corresponding phenotype labels."""
  dataset = pd.read_csv(dataset_path)
  print(f'Number of entries:', len(dataset))
  print(f'Notes category:', dataset['CATEGORY'].unique())
  notes = dataset['TEXT'].to_numpy()
  avg_note_length = sum([len(note.split(' ')) for note in notes]) / len(notes)
  print(f'Approx. average note length: {avg_note_length}')
  labels = dataset[PHENOTYPE_NAMES].to_numpy()
  return notes, labels


def split_dataset(
    notes: np.array, labels: np.array, train_split: float = 0.7,
    val_split: float = 0.1,
) -> Mapping[str, np.array]:
  """Splits notes and labels into train/val/test splits."""
  num_train = int(len(notes) * train_split)
  num_val = int(len(notes) * val_split)
  print('\nDataset statistics:'
      f'\n#Train: {num_train}'
      f'\n#Val: {num_val}'
      f'\n#Test: {len(notes) - num_train - num_val}')
  shuffled_indices = np.random.permutation(len(notes))
  shuffled_notes = notes[shuffled_indices]
  shuffled_labels = labels[shuffled_indices]
  return {
      'train_x': shuffled_notes[: num_train],
      'train_y': shuffled_labels[: num_train],
      'val_x': shuffled_notes[num_train: num_train + num_val],
      'val_y': shuffled_labels[num_train: num_train + num_val],
      'test_x': shuffled_notes[num_train + num_val:],
      'test_y': shuffled_labels[num_train + num_val:]
  }


if __name__ == '__main__':
  notes, labels = extract_notes_labels(DATASET_PATH)
  check_class_imbalance(labels)
  data = split_dataset(notes, labels, train_split=0.7, val_split=0.1)
