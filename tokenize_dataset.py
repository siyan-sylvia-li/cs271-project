"""Adapted from CS271 HW2 starter code."""

import os
import re
from tqdm import tqdm

import pandas as pd
import spacy

import heuristic_tokenize


DATA_DIR = 'data/'
DATASET_PATH = os.path.join(DATA_DIR, 'dataset.csv')
TOKENIZED_DATASET_PATH = os.path.join(DATA_DIR, 'discharge_tokenized_dataset.csv')
# CATEGORIES = [
#     'Discharge summary', 'ECG', 'Radiology', 'Nursing/other', 'Echo',
#     'Physician ', 'Nursing', 'Respiratory ', 'Social Work', 'Rehab Services',
#     'Nutrition', 'Case Management ', 'General', 'Pharmacy',
# ]
CATEGORIES = ['Discharge summary']


# setting sentence boundaries
@spacy.Language.component("sbd_component")
def sbd_component(doc):
    for i, token in enumerate(doc[:-2]):
        # define sentence start if period + titlecase token
        if token.text == "." and doc[i + 1].is_title:
            doc[i + 1].sent_start = True
        if token.text == "-" and doc[i + 1].text != "-":
            doc[i + 1].sent_start = True
    return doc

# pip install spacy
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
nlp = spacy.load(
    "en_core_sci_md",
    disable=[
        "tagger",
        "ner",
    ],
)
nlp.add_pipe("sbd_component", before="parser")

# convert de-identification text into one token
def fix_deid_tokens(text, processed_text):
    deid_regex = r"\[\*\*.{0,15}.*?\*\*\]"
    if text:
        indexes = [
            m.span() for m in re.finditer(deid_regex, text, flags=re.IGNORECASE)
        ]
    else:
        indexes = []
    with processed_text.retokenize() as retokenizer:
        for start, end in indexes:
            try:
                span = processed_text.char_span(start, end)
                processed_text.set_ents(entities=[span])
                retokenizer.merge(span)
            except AttributeError:
                pass
        # debug: processed_text.merge(start_idx=start, end_idx=end)
    return processed_text

def process_section(section, note, processed_sections):
    # perform spacy processing on section
    processed_section = nlp(section["sections"])
    #     print(processed_section)
    processed_section = fix_deid_tokens(section["sections"], processed_section)
    processed_sections.append(processed_section)

def process_note_helper(note):
    # split note into sections
    note_sections = heuristic_tokenize.sent_tokenize_rules(note)
    processed_sections = []
    section_frame = pd.DataFrame({"sections": note_sections})
    section_frame.apply(
        process_section,
        args=(
            note,
            processed_sections,
        ),
        axis=1,
    )
    return processed_sections

def process_text(sent, note):
    sent_text = sent["sents"].text
    if len(sent_text) > 0 and sent_text.strip() != "\n":
        if "\n" in sent_text:
            sent_text = sent_text.replace("\n", " ")
        note["TEXT"] += sent_text + "\n"

def get_sentences(processed_section, note):
    # get sentences from spacy processing
    sent_frame = pd.DataFrame({"sents": list(processed_section["sections"].sents)})
    sent_frame.apply(process_text, args=(note,), axis=1)

def process_note(sub_dataset):
    # print('\n\nsub_dataset:', sub_dataset['TEXT'].to_numpy())
    for i, note in tqdm(sub_dataset.iterrows()):
        # print('\n\nnote:', note['TEXT'])
        note_text = note["TEXT"]  # unicode(note['text'])
        note["TEXT"] = ""
        processed_sections = process_note_helper(note_text)
        ps = {"sections": processed_sections}
        ps = pd.DataFrame(ps)
        ps.apply(get_sentences, args=(note,), axis=1)
        sub_dataset.at[i] = note
        # print('\n\nnote (after):', note['TEXT'])
    # print('\n\nsub_dataset (after):', sub_dataset['TEXT'].to_numpy())
    return sub_dataset


if __name__ == "__main__":
    dataset = pd.read_csv(DATASET_PATH)
    sub_dataset = dataset.loc[dataset["CATEGORY"].isin(CATEGORIES)]
    tokenized_dataset = process_note(sub_dataset)
    tokenized_dataset.to_csv(TOKENIZED_DATASET_PATH)
