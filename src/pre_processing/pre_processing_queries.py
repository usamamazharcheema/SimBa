import argparse
from pathlib import Path
import pandas as pd
from nltk.tokenize import sent_tokenize

from src.pre_processing import DATA_PATH


def incorporate_context_string(sentence, full_document, context_start=120, context_end=120):
    sentence_start = full_document.find(sentence)
    if sentence_start == int(-1):
        context = sentence
    else:
        sentence_end = sentence_start + len(sentence)
        context = full_document[sentence_start-context_start:sentence_end+context_end]
    return context


def incorporate_context_sentences(sentence, full_document, context_start=1, context_end=1):
    try:
        sentence_pos = full_document.index(sentence)
        if sentence_pos > 0:
            front_context = full_document[sentence_pos - 1]
        else:
            front_context = ""
        if sentence_pos < (len(full_document) - context_end):
            end_context = full_document[sentence_pos + 1]
        else:
            end_context = ""
        context = front_context+ " " + sentence+ " " + end_context
    except:
        context = sentence
    return context


def delete_surrounding_quotation_marks(sentence):
    if sentence[:3] == '"""':
        sentence = sentence[3:]
    if sentence[-3:] == '"""':
        sentence = sentence[:-3]
    return sentence


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_queries', type=str, help='Name under which the queries are stored.')
    parser.add_argument('data_orig', type=str, help='Name under which the original documents are stored.')
    parser.add_argument('data', type=str, help='Name under which the documents should be stored.')
    parser.add_argument('-context_start', type=int, default=1, help='How many context sentences in front of sentence.')
    parser.add_argument('-context_end', type=int, default=1, help='How many context sentences at the end of sentence.')
    args = parser.parse_args()

    output_dir = DATA_PATH + args.data + '/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_queries_data_path = output_dir + "/queries.tsv"

    with open(DATA_PATH + args.data_orig + "/full_text.txt", "r", encoding="utf-8") as text_file:
        document = text_file.read().rstrip()
    document_tokenized = sent_tokenize(document)
    all_sentences_df = pd.read_csv(DATA_PATH + args.data_queries + "/queries.tsv", sep='\t', dtype=str, header=None)
    all_sentences = all_sentences_df.iloc[:, 1].to_list()

    pp_sentences = []

    for sentence in all_sentences:
        sentence = delete_surrounding_quotation_marks(sentence)
        pp_sentences.append(incorporate_context_sentences(sentence, document_tokenized, args.context_start,args.context_end))

    queries_df = pd.DataFrame(columns=['uuid', 'text'])
    queries_df['uuid'] = all_sentences_df.iloc[:, 0]
    queries_df['text'] = pd.Series(pp_sentences)

    queries_df.to_csv(output_queries_data_path, sep='\t', header=False, index=False)


if __name__ == "__main__":
    run()