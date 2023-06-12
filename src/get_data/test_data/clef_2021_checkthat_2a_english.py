import os
import shutil

import requests
from zipfile import ZipFile
from pathlib import Path
from os.path import join
from os import listdir, rmdir
from shutil import move
from src.get_data import DATA_PATH
from src.utils import delete_first_line_of_tsv


def run():
    file_path = DATA_PATH+"clef_2021_checkthat_2a_english"
    Path(file_path).mkdir(parents=True, exist_ok=True)
    #Path("../../data/clef_2021_checkthat_2a_english").mkdir(parents=True, exist_ok=True)
    general_path = DATA_PATH+"clef_2021_checkthat_2a_english"
    queries_path = general_path + "/queries.tsv"
    queries = requests.get("https://gitlab.com/checkthat_lab/clef2021-checkthat-lab/-/raw/master/task2/test-input/subtask-2a--english.zip")
    with open(general_path+"/data.zip", 'wb') as f:
        f.write(queries.content)
    with ZipFile(general_path+"/data.zip", 'r') as zipObj:
        for file in zipObj.namelist():
            if file.startswith('subtask'):
                zipObj.extract(file, general_path)
    os.rename(general_path + "/subtask-2a--english/tweets-test.tsv", queries_path)
    delete_first_line_of_tsv(queries_path)
    shutil.rmtree(general_path + "/subtask-2a--english", ignore_errors=False, onerror=None)
    os.remove(general_path+"/data.zip")

    corpus = requests.get("https://gitlab.com/checkthat_lab/clef2021-checkthat-lab/-/raw/master/task2/data/subtask-2a--english/v1.zip")
    corpus_filepath = DATA_PATH+"clef_2021_checkthat_2a_english/corpus"
    with open(corpus_filepath+".zip", 'wb') as f:
        f.write(corpus.content)
    with ZipFile(corpus_filepath+".zip", 'r') as zipObj:
        for file in zipObj.namelist():
            zipObj.extract(file, corpus_filepath)
    for filename in listdir(corpus_filepath+'/train/vclaims'):
        move(join(corpus_filepath+'/train/vclaims/', filename), join(corpus_filepath, filename))
    shutil.rmtree(corpus_filepath+'/train', ignore_errors=False, onerror=None)
    os.remove(general_path + "/corpus.zip")

    gold_file_path = DATA_PATH+"clef_2021_checkthat_2a_english/gold.tsv"
    gold_file = requests.get("https://gitlab.com/checkthat_lab/clef2021-checkthat-lab/-/raw/master/task2/test-gold/subtask-2a--english.zip")
    with open(general_path+"/data.zip", 'wb') as f:
        f.write(gold_file.content)
    with ZipFile(general_path+"/data.zip", 'r') as zipObj:
        for file in zipObj.namelist():
            if file.startswith('subtask'):
                zipObj.extract(file, general_path)
    os.rename(general_path + "/subtask-2a--english/qrels-test.tsv", gold_file_path)
    shutil.rmtree(general_path + "/subtask-2a--english", ignore_errors=False, onerror=None)
    os.remove(general_path+"/data.zip")


if __name__ == "__main__":
    run()