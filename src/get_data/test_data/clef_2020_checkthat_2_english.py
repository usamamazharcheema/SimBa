import os
import shutil

import requests
from zipfile import ZipFile
from pathlib import Path
from os.path import join
from os import listdir, rmdir
from shutil import move

from src.get_data import DATA_PATH
from src.utils import delete_first_line_of_tsv, prepare_corpus_tsv


def run():
    file_path = DATA_PATH+"clef_2020_checkthat_2_english"
    Path(file_path).mkdir(parents=True, exist_ok=True)
    file_url = requests.get("https://github.com/sshaar/clef2020-factchecking-task2/raw/master/test-input/test-input.zip")

    with open(file_path + "/clef_2020_checkthat_2_english.zip", 'wb') as f:
        f.write(file_url.content)
    with ZipFile(file_path + "/clef_2020_checkthat_2_english.zip", 'r') as zipObj:
        for file in zipObj.namelist():
            zipObj.extract(file, file_path)
    delete_first_line_of_tsv(file_path+"/tweets.queries.tsv")
    os.rename(file_path+"/tweets.queries.tsv", file_path+"/queries.tsv")

    file_url = requests.get("https://github.com/sshaar/clef2020-factchecking-task2/raw/master/data/v3.zip")
    file_path = file_path
    with open(file_path + ".zip", 'wb') as f:
        f.write(file_url.content)
    with ZipFile(file_path + ".zip", 'r') as zipObj:
        for file in zipObj.namelist():
            zipObj.extract(file, file_path)

    corpus_path = file_path+"/v3.0/verified_claims.docs.tsv"
    prepare_corpus_tsv(corpus_path)
    os.rename(corpus_path, file_path+"/corpus")

    os.rename(file_path+"/tweet-vclaim-pairs.qrels", file_path + "/gold.tsv")

    os.remove(file_path + "/clef_2020_checkthat_2_english.zip")
    os.remove(DATA_PATH+"clef_2020_checkthat_2_english.zip")
    shutil.rmtree(file_path+"/v3.0", ignore_errors=False, onerror=None)


if __name__ == "__main__":
    run()