import requests
from zipfile import ZipFile
from pathlib import Path
from os.path import join
from os import listdir, rmdir
from shutil import move

from src.get_data import DATA_PATH


def run():
    Path(DATA_PATH+"clef_2022_checkthat_2a_english").mkdir(parents=True, exist_ok=True)
    queries = requests.get('https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab/-/raw/main/task2/data/subtask-2a--english/test/CT2022-Task2A-EN-Test_Queries.tsv')
    with open(DATA_PATH+"clef_2022_checkthat_2a_english/queries.tsv", 'wb') as f:
        f.write(queries.content)
    corpus = requests.get('https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab/-/raw/main/task2/data/subtask-2a--english/vclaims_json.zip')
    corpus_filepath = DATA_PATH+"clef_2022_checkthat_2a_english/corpus"
    with open(corpus_filepath+".zip", 'wb') as f:
        f.write(corpus.content)
    with ZipFile(corpus_filepath+".zip", 'r') as zipObj:
        for file in zipObj.namelist():
            if file.startswith('vclaims'):
                zipObj.extract(file, corpus_filepath)
    for filename in listdir(join(corpus_filepath, 'vclaims')):
        move(join(corpus_filepath, 'vclaims', filename), join(corpus_filepath, filename))
    rmdir(join(corpus_filepath, 'vclaims'))
    gold_file = requests.get("https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab/-/raw/main/task2/data/subtask-2a--english/test/CT2022-Task2A-EN-Test_Qrels_gold.tsv")
    with open(DATA_PATH+"clef_2022_checkthat_2a_english/gold.tsv", 'wb') as f:
        f.write(gold_file.content)



if __name__ == "__main__":
    run()