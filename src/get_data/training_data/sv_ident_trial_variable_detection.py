import os
from pathlib import Path
import pandas as pd
import requests

from src.get_data import DATA_PATH


def get_variable_names(sequence):
    splitted = sequence.split("-")
    variable = splitted[0]
    relevance = splitted[1]
    if relevance == "Yes":
        return True, "v"+variable
    else:
        return False, variable

def run():

    # queries
    # qrels/gold
    # targets
    data_name = "sv_ident_trial_variable_detection"
    Path("../../../data/variable_detection/"+data_name).mkdir(parents=True, exist_ok=True)
    general_path = "../../../data/variable_detection/"+data_name

    output_df_path = general_path + "/variable_detection_df.tsv"

    # get english data

    queries_path_en_train = general_path + "/queries_en_train.tsv"
    queries_path_en_test = general_path + "/queries_en_test.tsv"

    queries_qrels_en_test = requests.get("https://github.com/vadis-project/sv-ident/raw/main/data/trial/test/en.tsv")
    queries_qrels_en_train = requests.get("https://github.com/vadis-project/sv-ident/raw/main/data/trial/train/en.tsv")

    with open(queries_path_en_test, 'wb') as f:
        f.write(queries_qrels_en_test.content)

    with open(queries_path_en_train, 'wb') as f:
        f.write(queries_qrels_en_train.content)

    df_en_test = pd.read_csv(queries_path_en_test, sep='\t')
    df_en_train = pd.read_csv(queries_path_en_train, sep='\t')
    df_en = pd.concat([df_en_test, df_en_train])

    os.remove(general_path + "/queries_en_train.tsv")
    os.remove(general_path + "/queries_en_test.tsv")
    
    # get german data
    
    queries_path_de_train = general_path + "/queries_de_train.tsv"
    queries_path_de_test = general_path + "/queries_de_test.tsv"

    queries_qrels_de_test = requests.get("https://github.com/vadis-project/sv-ident/raw/main/data/trial/test/de.tsv")
    queries_qrels_de_train = requests.get("https://github.com/vadis-project/sv-ident/raw/main/data/trial/train/de.tsv")

    with open(queries_path_de_test, 'wb') as f:
        f.write(queries_qrels_de_test.content)

    with open(queries_path_de_train, 'wb') as f:
        f.write(queries_qrels_de_train.content)

    df_de_test = pd.read_csv(queries_path_de_test, sep='\t')
    df_de_train = pd.read_csv(queries_path_de_train, sep='\t')
    df_de = pd.concat([df_de_test, df_de_train])

    os.remove(general_path + "/queries_de_train.tsv")
    os.remove(general_path + "/queries_de_test.tsv")

    # both languages

    df_en_and_de = pd.concat([df_de, df_en])
    df_en_and_de = df_en_and_de[['uuid', 'text', 'is_variable']]

    df_en_and_de.to_csv(output_df_path, sep='\t', header=False, index=False)


if __name__ == "__main__":
    run()