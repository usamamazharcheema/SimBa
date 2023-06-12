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
    data_name = "sv_ident_trial_train_and_val_variable_detection"
    Path("../../../data/variable_detection/"+data_name).mkdir(parents=True, exist_ok=True)
    general_path = "../../../data/variable_detection/"+data_name

    output_df_path = general_path + "/variable_detection_df.tsv"

    # get english trial data

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
    
    # get german trial data
    
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

    # both languages trial

    df_en_and_de = pd.concat([df_de, df_en])
    df_en_and_de = df_en_and_de[['uuid', 'text', 'is_variable']]

    # get train data

    queries_path_train = general_path + "/queries_train.tsv"

    queries_qrels_train = requests.get("https://github.com/vadis-project/sv-ident/raw/main/data/train/train.tsv")

    with open(queries_path_train, 'wb') as f:
        f.write(queries_qrels_train.content)

    train_df = pd.read_csv(queries_path_train, sep='\t') #sentence	is_variable	variable	research_data	doc_id	uuid	lang

    os.remove(queries_path_train)
    
    # get val data

    queries_path_val = general_path + "/queries_val.tsv"

    queries_qrels_val = requests.get("https://github.com/vadis-project/sv-ident/raw/main/data/train/val.tsv")

    with open(queries_path_val, 'wb') as f:
        f.write(queries_qrels_val.content)

    val_df = pd.read_csv(queries_path_val, sep='\t') #sentence	is_variable	variable	research_data	doc_id	uuid	lang

    os.remove(queries_path_val)

    # concatenate train and val data

    df_train_and_val = pd.concat([train_df, val_df])
    df_train_and_val = df_train_and_val[['uuid', 'sentence', 'is_variable']].rename(columns={'sentence':'text'})

    # concatenate all

    output_df = pd.concat([df_train_and_val, df_en_and_de])
    output_df.to_csv(output_df_path, sep='\t', header=False, index=False)


if __name__ == "__main__":
    run()