import shutil
import subprocess
from pathlib import Path


def run():

    data_name = "test"

    subprocess.call(["python",
                     "../../src/candidate_retrieval/retrieval.py",
                     "../../data/"+data_name+"/queries.tsv",
                     "../../data/"+data_name+"/corpus",
                     data_name,
                     data_name,
                     "braycurtis",
                     "50",
                     '-sentence_embedding_models', "all-mpnet-base-v2"
                     ])

    subprocess.call(["python",
                     "../../src/re_ranking/re_ranking.py",
                     "../../data/"+data_name+"/queries.tsv",
                     "../../data/"+data_name+"/corpus",
                     data_name,
                     data_name,
                     "braycurtis",
                     "5",
                     '-sentence_embedding_models', "all-mpnet-base-v2",
                     '-lexical_similarity_measures', "similar_words_ratio",
                     ])






if __name__ == "__main__":
    run()