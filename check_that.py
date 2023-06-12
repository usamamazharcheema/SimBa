import os
import shutil
import subprocess
from pathlib import Path


def run():

    # Data names for SimBa as independent repo
    # # data_names = ['clef_2020_checkthat_2_english', 'clef_2021_checkthat_2a_english', 'clef_2021_checkthat_2b_english', 'clef_2022_checkthat_2a_english', 'clef_2022_checkthat_2b_english']
    # Data names for SimBa as submodule
    data_names = ['2020-2a', '2021-2a', '2021-2b', '2022-2a', '2022-2b']
    base_path = "../"
    directionary_path = os.path.dirname(__file__)
    for data_name in data_names:

        # data name queries for SimBa as independent repo
        #  # data_name_queries =  "data/"+data_name+"/queries.tsv",
        # Data name queries for SimBa as submodule
        if data_name == '2020-2a':
            data_name_queries = os.path.join(base_path, "claimlinking_riet/claimlinking_clef2020-factchecking-task2/test-input/tweets.queries.tsv")
        elif data_name == '2021-2a':
            data_name_queries = os.path.join(base_path, "claimlinking_riet/claimlinking_clef2021-checkthat-lab/task2/test-gold/subtask-2a--english/tweets-test.tsv")
        elif data_name == '2021-2b':
            data_name_queries = os.path.join(base_path, "claimlinking_riet/claimlinking_clef2021-checkthat-lab/task2/test-gold/subtask-2b--english/queries.tsv")
        elif data_name == '2022-2a':
            data_name_queries = os.path.join(base_path, "claimlinking_riet/claimlinking_clef2022-checkthat-lab/task2/data/subtask-2a--english/test/CT2022-Task2A-EN-Test_Queries.tsv")
        elif data_name == '2022-2b':
            data_name_queries = os.path.join(base_path, "claimlinking_riet/claimlinking_clef2022-checkthat-lab/task2/data/subtask-2b--english/test/CT2022-Task2B-EN-Test_Queries.tsv")

        # data name targets for SimBa as independent repo
        #  # data_name_targets =  "data/" + data_name + "/corpus",
        # Data name targets for SimBa as submodule
        if data_name == '2020-2a':
            data_name_targets = base_path + "claimlinking_riet/claimlinking_clef2020-factchecking-task2/data/v3.0/verified_claims.docs.tsv"
        else:
            data_name_targets = base_path + data_name + "-vclaims.tsv"

        subprocess.call(["python",
                         directionary_path + "/src/candidate_retrieval/retrieval.py",
                         data_name_queries,
                         data_name_targets,
                         data_name,
                         data_name,
                         "braycurtis",
                         "50",
                         '-sentence_embedding_models', "all-mpnet-base-v2"
                         ])

        subprocess.call(["python",
                         directionary_path + "/src/re_ranking/re_ranking.py",
                         data_name_queries,
                         data_name_targets,
                         data_name,
                         data_name,
                         "braycurtis",
                         "5",
                         '-sentence_embedding_models', "all-mpnet-base-v2", "princeton-nlp/unsup-simcse-roberta-base",
                         '-lexical_similarity_measures', "similar_words_ratio", "similar_words_ratio_length"
                         ])

        # data name gold for SimBa as independent repo
        #  # data_name_gold =  "data/"+data_name+"/gold.tsv"
        # Data name gold for SimBa as submodule

        if data_name == '2020-2a':
            data_name_gold = os.path.join(base_path, "claimlinking_riet/claimlinking_clef2020-factchecking-task2/test-input/tweet-vclaim-pairs.qrels")
        elif data_name == '2021-2a':
            data_name_gold = os.path.join(base_path, "claimlinking_riet/claimlinking_clef2021-checkthat-lab/task2/test-gold/subtask-2a--english/qrels-test.tsv")
        elif data_name == '2021-2b':
            data_name_gold = os.path.join(base_path, "claimlinking_riet/claimlinking_clef2021-checkthat-lab/task2/test-gold/subtask-2b--english/task2b-test.tsv")
        elif data_name == '2022-2a':
            data_name_gold = os.path.join(base_path, "claimlinking_riet/claimlinking_clef2022-checkthat-lab/task2/data/subtask-2a--english/test/CT2022-Task2A-EN-Test_Qrels_gold.tsv")
        elif data_name == '2022-2b':
            data_name_gold = os.path.join(base_path, "claimlinking_riet/claimlinking_clef2022-checkthat-lab/task2/data/subtask-2b--english/test/CT2022-Task2B-EN-Test_Qrels_gold.tsv")

        print("Evaluation Scores for dataset "+ data_name)
        subprocess.call(["python", directionary_path + "/evaluation/scorer/main.py",
                         data_name_gold,
                         "data/" + data_name + "/pred_qrels.tsv"])

        Path("run0").mkdir(parents=True, exist_ok=True)

        output_file = "data/" + data_name + "/pred_qrels.tsv"
        if data_name == '2020-2a':
            new_file = "run0/2020.tsv"
        if data_name == '2021-2a':
            new_file = "run0/2021a.tsv"
        if data_name == '2021-2b':
            new_file = "run0/2021b.tsv"
        if data_name == '2022-2a':
            new_file = "run0/2022a.tsv"
        if data_name == '2022-2b':
            new_file = "run0/2022b.tsv"

        shutil.copy(output_file, new_file)


if __name__ == "__main__":
    run()