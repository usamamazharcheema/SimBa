import os
import shutil
import subprocess
import time


def run():

    data_name_queries = 'clef_2020_checkthat_2_english'

    # # create different corpora
    # subprocess.call(["python", "../../src/get_data/draw_sample.py"])
    corpus_sizes = ['1k', '5k', '10k']
    for corpus_size in corpus_sizes:

        data_name = "check_that" + corpus_size
        # delete old cache
        caching_path_old_cache = "data/cache/" + data_name
        if os.path.exists(caching_path_old_cache):
            shutil.rmtree(caching_path_old_cache, ignore_errors=False, onerror=None)

        data_name_corpus = "clef_2020_checkthat_2_english/corpus_" + corpus_size + "_sample.tsv"

        caching_path = "data/cache/corpus_size_targets_"+data_name
        if os.path.exists(caching_path):
            shutil.rmtree(caching_path, ignore_errors=False, onerror=None)

        print("retrieval in order to cache target embeddings")
        subprocess.call(["python",
                         "src/candidate_retrieval/retrieval.py",
                         "data/" + data_name_queries + "/queries.tsv",
                         "data/" + data_name_corpus,
                         data_name,
                         data_name,
                         "braycurtis",
                         "10",
                         "--union_of_top_k_per_feature",
                         "--corpus_sizes",
                         '-sentence_embedding_models', "all-mpnet-base-v2", "sentence-transformers/sentence-t5-base", "princeton-nlp/unsup-simcse-roberta-base",
                         '-lexical_similarity_measures', "similar_words_ratio"
                         ])

        print("delete query embeddings and keep target embeddings")
        caching_path = "data/cache/" + data_name
        candidates_path = "data/" + data_name + "candidates.pickle.zip"
        # targets are cached and loaded from specific directory (because keyword --corpus_sizes is given)
        # everything else is deleted from teh cache
        if os.path.exists(caching_path):
            shutil.rmtree(caching_path, ignore_errors=False, onerror=None)
        if os.path.exists(candidates_path):
            shutil.rmtree(candidates_path, ignore_errors=False, onerror=None)

        # Start measuring time here
        print("Measuring time for corpus size of " + corpus_size)
        start_time = time.time() # Measure time for retrieval and re-ranking

        subprocess.call(["python",
                         "src/candidate_retrieval/retrieval.py",
                         "data/"+data_name_queries+"/queries.tsv",
                         "data/"+data_name_corpus,
                         data_name,
                         data_name,
                         "braycurtis",
                         "10",
                         "--union_of_top_k_per_feature",
                         "--corpus_sizes",
                         '-sentence_embedding_models', "all-mpnet-base-v2"
                         ])

        subprocess.call(["python",
                         "src/re_ranking/re_ranking.py",
                         "data/"+data_name_queries+"/queries.tsv",
                         "data/"+data_name_corpus,
                         data_name,
                         data_name,
                         "braycurtis",
                         "10",
                         "--corpus_sizes",
                         '-sentence_embedding_models', "all-mpnet-base-v2", "sentence-transformers/sentence-t5-base", "princeton-nlp/unsup-simcse-roberta-base",
                         '-lexical_similarity_measures', "similar_words_ratio"
                         ])

        print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    run()