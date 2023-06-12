import os
import shutil
import subprocess
import time
def run():

    base_path = os.path.abspath(os.path.dirname(__file__))
    times = []
    offline_times = []
    for size in ["1k", "1k", "1k", "1k", "5k", "5k", "5k", "10k", "10k", "10k"]:
        dataset = os.path.join(base_path, "..", size + "_sample.tsv")
        queries_path = os.path.join(base_path, "../claimlinking_riet/claimlinking_clef2020-factchecking-task2/test-input/tweets.queries.tsv")
        # get the verified claims embeddings
        data_name = "check_that" + size
        # delete old cache
        caching_path_old_cache = base_path + "/data/cache/" + data_name
        if os.path.exists(caching_path_old_cache):
            shutil.rmtree(caching_path_old_cache, ignore_errors=False, onerror=None)

        #data_name_corpus = "clef_2020_checkthat_2_english/corpus_" + size + "_sample.tsv"
        data_name_corpus = base_path + "/../" + size + "_sample.tsv"

        caching_path = base_path + "/data/cache/corpus_size_targets_"+data_name
        if os.path.exists(caching_path):
            shutil.rmtree(caching_path, ignore_errors=False, onerror=None)


        offline_start_time = time.time()
        subprocess.call(["python",
                         base_path + "/src/candidate_retrieval/retrieval.py",
                         queries_path,
                         data_name_corpus,
                         data_name,
                         data_name,
                         "braycurtis",
                         "50",
                         "--union_of_top_k_per_feature",
                         "--corpus_sizes",
                         '-sentence_embedding_models', "all-mpnet-base-v2", "princeton-nlp/unsup-simcse-roberta-base",
                         '-lexical_similarity_measures', "similar_words_ratio"
                         ])

        subprocess.call(["python",
                         base_path + "/src/re_ranking/re_ranking.py",
                         queries_path,
                         data_name_corpus,
                         data_name,
                         data_name,
                         "braycurtis",
                         "5",
                         "--corpus_sizes",
                         '-sentence_embedding_models', "all-mpnet-base-v2", "princeton-nlp/unsup-simcse-roberta-base",
                         '-lexical_similarity_measures', "similar_words_ratio"
                         ])
        offline_times.append(time.time() - offline_start_time)

        print("delete query embeddings and keep target embeddings")
        caching_path = base_path + "/data/cache/" + data_name
        candidates_path = base_path + "/data/" + data_name + "candidates.pickle.zip"
        # targets are cached and loaded from specific directory (because keyword --corpus_sizes is given)
        # everything else is deleted from teh cache
        if os.path.exists(caching_path):
            shutil.rmtree(caching_path, ignore_errors=False, onerror=None)
        if os.path.exists(candidates_path):
            shutil.rmtree(candidates_path, ignore_errors=False, onerror=None)
        
        print("Measuring time for %s" %dataset)
        start_time = time.time() # Measure time for retrieval and re-ranking
        subprocess.call(["python",
                         base_path + "/src/candidate_retrieval/retrieval.py",
                         queries_path,
                         data_name_corpus,
                         data_name,
                         data_name,
                         "braycurtis",
                         "50",
                         "--union_of_top_k_per_feature",
                         "--corpus_sizes",
                         '-sentence_embedding_models', "all-mpnet-base-v2"
                         ])

        subprocess.call(["python",
                         base_path + "/src/re_ranking/re_ranking.py",
                         queries_path,
                         data_name_corpus,
                         data_name,
                         data_name,
                         "braycurtis",
                         "5",
                         "--corpus_sizes",
                         '-sentence_embedding_models', "all-mpnet-base-v2", "princeton-nlp/unsup-simcse-roberta-base",
                         '-lexical_similarity_measures', "similar_words_ratio"
                         ])
        times.append(time.time() - start_time)
        # Check if output file was created correctly by evaluating with standard evaluation script
        # Don't measure exceution time for that
        clef2020_path = os.path.join(base_path, "../claimlinking_riet/claimlinking_clef2020-factchecking-task2/") 
        clef2022_path = os.path.join(base_path, "../claimlinking_riet/claimlinking_clef2022-checkthat-lab/task2/")
        pred_file_path = base_path + "/data/"+data_name+"/pred_qrels.tsv"
        subprocess.call(["python", os.path.join(clef2022_path, "scorer/main.py"), "--pred-file-path", pred_file_path, "--gold-file-path", os.path.join(clef2020_path, "test-input/tweet-vclaim-pairs.qrels")], cwd=clef2022_path)
    return times, offline_times


if __name__ == "__main__":
    n_input_claims = 200
    times, offline_times = run()
    for time in times[1:]:
        print("Online retrieval: needed %.3f seconds" %time)
    for offline_time in offline_times[1:]:
        print("Offline processing: needed %.3f seconds" %offline_time)
    
    avg_time_1k = sum(times[1:4]) / 3.0
    avg_time_5k = sum(times[4:7]) / 3.0
    avg_time_10k = sum(times[7:]) / 3.0
    print("Average time for matching %d input claims to 1k verified claims: %.3f" %(n_input_claims, avg_time_1k))
    print("Average time for matching 1 claim to 1k verified claims: %.3f" %(avg_time_1k/n_input_claims))
    print(str(times[1:4]))
    print()
    print("Average time for matching %d input claims to 5k verified claims: %.3f" %(n_input_claims, avg_time_5k))
    print("Average time for matching 1 claim to 5k verified claims: %.3f" %(avg_time_5k/n_input_claims))
    print(str(times[4:7]))
    print()
    print("Average time for matching %d input claims to 10k verified claims: %.3f" %(n_input_claims, avg_time_10k))
    print("Average time for matching 1 claim to 10k verified claims: %.3f" %(avg_time_10k/n_input_claims))
    print(str(times[7:]))
