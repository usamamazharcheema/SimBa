import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from tensorflow.python.framework.ops import EagerTensor

from scipy.spatial.distance import cdist
from pathlib import Path

base_path = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(base_path, "../../data/")
sys.path.append(os.path.join(base_path, "../create_similarity_features"))
sys.path.append(os.path.join(base_path, ".."))
import src.re_ranking
import src.create_similarity_features.lexical_similarity
import src.create_similarity_features.referential_similarity
import src.create_similarity_features.sentence_encoder
import src.create_similarity_features.string_similarity
sys.path.insert(0, os.path.join(base_path, ".."))
import src.utils

from src.utils import get_queries, get_targets, load_pickled_object, decompress_file, pickle_object, \
    compress_file, make_top_k_dictionary
from src.create_similarity_features.sentence_encoder import encode_queries, encode_targets
from src.create_similarity_features.referential_similarity import get_sequence_entities
from src.create_similarity_features.string_similarity import get_string_similarity
from src.create_similarity_features.lexical_similarity import get_lexical_entities

def run():
    """
    input:
    queries, targets
    output:
    {query: list of top k targets (ordered if union is not chosen)}

    all_sim_scores: {query_id: list_of_sim_scores, list entries are arrays of shape (1, target_n) with target_n similarity scores between 0 and 100}
    """
    parser = argparse.ArgumentParser()
    #input
    parser.add_argument('queries', type=str, help='Input queries path as tsv file.')
    parser.add_argument('targets', type=str, help='Input targets path as tsv file.')
    # parameters
    parser.add_argument('data_cache', type=str, help='Name under which the cached documents should be stored.')
    parser.add_argument('data', type=str, help='Name under which the documents should be stored.')
    parser.add_argument('similarity_measure', type=str, default='braycurtis', help='Distance measure for sentence embeddings')
    parser.add_argument('k', type=int, default=100, help='How many targets per queries should be retrieved')
    parser.add_argument('--union_of_top_k_per_feature', action="store_true",
                        help='How to combine the features: either take mean of different features or union of top k per feature. If not selected teh output is the top k of mean of features.')
    parser.add_argument('--gesis_unsup', action="store_true", help = 'cache targets for gesis unsup')
    parser.add_argument('--corpus_sizes', action="store_true",
                        help='cache targets only (for the time measurement of different corpus sizes)')
    parser.add_argument('-sentence_embedding_models', type=str, nargs='+', default=[],
                    help='Pass a list of sentence embedding models hosted by Huggingface or Tensorflow or simply pass "infersent" to use the infersent encoder.')
    parser.add_argument('-referential_similarity_measures', type=str, nargs='+', default=[])
    parser.add_argument('-lexical_similarity_measures', type=str, nargs='+', default=[],
                        help='Pass a list of lexical similarity measures to use')
    parser.add_argument('-string_similarity_measures', type=str, nargs='+', default=[])
    parser.add_argument('-discrete_similarity_measures', type=str, nargs='+', default=[])
    args = parser.parse_args()
    """
    Name datapaths and load queries and targets.
    """
    caching_directory = DATA_PATH + "cache/" + args.data_cache
    if args.gesis_unsup:
        caching_directory_targets = DATA_PATH + "cache/gesis_unsup_labels"
    elif args.corpus_sizes:
        caching_directory_targets = DATA_PATH + "cache/corpus_size_targets_"+args.data
    else:
        caching_directory_targets = caching_directory
    Path(caching_directory).mkdir(parents=True, exist_ok=True)
    Path(caching_directory_targets).mkdir(parents=True, exist_ok=True)
    queries = get_queries(args.queries)
    targets = get_targets(args.targets)
    all_features = []
    all_sim_scores = {}
    for query_id in list(queries.keys()):
        all_sim_scores[query_id] = []
    output_path_root = DATA_PATH + args.data
    Path(output_path_root).mkdir(parents=True, exist_ok=True)
    output_path = output_path_root + "/candidates"
    if args.k > len(targets):
        print("Reduced k to "+str(len(targets)))
        k = len(targets)
    else:
        k = args.k
    """
    0. Get number of tokens of
    0.1 queries
    0.2 targets
    0.3 pairs
    """
    # token_lens = {}
    # stored_token_lens = caching_directory + "/token_lens"
    # if os.path.exists(stored_token_lens + ".pickle" + ".zip"):
    #     token_lens = load_pickled_object(decompress_file(stored_token_lens+".pickle"+".zip"))
    # else:
    #     for query_id, query_text in queries.items():
    #         n_query_tokens = get_number_of_tokens(query_text)
    #         this_query_token_lens = {}
    #         for target_id, target_text in targets.items():
    #             n_target_tokens = get_number_of_tokens(target_text)
    #             this_query_token_lens[target_id] = n_query_tokens + n_target_tokens
    #         token_lens[query_id] = this_query_token_lens
    #     pickle_object(stored_token_lens, token_lens)
    #     compress_file(stored_token_lens + ".pickle")
    #     os.remove(stored_token_lens + ".pickle")
    """
    1. For all sentence embedding models:
    1.1 Embed all queries and cache
    1.2. Embed all targets and cache
    1.3. Calculate all similarity scores for all combinations -> value between 0 and 100 and cache
    """
    for model in args.sentence_embedding_models:
        print(model)
        all_features.append(model)
        if "/" or ":" or "." in str(model):
            model_name = str(model).replace("/", "_").replace(":", "_").replace(".", "_")
        else:
            model_name = str(model)
        stored_embedded_queries = caching_directory + "/embedded_queries_" + model_name
        stored_embedded_targets = caching_directory_targets + "/embedded_targets_" + model_name
        stored_sim_scores = caching_directory + "/sim_scores_" + model_name
        if os.path.exists(stored_sim_scores + ".pickle" + ".zip"):
            sim_scores_to_store = load_pickled_object(decompress_file(stored_sim_scores+".pickle"+".zip"))
            print("loaded sim scores")
            for query_id in list(queries.keys()):
                all_sim_scores[query_id].append(sim_scores_to_store[query_id])
        else:
            sim_scores_to_store = {}
            if os.path.exists(stored_embedded_queries + ".pickle" + ".zip"):
                embedded_queries = load_pickled_object(decompress_file(stored_embedded_queries+".pickle"+".zip"))
                print('queries loaded')
            else:
                print('compute queries')
                embedded_queries = encode_queries(queries, model)
                pickle_object(stored_embedded_queries, embedded_queries)
                compress_file(stored_embedded_queries + ".pickle")
                os.remove(stored_embedded_queries + ".pickle")
            if os.path.exists(stored_embedded_targets + ".pickle" + ".zip"):
                embedded_targets = load_pickled_object(decompress_file(stored_embedded_targets+".pickle"+".zip"))
                print('targets loaded')
            else:
                print('compute targets')
                embedded_targets = encode_targets(targets, model)
                pickle_object(stored_embedded_targets, embedded_targets)
                compress_file(stored_embedded_targets + ".pickle")
                os.remove(stored_embedded_targets + ".pickle")
            for query_id in list(queries.keys()):
                query_embedding = embedded_queries[query_id]
                if type(query_embedding) == EagerTensor:
                    query_embedding = query_embedding.numpy()
                query_embedding = query_embedding.reshape(1, -1)
                embedded_targets_array = np.array(list(embedded_targets.values()))
                sim_scores = (1 - cdist(query_embedding, embedded_targets_array,
                                        metric=args.similarity_measure)) * 100
                n_targets = sim_scores.shape[1]
                sim_scores = sim_scores.reshape(n_targets,)
                all_sim_scores[query_id].append(sim_scores)
                sim_scores_to_store[query_id] = sim_scores
            pickle_object(stored_sim_scores, sim_scores_to_store)
            compress_file(stored_sim_scores + ".pickle")
            os.remove(stored_sim_scores + ".pickle")
    """
    2. For all referential similarity measures\
    2.1 get entities for all queries and cache or load from cache\
    2.2. get entities for all targets and cache or load from cache\
    2.3. Calculate all similarity scores for one query and its *candidate targets* or load from cache -> value between 0 and 100 and cache
    """
    for ref_feature in args.referential_similarity_measures:
        print(ref_feature)
        all_features.append(ref_feature)
        stored_entities_queries = caching_directory + "/queries_" + str(ref_feature)
        stored_entities_targets = caching_directory_targets + "/targets_" + str(ref_feature)
        stored_sim_scores = caching_directory + "/sim_scores_" + ref_feature
        if os.path.exists(stored_sim_scores + ".pickle" + ".zip"):
            sim_scores_to_store = load_pickled_object(decompress_file(stored_sim_scores+".pickle"+".zip"))
            print("loaded sim scores")
            for query_id in list(queries.keys()):
                all_sim_scores[query_id].append(sim_scores_to_store[query_id])
        else:
            sim_scores_to_store = {}
            if os.path.exists(stored_entities_queries + ".pickle" + ".zip"):
                entities_queries = load_pickled_object(decompress_file(stored_entities_queries+".pickle"+".zip"))
                print('queries loaded')
            else:
                print('compute queries')
                entities_queries = get_sequence_entities(queries, ref_feature)
                pickle_object(stored_entities_queries, entities_queries)
                compress_file(stored_entities_queries + ".pickle")
                os.remove(stored_entities_queries + ".pickle")
            if os.path.exists(stored_entities_targets + ".pickle" + ".zip"):
                entities_targets = load_pickled_object(decompress_file(stored_entities_targets+".pickle"+".zip"))
                print('targets loaded')
            else:
                print('compute targets')
                entities_targets = get_sequence_entities(targets, ref_feature)
                pickle_object(stored_entities_targets, entities_targets)
                compress_file(stored_entities_targets + ".pickle")
                os.remove(stored_entities_targets + ".pickle")
            for query_id in list(queries.keys()):
                query_entities = set(entities_queries[query_id])
                sim_scores = np.zeros(len(list(targets.keys())))
                if query_entities:
                    len_query_entities = len(query_entities)
                    for idx, target_id in enumerate(list(targets.keys())):
                        target_entities = set(entities_targets[target_id])
                        len_target_entities = len(target_entities)
                        len_intersection = len(query_entities.intersection(target_entities))
                        ratio = (100/(len_query_entities+len_target_entities))*len_intersection
                        sim_scores[idx] = ratio
                else:
                    sim_scores[idx] = 0
                all_sim_scores[query_id].append(sim_scores)
                sim_scores_to_store[query_id] = sim_scores
            pickle_object(stored_sim_scores, sim_scores_to_store)
            compress_file(stored_sim_scores + ".pickle")
            os.remove(stored_sim_scores + ".pickle")
    """
    3. For all lexical similarity measures
    3.1 get entities for all queries and cache or load from cache\
    3.2. get entities for all targets and cache or load from cache\
    3.3. Calculate all similarity scores for all combinations -> value between 0 and 100 and cache
    """
    for lex_feature in args.lexical_similarity_measures:
        print(lex_feature)
        all_features.append(lex_feature)
        stored_entities_queries = caching_directory + "/queries_" + str(lex_feature)
        stored_entities_targets = caching_directory_targets + "/targets_" + str(lex_feature)
        stored_sim_scores = caching_directory + "/sim_scores_" + lex_feature
        if os.path.exists(stored_sim_scores + ".pickle" + ".zip"):
            sim_scores_to_store = load_pickled_object(decompress_file(stored_sim_scores+".pickle"+".zip"))
            print("loaded sim scores")
            for query_id in list(queries.keys()):
                all_sim_scores[query_id].append(sim_scores_to_store[query_id])
        else:
            sim_scores_to_store = {}
            if os.path.exists(stored_entities_queries + ".pickle" + ".zip"):
                entities_queries = load_pickled_object(decompress_file(stored_entities_queries+".pickle"+".zip"))
                print('queries loaded')
            else:
                print('compute queries')
                entities_queries = get_lexical_entities(queries, lex_feature)
                pickle_object(stored_entities_queries, entities_queries)
                compress_file(stored_entities_queries + ".pickle")
                os.remove(stored_entities_queries + ".pickle")
            if os.path.exists(stored_entities_targets + ".pickle" + ".zip"):
                entities_targets = load_pickled_object(decompress_file(stored_entities_targets+".pickle"+".zip"))
                print('targets loaded')
            else:
                print('compute targets')
                entities_targets = get_lexical_entities(targets, lex_feature)
                pickle_object(stored_entities_targets, entities_targets)
                compress_file(stored_entities_targets + ".pickle")
                os.remove(stored_entities_targets + ".pickle")
            for query_id in list(queries.keys()):
                query_entities = set(entities_queries[query_id])
                sim_scores = np.zeros(len(list(targets.keys())))
                if query_entities:
                    len_query_entities = len(query_entities)
                    for idx, target_id in enumerate(list(targets.keys())):
                        target_entities = set(entities_targets[target_id])
                        len_target_entities = len(target_entities)
                        len_intersection = len(query_entities.intersection(target_entities))
                        len_union = len_query_entities+len_target_entities
                        if lex_feature == "similar_words_ratio":
                            ratio = (100/(len_union/2))*len_intersection
                        elif lex_feature == "similar_words_ratio_length":
                            ratio = (100/len_union)*len_intersection
                        sim_scores[idx] = ratio
                all_sim_scores[query_id].append(sim_scores)
                sim_scores_to_store[query_id] = sim_scores
            pickle_object(stored_sim_scores, sim_scores_to_store)
            compress_file(stored_sim_scores + ".pickle")
            os.remove(stored_sim_scores + ".pickle")
    """
    4. For all string similarity measures
        4.1 Calculate all similarity scores for all combinations -> value between 0 and 100 and cache
    """
    for string_feature in args.string_similarity_measures:
        print(string_feature)
        all_features.append(string_feature)
        stored_sim_scores = caching_directory + "/sim_scores_" + string_feature
        sim_scores_to_store = {}
        if os.path.exists(stored_sim_scores + ".pickle" + ".zip"):
            sim_scores_to_store = load_pickled_object(decompress_file(stored_sim_scores+".pickle"+".zip"))
            print("loaded sim scores")
            for query_id in list(queries.keys()):
                all_sim_scores[query_id].append(sim_scores_to_store[query_id])
        else:
            for query_id in list(queries.keys()):
                sim_scores = np.zeros(len(list(targets.keys())))
                for idx, target_id in enumerate(list(targets.keys())):
                    query = queries[query_id]
                    target = targets[target_id]
                    sim_scores[idx] = get_string_similarity(query, target, string_feature)
                sim_scores_to_store[query_id] = sim_scores
                all_sim_scores[query_id].append(sim_scores)
            pickle_object(stored_sim_scores, sim_scores_to_store)
            compress_file(stored_sim_scores + ".pickle")
            os.remove(stored_sim_scores + ".pickle")            
    """
    5. For all discrete similarity measures\
    2.1 get entities for all queries and cache or load from cache\
    2.2. get entities for all targets and cache or load from cache\
    2.3. Calculate all similarity scores for one query and its *candidate targets* or load from cache -> value between 0 and 100 and cache
    """
    for discrete_feature in args.discrete_similarity_measures:
        print(discrete_feature)
        all_features.append(discrete_feature)
        stored_entities_queries = caching_directory + "/queries_" + str(discrete_feature)
        stored_entities_targets = caching_directory_targets + "/targets_" + str(discrete_feature)
        stored_sim_scores = caching_directory + "/sim_scores_" + discrete_feature + "_discrete"
        if os.path.exists(stored_sim_scores + ".pickle" + ".zip"):
            sim_scores_to_store = load_pickled_object(decompress_file(stored_sim_scores+".pickle"+".zip"))
            print("loaded sim scores")
            for query_id in list(queries.keys()):
                all_sim_scores[query_id].append(sim_scores_to_store[query_id])
        else:
            sim_scores_to_store = {}
            if os.path.exists(stored_entities_queries + ".pickle" + ".zip"):
                entities_queries = load_pickled_object(decompress_file(stored_entities_queries+".pickle"+".zip"))
                print('queries loaded')
            else:
                print('compute queries')
                if discrete_feature == "similar_words_ratio" or lex_feature == "similar_words_ratio_length":
                    entities_queries = get_lexical_entities(queries, discrete_feature)
                else:
                    entities_queries = get_sequence_entities(queries, discrete_feature)
                pickle_object(stored_entities_queries, entities_queries)
                compress_file(stored_entities_queries + ".pickle")
                os.remove(stored_entities_queries + ".pickle")
            if os.path.exists(stored_entities_targets + ".pickle" + ".zip"):
                entities_targets = load_pickled_object(decompress_file(stored_entities_targets+".pickle"+".zip"))
                print('targets loaded')
            else:
                print('compute targets')
                if discrete_feature == "similar_words_ratio" or lex_feature == "similar_words_ratio_length":
                    entities_targets = get_lexical_entities(targets, discrete_feature)
                else:
                    entities_targets = get_sequence_entities(targets, discrete_feature)
                pickle_object(stored_entities_targets, entities_targets)
                compress_file(stored_entities_targets + ".pickle")
                os.remove(stored_entities_targets + ".pickle")
            for query_id in list(queries.keys()):
                query_entities = set(entities_queries[query_id])
                sim_scores = np.zeros(len(list(targets.keys())))
                if query_entities:
                    for idx, target_id in enumerate(list(targets.keys())):
                        target_entities = set(entities_targets[target_id])
                        len_intersection = len(query_entities.intersection(target_entities))
                        sim_scores[idx] = len_intersection
                else:
                    sim_scores[idx] = 0
                all_sim_scores[query_id].append(sim_scores)
                sim_scores_to_store[query_id] = sim_scores
            pickle_object(stored_sim_scores, sim_scores_to_store)
            compress_file(stored_sim_scores + ".pickle")
            os.remove(stored_sim_scores + ".pickle")
    """
    Evaluation step:
    Get mean and variance of all different similarity scores to better understand how to normalize them
    """
    all_sim_scores_df = pd.DataFrame.from_dict(all_sim_scores, orient='index', columns=all_features )
    for feature in all_features:
        print(feature)
        sim_scores = all_sim_scores_df[feature].to_numpy().flatten()
        sim_scores_mean = np.mean(np.mean(sim_scores, axis=0), axis=0)
        print(round(sim_scores_mean, 3))
    """
    6. get top k targets per query:
    6.1. either create union of features and compute top k
    6.2. or compute mean of features and compute top k  
    """
    output = {}
    for query_id in list(queries.keys()):
        output[query_id] = []
    feature_top_ks = {}
    if args.union_of_top_k_per_feature:
        for feature in all_features:
            this_feature_sim_scores = np.stack(all_sim_scores_df[feature])
            sim_scores_top_k_values, sim_scores_top_k_idx = torch.topk(torch.tensor(this_feature_sim_scores), k=k,
                                                                   dim=1, largest=True)
            sim_scores_top_k_values = sim_scores_top_k_values.cpu().tolist()
            sim_scores_top_k_idx = sim_scores_top_k_idx.cpu().tolist()
            this_feature_top_k = make_top_k_dictionary(list(queries.keys()), list(targets.keys()), sim_scores_top_k_values, sim_scores_top_k_idx)
            feature_top_ks[feature] = this_feature_top_k
        for query_id in list(queries.keys()):
            for feature in all_features:
                this_feature_top_k = feature_top_ks[feature]
                output[query_id].extend(this_feature_top_k[query_id].keys())
                output[query_id] = list(set(output[query_id]))
    else:
        all_features_sim_scores = np.stack(all_sim_scores_df.to_numpy())
        sim_scores_mean = np.stack(np.mean(all_features_sim_scores, axis=1))
        sim_scores_top_k_values, sim_scores_top_k_idx = torch.topk(torch.tensor(sim_scores_mean), k=k,
                                                                   dim=1, largest=True)
        sim_scores_top_k_values = sim_scores_top_k_values.cpu().tolist()
        sim_scores_top_k_idx = sim_scores_top_k_idx.cpu().tolist()
        mean_top_k = make_top_k_dictionary(list(queries.keys()), list(targets.keys()), sim_scores_top_k_values,
                                                   sim_scores_top_k_idx)
        for query_id in list(queries.keys()):
            output[query_id].extend(mean_top_k[query_id].keys())
            output[query_id] = list(set(output[query_id]))

    pickle_object(output_path, output)
    compress_file(output_path + ".pickle")
    os.remove(output_path + ".pickle")


if __name__ == "__main__":
    run()
