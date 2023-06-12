import argparse
import os
import sys

import numpy as np

import tensorflow as tf
import pandas as pd
from sklearn import svm, naive_bayes, preprocessing
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest

from tensorflow.python.framework.ops import EagerTensor

from scipy.spatial.distance import cdist
from pathlib import Path

base_path = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(base_path, "../../data/")
sys.path.append(os.path.join(base_path, "../create_similarity_features"))
sys.path.append(os.path.join(base_path, "../learning"))
sys.path.append(os.path.join(base_path, ".."))
import re_ranking
import lexical_similarity
import referential_similarity
import sentence_encoder
import string_similarity
sys.path.insert(0, os.path.join(base_path, ".."))
import utils
import create_feature_set


from utils import get_queries, get_targets, all_targets_as_query_candidates, load_pickled_object, \
    decompress_file, get_candidate_targets, pickle_object, compress_file, supervised_output_to_pred_qrels, \
    output_dict_to_pred_qrels
from sentence_encoder import encode_queries, encode_targets
from referential_similarity import get_sequence_entities
from string_similarity import get_string_similarity
from lexical_similarity import get_lexical_entities
from create_feature_set import create_feature_set, create_test_set

#classifier = LogisticRegression()
#classifier = svm.SVC(probability=True)
classifier = naive_bayes.MultinomialNB()
scaler = preprocessing.MinMaxScaler()
#sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel = SelectKBest(f_classif, k=4)
# scaler = StandardScaler()

def run():
    """
    input:
    queries, targets, {query: list of top k targets (ordered if union is not chosen)}
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
    parser.add_argument('--supervised', action="store_true",
                        help='If selected re-ranking is based on learning true pairs using the similarity features of the training data')
    parser.add_argument('--union', action="store_true",
                        help='If selected re-ranking is based on the conjuction of fifferent features, not their mean.')
    parser.add_argument('--ranking_only', action="store_true",
                        help='If selected all targets are selected as candidates.')
    parser.add_argument('--gesis_unsup', action="store_true", help='cache targets for gesis unsup')
    parser.add_argument('--corpus_sizes', action="store_true", help='cache targets only (for the time measurement of different corpus sizes)')
    parser.add_argument('-sentence_embedding_models', type=str, nargs='+',
                    default=[],
                    help='Pass a list of sentence embedding models hosted by Huggingface or Tensorflow or simply pass "infersent" to use the infersent encoder.')
    parser.add_argument('-referential_similarity_measures', type=str, nargs='+',
                        default=[])
    parser.add_argument('-lexical_similarity_measures', type=str, nargs='+', default=[],
                        help='Pass a list of lexical similarity measures to use')

    # parser.add_argument('-string_similarity_measures', type=str, nargs='+', default=["levenshtein_similarity"])
    # parser.add_argument('-referential_similarity_measures', type=str, nargs='+',
    #                     default=["ne_similarity"])
    parser.add_argument('-string_similarity_measures', type=str, nargs='+', default=[])
    parser.add_argument('-discrete_similarity_measures', type=str, nargs='+', default=[])
    args = parser.parse_args()
    """
    Name datapaths and load queries and targets.
    queries: {query_id: text}
    targets: {target_id: text}
    candidates: {query_id: list of candidate target ids}
    candidate_targets: {candidate_target_id (taken from all possible canidates): target_text}
    ---
    possibly stored files:
    stored_sim_scores: {query_id: list of sim scores for all targets in order of original targets}
    """
    caching_directory = os.path.join(DATA_PATH, "cache", args.data_cache)
    if args.gesis_unsup:
        caching_directory_targets = os.path.join(DATA_PATH, "cache/gesis_unsup_labels")
    elif args.corpus_sizes:
        caching_directory_targets = os.path.join(DATA_PATH, "cache/corpus_size_targets_"+args.data)
    else:
        caching_directory_targets = caching_directory
    Path(caching_directory).mkdir(parents=True, exist_ok=True)
    Path(caching_directory_targets).mkdir(parents=True, exist_ok=True)
    queries = get_queries(args.queries)
    targets = get_targets(args.targets)
    original_target_ids = list(targets.keys())

    if args.ranking_only:
        candidates = all_targets_as_query_candidates(list(queries.keys()), list(targets.keys()))
    else:
        candidates_path = os.path.join(DATA_PATH, args.data, "candidates")
        candidates = load_pickled_object(decompress_file(candidates_path+".pickle"+".zip"))
    candidate_targets = get_candidate_targets(candidates, targets)
    candidate_target_ids = list(candidate_targets.keys())
    all_features = []
    all_sim_scores = {}
    for query_id in list(queries.keys()):
        all_sim_scores[query_id] = []
    output_path = os.path.join(DATA_PATH, args.data, "pred_qrels.tsv")
    Path(os.path.join(DATA_PATH, args.data)).mkdir(parents=True, exist_ok=True)
    """
    0. Learning
    """
    if args.supervised:
        training_df = create_feature_set(args.data, targets, args.similarity_measure, args.sentence_embedding_models, args.referential_similarity_measures,
                            args.lexical_similarity_measures, args.string_similarity_measures)
        X_train = training_df.iloc[:,2:-1]
        y_train = training_df.iloc[:,-1:]
        y_train = y_train.values.ravel()
       # X_train = sel.fit_transform(X_train)
        X_train = sel.fit_transform(X_train, y_train)
        print(sel.get_feature_names_out())
        X_train = scaler.fit_transform(X_train, y_train)
        classifier.fit(X_train, y_train)
    """
    1. For all sentence embedding models\
    1.1 Embed all queries and cache or load from cache\
    1.2. Embed all *relevant targets* or load from cache\
    1.3. Calculate all similarity scores for one query and its *candidate targets* or load from cache -> value between 0 and 100 and cache   
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
        sim_scores_to_store = {}
        if os.path.exists(stored_sim_scores + ".pickle" + ".zip"):
            sim_scores_to_store = load_pickled_object(decompress_file(stored_sim_scores+".pickle"+".zip"))
            print("loaded sim scores")
            for query_id in list(queries.keys()):
                current_candidate_ids = candidates[query_id]
                current_sim_scores = sim_scores_to_store[query_id]
                sim_scores_dict = dict(zip(original_target_ids, current_sim_scores))
                n_targets = len(current_candidate_ids)
                c_list = [sim_scores_dict[target_id] for target_id in current_candidate_ids]
                current_candidate_sim_scores = np.array(c_list).reshape(n_targets,)
                all_sim_scores[query_id].append(current_candidate_sim_scores)
        else:
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
                # keep only candidate targets
                embedded_candidate_targets = {key: embedded_targets[key] for key in candidate_target_ids}
                #embedded_candidate_targets = dict((k[embedded_targets.values()]) for k in candidate_target_ids if k in list(embedded_targets.values()))
            else:
                print('compute targets')
                embedded_candidate_targets = encode_targets(candidate_targets, model)
                if args.ranking_only:
                    pickle_object(stored_embedded_targets, embedded_candidate_targets)
                    compress_file(stored_embedded_targets + ".pickle")
                    os.remove(stored_embedded_targets + ".pickle")
            for query_id in list(queries.keys()):
                if type(embedded_queries[query_id]) == EagerTensor:
                    query_embedding = tf.experimental.numpy.reshape(embedded_queries[query_id], (1, -1))
                else:
                    query_embedding = embedded_queries[query_id].reshape(1, -1)
                current_candidate_ids = candidates[query_id]
                current_embedded_candidate_targets = {key: embedded_candidate_targets[key] for key in current_candidate_ids}
                embedded_targets_array = np.array(list(current_embedded_candidate_targets.values()))
                sim_scores = (1 - cdist(query_embedding, embedded_targets_array,
                                        metric=args.similarity_measure)) * 100
                n_targets = len(current_candidate_ids)
                c_list = sim_scores.tolist()
                current_candidate_sim_scores = np.array(c_list).reshape(n_targets, )
                all_sim_scores[query_id].append(current_candidate_sim_scores)
                sim_scores_to_store[query_id] = current_candidate_sim_scores
            if args.ranking_only:
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
        sim_scores_to_store = {}
        if os.path.exists(stored_sim_scores + ".pickle" + ".zip"):
            sim_scores_to_store = load_pickled_object(decompress_file(stored_sim_scores+".pickle"+".zip"))
            print("loaded sim scores")
            for query_id in list(queries.keys()):
                current_candidate_ids = candidates[query_id]
                current_sim_scores = sim_scores_to_store[query_id]
                sim_scores_dict = dict(zip(original_target_ids, current_sim_scores))
                n_targets = len(current_candidate_ids)
                c_list = [sim_scores_dict[target_id] for target_id in current_candidate_ids]
                current_candidate_sim_scores = np.array(c_list).reshape(n_targets,)
                all_sim_scores[query_id].append(current_candidate_sim_scores)
        else:
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
                entities_candidate_targets = {k: entities_targets[k] for k in candidate_target_ids}
            else:
                print('compute targets')
                entities_candidate_targets = get_sequence_entities(candidate_targets, ref_feature)
                if args.ranking_only:
                    pickle_object(stored_entities_targets, entities_candidate_targets)
                    compress_file(stored_entities_targets + ".pickle")
                    os.remove(stored_entities_targets + ".pickle")
            for query_id in list(queries.keys()):
                current_candidate_ids = candidates[query_id]
                query_entities = set(entities_queries[query_id])
                sim_scores = np.zeros(len(current_candidate_ids))
                if query_entities:
                    len_query_entities = len(query_entities)
                    for idx, target_id in enumerate(current_candidate_ids):
                        target_entities = set(entities_candidate_targets[target_id])
                        len_target_entities = len(target_entities)
                        len_intersection = len(query_entities.intersection(target_entities))
                        ratio = (100/(len_query_entities+len_target_entities))*len_intersection
                        sim_scores[idx] = ratio
                current_candidate_sim_scores = sim_scores
                all_sim_scores[query_id].append(current_candidate_sim_scores)
                sim_scores_to_store[query_id] = current_candidate_sim_scores
            if args.ranking_only:
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
        sim_scores_to_store = {}
        if os.path.exists(stored_sim_scores + ".pickle" + ".zip"):
            sim_scores_to_store = load_pickled_object(decompress_file(stored_sim_scores+".pickle"+".zip"))
            print("loaded sim scores")
            for query_id in list(queries.keys()):
                current_candidate_ids = candidates[query_id]
                current_sim_scores = sim_scores_to_store[query_id]
                sim_scores_dict = dict(zip(original_target_ids, current_sim_scores))
                n_targets = len(current_candidate_ids)
                c_list = [sim_scores_dict[target_id] for target_id in current_candidate_ids]
                current_candidate_sim_scores = np.array(c_list).reshape(n_targets,)
                all_sim_scores[query_id].append(current_candidate_sim_scores)
        else:
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
                entities_candidate_targets = {k: entities_targets[k] for k in candidate_target_ids}
            else:
                print('compute targets')
                entities_candidate_targets = get_lexical_entities(candidate_targets, lex_feature)
                if args.ranking_only:
                    pickle_object(stored_entities_targets, entities_candidate_targets)
                    compress_file(stored_entities_targets + ".pickle")
                    os.remove(stored_entities_targets + ".pickle")
            for query_id in list(queries.keys()):
                current_candidate_ids = candidates[query_id]
                query_entities = set(entities_queries[query_id])
                sim_scores = np.zeros(len(current_candidate_ids))
                if query_entities:
                    len_query_entities = len(query_entities)
                    for idx, target_id in enumerate(current_candidate_ids):
                        target_entities = set(entities_candidate_targets[target_id])
                        len_target_entities = len(target_entities)
                        len_intersection = len(query_entities.intersection(target_entities))
                        len_union = len_query_entities+len_target_entities
                        if lex_feature == "similar_words_ratio":
                            ratio = (100/(len_union/2))*len_intersection
                        elif lex_feature == "similar_words_ratio_length":
                            ratio = (100/len_union)*len_intersection
                        sim_scores[idx] = ratio
                current_candidate_sim_scores = sim_scores
                all_sim_scores[query_id].append(current_candidate_sim_scores)
                sim_scores_to_store[query_id] = current_candidate_sim_scores
            if args.ranking_only:
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
                current_candidate_ids = candidates[query_id]
                current_sim_scores = sim_scores_to_store[query_id]
                sim_scores_dict = dict(zip(original_target_ids, current_sim_scores))
                n_targets = len(current_candidate_ids)
                c_list = [sim_scores_dict[target_id] for target_id in current_candidate_ids]
                current_candidate_sim_scores = np.array(c_list).reshape(n_targets,)
                all_sim_scores[query_id].append(current_candidate_sim_scores)
        else:
            for query_id in list(queries.keys()):
                current_candidate_ids = candidates[query_id]
                sim_scores = np.zeros(len(current_candidate_ids))
                for idx, target_id in enumerate(current_candidate_ids):
                    query = queries[query_id]
                    target = candidate_targets[target_id]
                    sim_scores[idx] = get_string_similarity(query, target, string_feature)
                current_candidate_sim_scores = sim_scores
                all_sim_scores[query_id].append(current_candidate_sim_scores)
                sim_scores_to_store[query_id] = current_candidate_sim_scores
            if args.ranking_only:
                pickle_object(stored_sim_scores, sim_scores_to_store)
                compress_file(stored_sim_scores + ".pickle")
                os.remove(stored_sim_scores + ".pickle")
    """
       5. For all referential similarity measures\
       5.1 get entities for all queries and cache or load from cache\
       5.2. get entities for all targets and cache or load from cache\
       5.3. Calculate all similarity scores for one query and its *candidate targets* or load from cache -> value between 0 and 100 and cache
     """
    for discrete_feature in args.discrete_similarity_measures:
        print(discrete_feature)
        all_features.append(discrete_feature)
        stored_entities_queries = caching_directory + "/queries_" + str(discrete_feature)
        stored_entities_targets = caching_directory_targets + "/targets_" + str(discrete_feature)
        stored_sim_scores = caching_directory + "/sim_scores_" + discrete_feature + "_discrete"
        sim_scores_to_store = {}
        if os.path.exists(stored_sim_scores + ".pickle" + ".zip"):
            sim_scores_to_store = load_pickled_object(decompress_file(stored_sim_scores + ".pickle" + ".zip"))
            print("loaded sim scores")
            for query_id in list(queries.keys()):
                current_candidate_ids = candidates[query_id]
                current_sim_scores = sim_scores_to_store[query_id]
                sim_scores_dict = dict(zip(original_target_ids, current_sim_scores))
                n_targets = len(current_candidate_ids)
                c_list = [sim_scores_dict[target_id] for target_id in current_candidate_ids]
                current_candidate_sim_scores = np.array(c_list).reshape(n_targets, )
                all_sim_scores[query_id].append(current_candidate_sim_scores)
        else:
            if os.path.exists(stored_entities_queries + ".pickle" + ".zip"):
                entities_queries = load_pickled_object(decompress_file(stored_entities_queries + ".pickle" + ".zip"))
                print('queries loaded')
            else:
                print('compute queries')
                if discrete_feature == "similar_words_ratio" or lex_feature == "similar_words_ratio_length":
                    entities_queries = get_lexical_entities(queries, lex_feature)
                else:
                    entities_queries = get_sequence_entities(queries, discrete_feature)
                pickle_object(stored_entities_queries, entities_queries)
                compress_file(stored_entities_queries + ".pickle")
                os.remove(stored_entities_queries + ".pickle")
            if os.path.exists(stored_entities_targets + ".pickle" + ".zip"):
                entities_targets = load_pickled_object(decompress_file(stored_entities_targets + ".pickle" + ".zip"))
                print('targets loaded')
                entities_candidate_targets = {k: entities_targets[k] for k in candidate_target_ids}
            else:
                print('compute targets')
                if discrete_feature == "similar_words_ratio" or lex_feature == "similar_words_ratio_length":
                    entities_candidate_targets = get_lexical_entities(candidate_targets, discrete_feature)
                else:
                    entities_candidate_targets = get_sequence_entities(candidate_targets, discrete_feature)
                if args.ranking_only:
                    pickle_object(stored_entities_targets, entities_candidate_targets)
                    compress_file(stored_entities_targets + ".pickle")
                    os.remove(stored_entities_targets + ".pickle")
            for query_id in list(queries.keys()):
                current_candidate_ids = candidates[query_id]
                query_entities = set(entities_queries[query_id])
                sim_scores = np.zeros(len(list(candidate_targets.keys())))
                if query_entities:
                    for idx, target_id in enumerate(current_candidate_ids):
                        target_entities = set(entities_candidate_targets[target_id])
                        len_intersection = len(query_entities.intersection(target_entities))
                        sim_scores[idx] = len_intersection
                current_candidate_sim_scores = sim_scores
                all_sim_scores[query_id].append(current_candidate_sim_scores)
                sim_scores_to_store[query_id] = current_candidate_sim_scores
            if args.ranking_only:
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
        sim_scores = all_sim_scores_df[feature].to_numpy()
        sim_scores = np.hstack(sim_scores)
        sim_scores_mean = np.mean(sim_scores, axis=0)
        print(sim_scores_mean)
    """
    6. get top k targets per query:
    6.1. either supervised using the model trained above
    6.2. or compute mean of features and compute top k  
    """
    output = {}
    if args.supervised:
        test_df = create_test_set(all_sim_scores, candidates, all_features, args.data)
        X_test = test_df.iloc[:,2:]
        #X_test = sel.transform(X_test)
        X_test = sel.transform(X_test)
        X_test = scaler.transform(X_test)
        y_test = classifier.predict_proba(X_test)*100
        test_df['label'] = y_test[:, 1]
        test_df_path = os.path.join(DATA_PATH, args.data, "pred_test.tsv")
        test_df.to_csv(test_df_path, index=False, header=False, sep='\t')
        supervised_output_path = os.path.join(DATA_PATH, args.data, "pred_qrels_supervised.tsv")
        supervised_output_to_pred_qrels(test_df, queries, args.k, supervised_output_path)
    else:
        if args.union:
            for query_id, query_sim_scores in list(all_sim_scores.items()):
                this_query = {}
                for feature_sim_scores in query_sim_scores:
                    targets_and_sim_scores = dict(zip(candidates[query_id], feature_sim_scores))
                    targets_and_sim_scores = dict(
                        sorted(targets_and_sim_scores.items(), key=lambda item: item[1], reverse=True))
                    targets_and_sim_scores = {x: targets_and_sim_scores[x] for x in list(targets_and_sim_scores)[:args.k]}
                    this_query.update(targets_and_sim_scores)
                output[query_id] = this_query
                output_dict_to_pred_qrels(output, output_path)

        else:
            for query_id, query_sim_scores in list(all_sim_scores.items()):
                mean_sim_scores = np.mean(query_sim_scores, axis=0)
                targets_and_sim_scores = dict(zip(candidates[query_id], mean_sim_scores))
                targets_and_sim_scores = dict(
                    sorted(targets_and_sim_scores.items(), key=lambda item: item[1], reverse=True))
                targets_and_sim_scores = {x: targets_and_sim_scores[x] for x in list(targets_and_sim_scores)[:args.k]}
                output[query_id] = targets_and_sim_scores
                output_dict_to_pred_qrels(output, output_path)


if __name__ == "__main__":
    run()
