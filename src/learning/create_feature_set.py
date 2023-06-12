import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys

import torch
from scipy.spatial.distance import cdist

base_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base_path, ".."))
import create_similarity_features

from create_similarity_features import DATA_PATH
from create_similarity_features.lexical_similarity import get_lexical_entities
from create_similarity_features.referential_similarity import get_sequence_entities
from create_similarity_features.sentence_encoder import encode_queries, encode_targets
from create_similarity_features.string_similarity import get_string_similarity

sys.path.insert(0, os.path.join(base_path, ".."))
import utils
#from utils import get_queries, get_correct_targets, decompress_file, load_pickled_object, \
#    get_number_of_tokens, pickle_object, compress_file, make_top_k_dictionary


def create_feature_set(data, targets, similarity_measure, sentence_embedding_models, referential_similarity_measures, lexical_similarity_measures, string_similarity_measures):
    """
    1. retrieve candidates for training queries
    2. calculate sim score for true pairs
    """
    caching_directory_targets = DATA_PATH + "cache/" + data
    Path(caching_directory_targets).mkdir(parents=True, exist_ok=True)
    caching_directory = DATA_PATH + "cache/training/" + data
    Path(caching_directory).mkdir(parents=True, exist_ok=True)
    queries_path = DATA_PATH + 'training/' + data + '/queries.tsv'
    queries = utils.get_queries(queries_path)
    all_features = []
    feature_names = ""
    for feature in sentence_embedding_models:
        if "/" or ":" or "." in str(feature):
            feature = str(feature).replace("/", "_").replace(":", "_").replace(".", "_").replace("-", "_")
        feature_names = feature_names + str(feature)[:10]
    for feature in referential_similarity_measures:
        feature_names = feature_names + str(feature)[:10]
    for feature in lexical_similarity_measures:
        feature_names = feature_names + str(feature)[:10]
    for feature in string_similarity_measures:
        feature_names = feature_names + str(feature)[:10]
    all_sim_scores = {}
    for query_id in list(queries.keys()):
        all_sim_scores[query_id] = []
    output_path_root = DATA_PATH + data
    Path(output_path_root).mkdir(parents=True, exist_ok=True)
    gold_path = DATA_PATH + 'training/' + data + '/gold.tsv'
    correct_targets = utils.get_correct_targets(gold_path)
    output_path = DATA_PATH + 'training/' + data + '/' + feature_names + '_feature_set'
    if os.path.exists(output_path):
        print('exists')
        feature_set_df = utils.load_pickled_object(utils.decompress_file(output_path+ ".pickle" + ".zip"))
    else:
        """
        0. Get number of tokens of
        0.1 queries
        0.2 targets
        0.3 pairs
        """
        token_lens = {}
        stored_token_lens = caching_directory + "/token_lens"
        if os.path.exists(stored_token_lens + ".pickle" + ".zip"):
            token_lens = utils.load_pickled_object(utils.decompress_file(stored_token_lens + ".pickle" + ".zip"))
        else:
            for query_id, query_text in queries.items():
                n_query_tokens = utils.get_number_of_tokens(query_text)
                this_query_token_lens = {}
                for target_id, target_text in targets.items():
                    n_target_tokens = utils.get_number_of_tokens(target_text)
                    this_query_token_lens[target_id] = n_query_tokens + n_target_tokens
                token_lens[query_id] = this_query_token_lens
            utils.pickle_object(stored_token_lens, token_lens)
            utils.compress_file(stored_token_lens + ".pickle")
            os.remove(stored_token_lens + ".pickle")
        """
        1. For all sentence embedding models:
        1.1 Embed all queries and cache
        1.2. Embed all targets and cache
        1.3. Calculate all similarity scores for all combinations -> value between 0 and 100 and cache
        """
        for model in sentence_embedding_models:
            all_features.append(model)
            if "/" or ":" or "." in str(model):
                model_name = str(model).replace("/", "_").replace(":", "_").replace(".", "_")
            else:
                model_name = str(model)
            stored_embedded_queries = caching_directory + "/embedded_queries_" + model_name
            stored_embedded_targets = caching_directory_targets + "/embedded_targets_" + model_name
            stored_sim_scores = caching_directory + "/sim_scores_" + model_name
            if os.path.exists(stored_sim_scores + ".pickle" + ".zip"):
                sim_scores_to_store = utils.load_pickled_object(utils.decompress_file(stored_sim_scores + ".pickle" + ".zip"))
                for query_id in list(queries.keys()):
                    all_sim_scores[query_id].append(sim_scores_to_store[query_id])
            else:
                sim_scores_to_store = {}
                if os.path.exists(stored_embedded_queries + ".pickle" + ".zip"):
                    embedded_queries = utils.load_pickled_object(utils.decompress_file(stored_embedded_queries + ".pickle" + ".zip"))
                else:
                    embedded_queries = encode_queries(queries, model)
                    utils.pickle_object(stored_embedded_queries, embedded_queries)
                    utils.compress_file(stored_embedded_queries + ".pickle")
                    os.remove(stored_embedded_queries + ".pickle")
                if os.path.exists(stored_embedded_targets + ".pickle" + ".zip"):
                    embedded_targets = utils.load_pickled_object(utils.decompress_file(stored_embedded_targets + ".pickle" + ".zip"))
                else:
                    embedded_targets = encode_targets(targets, model)
                    utils.pickle_object(stored_embedded_targets, embedded_targets)
                    utils.compress_file(stored_embedded_targets + ".pickle")
                    os.remove(stored_embedded_targets + ".pickle")
                for query_id in list(queries.keys()):
                    query_embedding = embedded_queries[query_id].reshape(1, -1)
                    embedded_targets_array = np.array(list(embedded_targets.values()))
                    sim_scores = (1 - cdist(query_embedding, embedded_targets_array,
                                            metric=similarity_measure)) * 100
                    n_targets = sim_scores.shape[1]
                    sim_scores = sim_scores.reshape(n_targets, )
                    all_sim_scores[query_id].append(sim_scores)
                    sim_scores_to_store[query_id] = sim_scores
                utils.pickle_object(stored_sim_scores, sim_scores_to_store)
                utils.compress_file(stored_sim_scores + ".pickle")
                os.remove(stored_sim_scores + ".pickle")
        """
        2. For all referential similarity measures\
        2.1 get entities for all queries and cache or load from cache\
        2.2. get entities for all targets and cache or load from cache\
        2.3. Calculate all similarity scores for one query and its *candidate targets* or load from cache -> value between 0 and 100 and cache
        """
        for ref_feature in referential_similarity_measures:
            all_features.append(ref_feature)
            stored_entities_queries = caching_directory + "/queries_" + str(ref_feature)
            stored_entities_targets = caching_directory_targets + "/targets_" + str(ref_feature)
            stored_sim_scores = caching_directory + "/sim_scores_" + ref_feature
            if os.path.exists(stored_sim_scores + ".pickle" + ".zip"):
                sim_scores_to_store = utils.load_pickled_object(utils.decompress_file(stored_sim_scores + ".pickle" + ".zip"))
                for query_id in list(queries.keys()):
                    all_sim_scores[query_id].append(sim_scores_to_store[query_id])
            else:
                sim_scores_to_store = {}
                if os.path.exists(stored_entities_queries + ".pickle" + ".zip"):
                    entities_queries = utils.load_pickled_object(utils.decompress_file(stored_entities_queries + ".pickle" + ".zip"))
                else:
                    entities_queries = get_sequence_entities(queries, ref_feature)
                    utils.pickle_object(stored_entities_queries, entities_queries)
                    utils.compress_file(stored_entities_queries + ".pickle")
                    os.remove(stored_entities_queries + ".pickle")
                if os.path.exists(stored_entities_targets + ".pickle" + ".zip"):
                    entities_targets = utils.load_pickled_object(utils.decompress_file(stored_entities_targets + ".pickle" + ".zip"))
                else:
                    entities_targets = get_sequence_entities(targets, ref_feature)
                    utils.pickle_object(stored_entities_targets, entities_targets)
                    utils.compress_file(stored_entities_targets + ".pickle")
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
                            ratio = (100 / (len_query_entities + len_target_entities)) * len_intersection
                            sim_scores[idx] = ratio
                    all_sim_scores[query_id].append(sim_scores)
                    sim_scores_to_store[query_id] = sim_scores
                utils.pickle_object(stored_sim_scores, sim_scores_to_store)
                utils.compress_file(stored_sim_scores + ".pickle")
                os.remove(stored_sim_scores + ".pickle")
        """
        3. For all lexical similarity measures
        3.1 get entities for all queries and cache or load from cache\
        3.2. get entities for all targets and cache or load from cache\
        3.3. Calculate all similarity scores for all combinations -> value between 0 and 100 and cache
        """
        for lex_feature in lexical_similarity_measures:
            all_features.append(lex_feature)
            stored_entities_queries = caching_directory + "/queries_" + str(lex_feature)
            stored_entities_targets = caching_directory + "/targets_" + str(lex_feature)
            stored_sim_scores = caching_directory + "/sim_scores_" + lex_feature
            if os.path.exists(stored_sim_scores + ".pickle" + ".zip"):
                sim_scores_to_store = utils.load_pickled_object(utils.decompress_file(stored_sim_scores + ".pickle" + ".zip"))
                for query_id in list(queries.keys()):
                    all_sim_scores[query_id].append(sim_scores_to_store[query_id])
            else:
                sim_scores_to_store = {}
                if os.path.exists(stored_entities_queries + ".pickle" + ".zip"):
                    entities_queries = utils.load_pickled_object(utils.decompress_file(stored_entities_queries + ".pickle" + ".zip"))
                else:
                    entities_queries = get_lexical_entities(queries, lex_feature)
                    utils.pickle_object(stored_entities_queries, entities_queries)
                    utils.compress_file(stored_entities_queries + ".pickle")
                    os.remove(stored_entities_queries + ".pickle")
                if os.path.exists(stored_entities_targets + ".pickle" + ".zip"):
                    entities_targets = utils.load_pickled_object(utils.decompress_file(stored_entities_targets + ".pickle" + ".zip"))
                else:
                    entities_targets = get_lexical_entities(targets, lex_feature)
                    utils.pickle_object(stored_entities_targets, entities_targets)
                    utils.compress_file(stored_entities_targets + ".pickle")
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
                            len_union = len_query_entities + len_target_entities
                            if lex_feature == "similar_words_ratio":
                                ratio = (100 / (len_union / 2)) * len_intersection
                            elif lex_feature == "similar_words_ratio_length":
                                ratio = (100 / len_union) * len_intersection
                            sim_scores[idx] = ratio
                    all_sim_scores[query_id].append(sim_scores)
                    sim_scores_to_store[query_id] = sim_scores
                utils.pickle_object(stored_sim_scores, sim_scores_to_store)
                utils.compress_file(stored_sim_scores + ".pickle")
                os.remove(stored_sim_scores + ".pickle")
        """
        4. For all string similarity measures
        4.1 Calculate all similarity scores for all combinations -> value between 0 and 100 and cache
        """
        for string_feature in string_similarity_measures:
            all_features.append(string_feature)
            stored_sim_scores = caching_directory + "/sim_scores_" + string_feature
            sim_scores_to_store = {}
            if os.path.exists(stored_sim_scores + ".pickle" + ".zip"):
                sim_scores_to_store = utils.load_pickled_object(utils.decompress_file(stored_sim_scores + ".pickle" + ".zip"))
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
                utils.pickle_object(stored_sim_scores, sim_scores_to_store)
                utils.compress_file(stored_sim_scores + ".pickle")
                os.remove(stored_sim_scores + ".pickle")

        """
        get top k targets per query:
        creating the union of features and compute top k
        """
        sim_scores_all_targets = all_sim_scores
        candidates = {}

        all_sim_scores_df = pd.DataFrame.from_dict(all_sim_scores, orient='index', columns=all_features)
        for query_id in list(queries.keys()):
            candidates[query_id] = []
        feature_top_ks = {}
        for feature in all_features:
            this_feature_sim_scores = np.stack(all_sim_scores_df[feature].to_numpy())
            sim_scores_top_k_values, sim_scores_top_k_idx = torch.topk(torch.tensor(this_feature_sim_scores), k=5,
                                                                   dim=1, largest=True)
            sim_scores_top_k_values = sim_scores_top_k_values.cpu().tolist()
            sim_scores_top_k_idx = sim_scores_top_k_idx.cpu().tolist()
            this_feature_top_k = utils.make_top_k_dictionary(list(queries.keys()), list(targets.keys()), sim_scores_top_k_values, sim_scores_top_k_idx)
            feature_top_ks[feature] = this_feature_top_k
        for query_id in list(queries.keys()):
            for feature in all_features:
                this_feature_top_k = feature_top_ks[feature]
                candidates[query_id].extend(this_feature_top_k[query_id].keys())
                candidates[query_id] = list(set(candidates[query_id]))

        ###
        original_target_ids = list(targets.keys())
        all_sim_scores = {}
        for query_id in list(queries.keys()):
            all_sim_scores[query_id] = []
        """
          1. all sentence embedding models 
          """
        for model in sentence_embedding_models:
            if "/" or ":" or "." in str(model):
                model_name = str(model).replace("/", "_").replace(":", "_").replace(".", "_")
            else:
                model_name = str(model)
            stored_sim_scores = caching_directory + "/sim_scores_" + model_name
            sim_scores_to_store = utils.load_pickled_object(utils.decompress_file(stored_sim_scores + ".pickle" + ".zip"))
            for query_id in list(queries.keys()):
                current_candidate_ids = candidates[query_id]
                current_sim_scores = sim_scores_to_store[query_id]
                sim_scores_dict = dict(zip(original_target_ids, current_sim_scores))
                n_targets = len(current_candidate_ids)
                c_list = [sim_scores_dict[target_id] for target_id in current_candidate_ids]
                current_candidate_sim_scores = np.array(c_list).reshape(n_targets, )
                all_sim_scores[query_id].append(current_candidate_sim_scores)
        """
        2. all referential similarity measures
        """
        for ref_feature in referential_similarity_measures:
            stored_sim_scores = caching_directory + "/sim_scores_" + ref_feature
            sim_scores_to_store = utils.load_pickled_object(utils.decompress_file(stored_sim_scores + ".pickle" + ".zip"))
            for query_id in list(queries.keys()):
                current_candidate_ids = candidates[query_id]
                current_sim_scores = sim_scores_to_store[query_id]
                sim_scores_dict = dict(zip(original_target_ids, current_sim_scores))
                n_targets = len(current_candidate_ids)
                c_list = [sim_scores_dict[target_id] for target_id in current_candidate_ids]
                current_candidate_sim_scores = np.array(c_list).reshape(n_targets, )
                all_sim_scores[query_id].append(current_candidate_sim_scores)
        """
        3. all lexical similarity measures
        """
        for lex_feature in lexical_similarity_measures:
            stored_sim_scores = caching_directory + "/sim_scores_" + lex_feature
            sim_scores_to_store = utils.load_pickled_object(utils.decompress_file(stored_sim_scores + ".pickle" + ".zip"))
            for query_id in list(queries.keys()):
                current_candidate_ids = candidates[query_id]
                current_sim_scores = sim_scores_to_store[query_id]
                sim_scores_dict = dict(zip(original_target_ids, current_sim_scores))
                n_targets = len(current_candidate_ids)
                c_list = [sim_scores_dict[target_id] for target_id in current_candidate_ids]
                current_candidate_sim_scores = np.array(c_list).reshape(n_targets, )
                all_sim_scores[query_id].append(current_candidate_sim_scores)
        """
        4. all string similarity measures
        """
        for string_feature in string_similarity_measures:
            stored_sim_scores = caching_directory + "/sim_scores_" + string_feature
            sim_scores_to_store = utils.load_pickled_object(utils.decompress_file(stored_sim_scores + ".pickle" + ".zip"))
            for query_id in list(queries.keys()):
                current_candidate_ids = candidates[query_id]
                current_sim_scores = sim_scores_to_store[query_id]
                sim_scores_dict = dict(zip(original_target_ids, current_sim_scores))
                n_targets = len(current_candidate_ids)
                c_list = [sim_scores_dict[target_id] for target_id in current_candidate_ids]
                current_candidate_sim_scores = np.array(c_list).reshape(n_targets, )
                all_sim_scores[query_id].append(current_candidate_sim_scores)

        all_sim_score_df_columns = ['query', 'target']
        all_sim_score_df_columns.extend(all_features)

        all_sim_score_df = pd.DataFrame(columns=all_sim_score_df_columns)

        for query_id, feature_sim_scores in all_sim_scores.items():
            for idx, target_id in enumerate(candidates[query_id]):
                this_row = [query_id, target_id]
                for feature in feature_sim_scores:
                    this_row.append(feature[idx])
                this_row_df = pd.DataFrame([this_row], columns=all_sim_score_df_columns)
                all_sim_score_df = pd.concat([all_sim_score_df, this_row_df])

        all_sim_score_df = all_sim_score_df.reset_index(drop=True)

        feature_set_columns = ['query', 'target']

        for feature in all_features:
            feature_set_columns.append(feature)
        feature_set_columns.append('label')
        feature_set_df = pd.DataFrame(columns=feature_set_columns)

        for index, row in all_sim_score_df.iterrows():
            query_id = row['query']
            target_id = row['target']
            correct_target = list(correct_targets[query_id])
            if index == 0:
                old_query_id = row['query']
                predicted = False
            if query_id != old_query_id:
                if not predicted:
                    for c_target in correct_target:
                        c_target_id = c_target[0]
                        try:
                            this_row_scores = [old_query_id, c_target_id]
                            for idx, _ in enumerate(all_features):
                                this_target_idx = list(targets.keys()).index(c_target_id)
                                sim_score = sim_scores_all_targets[query_id][idx][this_target_idx]
                                this_row_scores.append(sim_score)
                            this_row_scores.append('1')
                            this_row_df = pd.DataFrame([this_row_scores], columns=feature_set_columns)
                            feature_set_df = pd.concat([feature_set_df, this_row_df])
                        except:
                            print('correct target for query with id '+ str(old_query_id) + ' not available')
            old_query_id = query_id
            feature_row = [query_id, target_id]
            for feature in all_features:
                feature_row.append(row[feature])
            if target_id in list(correct_targets[query_id]):
                feature_row.append('1')
                predicted = True
                this_row_df = pd.DataFrame([feature_row], columns=feature_set_columns)
                feature_set_df = pd.concat([feature_set_df, this_row_df])
            else:
                feature_row.append('0')
                this_row_df = pd.DataFrame([feature_row], columns=feature_set_columns)
                feature_set_df = pd.concat([feature_set_df, this_row_df])
            # last one
        if not predicted:
            for c_target in correct_target:
                c_target_id = c_target[0]
                try:
                    this_row_scores = [old_query_id, c_target_id]
                    for idx, _ in enumerate(all_features):
                        this_target_idx = list(targets.keys()).index(c_target_id)
                        sim_score = sim_scores_all_targets[query_id][idx][this_target_idx]
                        this_row_scores.append(sim_score)
                    this_row_scores.append('1')
                    this_row_df = pd.DataFrame([this_row_scores], columns=feature_set_columns)
                    feature_set_df = pd.concat([feature_set_df, this_row_df])
                except:
                    print('correct target for query with id ' + str(old_query_id) + ' not available')

        utils.pickle_object(output_path, feature_set_df)
        utils.compress_file(output_path + ".pickle")
        os.remove(output_path + ".pickle")
        output_path_tsv = output_path+ '.tsv'
        feature_set_df.to_csv(output_path_tsv, index=False, header=True, sep='\t')
    return feature_set_df


def create_test_set(all_sim_scores, candidates, all_features, data):
    columns = ['query', 'target']
    columns.extend(all_features)
    test_df = pd.DataFrame(columns=columns)
    for query_id, feature_sim_scores in all_sim_scores.items():
        this_query_df = pd.DataFrame(columns=columns)
        this_query_df['query'] = [query_id]*len(candidates[query_id])
        this_query_df['target'] = candidates[query_id]
        for idx, feature in enumerate(all_features):
            this_query_df[feature] = feature_sim_scores[idx]
        test_df = pd.concat([test_df, this_query_df])

    output_path_test_set = DATA_PATH + data + "/test_df"
    utils.pickle_object(output_path_test_set, test_df)
    utils.compress_file(output_path_test_set + ".pickle")
    os.remove(output_path_test_set + ".pickle")
    output_path_tsv = DATA_PATH + data + '/test_feature_set.tsv'
    test_df.to_csv(output_path_tsv, index=False, header=True, sep='\t')

    return test_df
