import subprocess

#fields document 2 : id	question_name	time_collection_max_year	title	topic	kind_data	question_type2	study_citation_en	countries_collection_en	time_collection_min_year	study_id	study_title	date	notes	kind_data_en	group_link_en	variable_name	methodology_collection_en	question_lang	methodology_collection_ddi	study_id_title	countries_collection	study_id_title_en	notes_en	topic_exploredata_orig_en	question_id	time_collection_years	methodology_collection_ddi_en	topic_exploredata_orig	group_description_en	study_lang	topic_en	variable_label_en	title_en	variable_label	study_title_en	methodology_collection	date_recency	analysis_unit


# 1.1 # '-fields', 'id', 'question_id', 'variable_label', 'question_text_en', 'study_title', 'topic_exploredata_orig_en', 'study_id', 'study_title_en', 'study_id_title_en', 'variable_label_en', 'title', 'variable_name', 'title_en', 'question_text', 'topic', 'date', 'question_label', 'question_label_en', 'time_method_en', 'study_id_title'
# 1.2 # '-fields', ['id', 'id', 'question_id', 'variable_label', 'study_title', 'topic_exploredata_orig_en', 'study_id', 'study_title_en', 'study_id_title_en', 'variable_label_en', 'title', 'variable_name', 'title_en', 'topic', 'date', 'study_id_title']
#1.3 ['id', 'id', 'question_id', 'variable_label', 'study_title', 'topic_exploredata_orig_en', 'study_id', 'study_title_en', 'study_id_title_en', 'variable_label_en', 'title', 'variable_name', 'title_en', 'topic', 'date', 'study_id_title']

# 2 # '-fields', 'variable_label', 'question_text_en', 'variable_label_en', 'title', 'title_en', 'question_text', 'topic', 'question_label', 'question_label_en'
# 2.2 # ['id', 'variable_label', 'variable_label_en', 'title', 'title_en', 'topic']

# 3 # '-fields', 'variable_label', 'question_text_en', 'variable_label_en', 'title', 'title_en', 'question_text'
# 4 # '-fields', 'study_id', 'question_label_en', 'variable_label', 'question_text_en', 'variable_label_en', 'title', 'title_en', 'question_text'



def run():

    data_name_queries = '11155_pp'
    data_name_targets = 'gesis_unsup'
    new_data_name_targets = 'gesis_unsup_labels'
    data_name = '11155'

    data_name_targets = 'gesis_unsup'
    new_data_name_targets = 'gesis_unsup_text'

    subprocess.call(["python",
                     "../../src/pre_processing/pre_processing_targets.py",
                     "../../data/"+data_name_targets+"/corpus_all",
                     new_data_name_targets,
                     '-fields', 'variable_label', 'variable_label_en', 'title', 'title_en', 'question_text', 'question_text_en'])


    data_name_queries = '35529/35529_pp'
    data_name_cache = '35529_text'
    data_name_targets = 'gesis_unsup_text'
    data_name = '35529_text'


    subprocess.call(["python",
                     "../../src/candidate_retrieval/retrieval.py",
                     "../../data/"+data_name_queries+"/queries.tsv",
                     "../../data/"+data_name_targets+"/corpus",
                     data_name_cache,
                     data_name,
                     "braycurtis",
                     "10",
                     "--union_of_top_k_per_feature",
                     "--gesis_unsup",
                     '-sentence_embedding_models', "sentence-transformers/sentence-t5-base", "all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large",
                     '-referential_similarity_measures', "synonym_similarity", "ne_similarity",
                     '-lexical_similarity_measures', "similar_words_ratio", "similar_words_ratio_length",
                     '-string_similarity_measures', "sequence_matching", "levenshtein"])
    #
    # # data_name_queries = '11155/11155_pp'
    # # data_name_targets = 'gesis_unsup_labels'
    # # data_name_cache = '11155'
    # # data_name = '11155/11155_spacy_ne'
    #
    subprocess.call(["python",
                     "../../src/re_ranking/re_ranking.py",
                     "../../data/"+data_name_queries+"/queries.tsv",
                     "../../data/"+data_name_targets+"/corpus",
                     data_name_cache,
                     data_name,
                     "braycurtis",
                     "10",
                     "--gesis_unsup",
                     "--ranking_only",
                     "--union",
                     #'-sentence_embedding_models', "sentence-transformers/sentence-t5-base", "all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large",#, "https://tfhub.dev/google/universal-sentence-encoder/4"])
                     #'-referential_similarity_measures', "spacy_ne_similarity"
                     #'-lexical_similarity_measures', "similar_words_ratio", "similar_words_ratio_length",
                     #'-string_similarity_measures', "sequence_matching", "levenshtein",
                     '-discrete_similarity_measures', "spacy_ne_similarity"
                     ])


if __name__ == "__main__":
    run()