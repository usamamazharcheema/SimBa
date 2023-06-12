import pandas as pd

from evaluation import DATA_PATH
from src.utils import get_queries, get_targets


def create_pred_file_with_text(data_name_orig, data_name, data_name_targets, score_threshold=25):

    queries_path = DATA_PATH + data_name_orig + "/queries.tsv"
    targets_path = DATA_PATH + data_name_targets + "/corpus"
    pred_path = DATA_PATH + data_name + "/pred_qrels.tsv"

    columns = ['query_id', 'target_id', 'score', 'query_text', 'target_text']

    queries = get_queries(queries_path)
    targets = get_targets(targets_path)

    pred_df = pd.read_csv(pred_path, names=['qid', 'Q0', 'docno', 'rank', 'score', 'tag'], sep='\t', index_col=False)

    output_df = pd.DataFrame(columns=columns)

    for _, row in pred_df.iterrows():
        if float(row['score']) >= score_threshold:
            new_row = [row['qid'], row['docno'], row['score'], queries[str(row['qid'])], targets[str(row['docno'])]]
            new_df = pd.DataFrame([new_row], columns=columns)
            output_df = pd.concat([output_df, new_df], names=columns)

    output_path = DATA_PATH + data_name + "/pred_with_text.tsv"
    output_df.to_csv(output_path, index=False, header=True, sep='\t')



# data_name_queries = '11235_pp'
# data_name_targets = '11235_fields_3'
# data_name = '11235_pp_queries_fields_3_no_retrieval'

# data_name_queries = '11235_pp'
# data_name_targets = '11235_fields_6'
# data_name = '11235_pp_queries_fields_6_no_retrieval'

# data_name_queries = '11235_pp'
# data_name_targets = '11235_fields_6'
# data_name = '11235_pp_fields_6_no_retrieval'
#
#
# create_pred_file_with_text(data_name_queries, data_name, data_name_targets)
#
# data_name_queries = '11658'
# data_name_targets = '11658_fields_3'
# data_name = '11658_pp_queries_fields_3_no_retrieval'


#create_pred_file_with_text(data_name_queries, data_name, data_name_targets)


# data_name_queries = '11235_pp'
# data_name_targets = '11235_fields_6'
# data_name = '11235_pp_fields_6_no_retrieval_only_semantic'
#
# create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=50)
#
# data_name_queries = '11658_pp_queries'
# data_name_targets = '11658_fields_3'
# data_name = '11658_pp_queries_fields_3_no_retrieval_only_referential'
#
# create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=3)

data_name_queries = '11155/11155_pp'
data_name_targets = 'gesis_unsup_labels'
data_name = '11155/11155_spacy_ne'
create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=3)

# data_name_queries = '11658_pp_queries'
# data_name_targets = '11658_fields_3'
# data_name = '11658_pp_queries_fields_3_no_retrieval_only_string'
#
# create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=8)

# data_name_queries = '11658_pp_queries'
# data_name_targets = '11658_fields_3'
# data_name = '11658_pp_queries_fields_3_no_retrieval_all_features'
#
# create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=14)