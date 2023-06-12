import argparse
from pathlib import Path
import pandas as pd

from src.pre_processing import DATA_PATH
from src.utils import get_certain_target_fields


def run():
    """
    input:
    targets
    output:
    pre-processed targets

    Keep document structure intact
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('targets', type=str, help='Input targets path as tsv file.')
    parser.add_argument('data', type=str, help='Name under which the documents should be stored.')
    parser.add_argument('-fields', type=str, nargs='+', default="all", help='Fields to keep in target file.')
    args = parser.parse_args()

    output_dir = DATA_PATH + args.data +'/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if args.fields != ['all'] and args.fields != ['analysis']:
        fields = ['id']
        fields.extend(args.fields)
        output_path = output_dir + 'corpus'
        targets_df = get_certain_target_fields(args.targets, fields)
        targets_df.to_csv(output_path, index=False, header=True, sep='\t')

    if args.fields == ['analysis']:
        targets_df = pd.read_csv(args.targets, sep='\t', dtype=str)
        wanted_fields = []
        for field in targets_df.columns:
            example_content = targets_df.iloc[1][field]
            print(example_content)
            if example_content:
                if str(example_content) not in ["[]", "nan", "['variable']", "false"]:
                    if len(example_content) > 1:
                        if str(example_content)[0] != "<":
                            if str(field) not in ["study_citation_html_en",
                                                  "study_citation_html",
                                                  "question_type1",
                                                  "selection_method_en",
                                                  "selection_method_ddi_en",
                                                  "related_research_data",
                                                  "link_count",
                                                  "variable_name_sorting",
                                                  "type",
                                                  "countries_iso",
                                                  "study_group_en",
                                                  "time_method_en",
                                                  "group_link",
                                                  "group_link_en"
                                                  "study_citation_en",
                                                  "study_lang_en",
                                                  "selection_method_ddi",
                                                  "study_group",
                                                  "analysis_unit_en",
                                                  "group_image_file",
                                                  "variable_order",
                                                  "additional_keywords",
                                                  "data_source",
                                                  "index_source",
                                                  "study_citation",
                                                  "time_method",
                                                  "selection_method",
                                                  "group_description"]:
                                if field != "id.1":
                                    # if len(example_content) < 6
                                    wanted_fields.append(field)
        print(wanted_fields)
        output_path = output_dir + 'analysis_pp_targets.tsv'
        targets_df = get_certain_target_fields(args.targets, wanted_fields)
        targets_df.to_csv(output_path, index=False, header=True, sep='\t')


if __name__ == "__main__":
    run()


