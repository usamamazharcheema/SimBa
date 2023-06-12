import random
import csv

from src.get_data import DATA_PATH


def get_rows(filename):
    with open(filename, "r", encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter = "\t")
        return list(csv_reader)[1:] #ignore header

def get_samples(rows):
    sample_10k = random.sample(rows, 10000)
    sample_5k = random.sample(sample_10k, 5000)
    sample_1k = random.sample(sample_5k, 1000)
    return (sample_1k, sample_5k, sample_10k)

def write_to_file(sample, filename):
    with open(filename, "w", encoding ='utf-8') as f:
        writer = csv.writer(f, delimiter = "\t")
        writer.writerows(sample)
    print("Wrote %s" %filename)


if __name__=="__main__":
    path = DATA_PATH + "clef_2020_checkthat_2_english/corpus"
    file_suffix = "_sample.tsv"
    sample_1k, sample_5k, sample_10k = get_samples(get_rows(path))
    write_to_file(sample_1k, path+"_1k" + file_suffix)
    write_to_file(sample_5k, path+"_5k" + file_suffix)
    write_to_file(sample_10k, path+"_10k" + file_suffix)
    
    assert len(sample_1k) == 1000
    assert len(sample_5k) == 5000
    assert len(sample_10k) == 10000

    for line in sample_1k:
        assert line in sample_5k
        assert line in sample_10k
    for line in sample_5k:
        assert line in sample_10k
