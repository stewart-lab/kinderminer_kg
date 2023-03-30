import gzip
import glob
import os
import sys

from os.path import dirname, join, abspath
parent_path = abspath(join(dirname(__file__), '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

import util.km_util as km_util

def main():
    n_skips = 0
    n_total_rels = 0
    header = ''

    # clean up and alphebetize individual files
    rels_gzip_path = os.path.join('4_build_graph', 'rels')
    wd = os.getcwd()
    gzip_files = glob.glob(os.path.join(rels_gzip_path, "*.tsv.gzip*"))
    
    for gzip_file in gzip_files:
        header = ''
        new_file_name =  os.path.splitext(gzip_file)[0]
        lines_set = set()

        with gzip.open(gzip_file, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                n_total_rels += 1

                if i > 0:
                    spl = line.split('\t')
                    entity_name_1 = str.join(' ', km_util.get_tokens(spl[0]))
                    entity_name_2 = str.join(' ', km_util.get_tokens(spl[3]))

                    # remove any entities that are just numbers
                    if entity_name_1.replace(' ', '').isnumeric() or entity_name_2.replace(' ', '').isnumeric():
                        n_skips += 1
                        continue

                    # entities must be at least 2 characters long
                    if len(entity_name_1) < 2 or len(entity_name_2) < 2:
                        n_skips += 1
                        continue

                    # remove entities that are subset terms of the other term
                    entity_1_tokens = km_util.get_tokens(entity_name_1)
                    entity_2_tokens = km_util.get_tokens(entity_name_2)

                    skip = False
                    for i in range(0, len(entity_1_tokens)):
                        for j in range(i, len(entity_1_tokens)):
                            subset_name = str.join(' ', entity_1_tokens[i:j + 1])
                            if subset_name == entity_name_2:
                                skip = True
                    for i in range(0, len(entity_2_tokens)):
                        for j in range(i, len(entity_2_tokens)):
                            subset_name = str.join(' ', entity_2_tokens[i:j + 1])
                            if subset_name == entity_name_1:
                                skip = True

                    if skip:
                        n_skips += 1
                        continue

                    entity_1_type = spl[1]
                    rel_type = spl[2]
                    entity_2_type = spl[4]

                    if rel_type == 'COREF' and entity_1_type != entity_2_type:
                        n_skips += 1
                        continue

                    spl[0] = entity_name_1
                    spl[3] = entity_name_2
                    lines_set.add(str.join('\t', spl))
                else:
                    header = line.upper()

        with open(new_file_name, 'w') as tsv:
            tsv.write(header)
            ordered_lines = [x for x in lines_set]
            ordered_lines.sort()
            for line in ordered_lines:
                tsv.write(line)

    # aggregate all small files into one large file
    tsv_files = glob.glob(os.path.join(rels_gzip_path, "*.tsv"))
    tsv_files = [x for x in tsv_files if 'aggregated' not in x]

    rel_dict = dict()
    for tsv_file in tsv_files:
        with open(tsv_file, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                spl = line.split('\t')

                rel = str.join('\t', spl[:len(spl) - 1])
                pmid = int(spl[len(spl) - 1])

                if rel not in rel_dict:
                    rel_dict[rel] = set()

                rel_dict[rel].add(pmid)

    aggregated_file = os.path.join(rels_gzip_path, 'aggregated.tsv')
    with open(aggregated_file, 'w') as tsv:
        tsv.write(header)

        for key in rel_dict:
            val = rel_dict[key]
            tsv.write(key + '\t' + str(val) + '\n')

    print('percent skipped: ' + str(n_skips / n_total_rels))

def get_from_aggregated(aggregated, query, literal = False):
    aggregated_dict = dict()

    with open(aggregated, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue

            # spl = line.strip().split('\t')
            # key = str.join('\t', spl[:len(spl) - 1])

            # last_col = spl[len(spl) - 1].strip('}').strip('{')
            # last_col = [int(x.strip()) for x in last_col.split(',')]

            # aggregated_dict[key] = last_col
            found = True

            for item in query:
                if item not in line:
                    found = False
                    break
                
            if found:
                spl = line.strip().split('\t')

                if literal and query[0] not in spl:
                    continue

                last_col = spl[len(spl) - 1].strip('}').strip('{')
                last_col = [x.strip() for x in last_col.split(',')]

                if len(last_col) > 5:
                    print(str.join('\t', spl[:len(spl) - 1]) + '\t' + str(len(last_col)))

def convert_to_new_format(txt_file, out_file):
    with open(out_file, 'w') as f_out:
        with open(txt_file, 'r') as f_in:
            for line in f_in:
                
                spl = line.strip().replace('/', ' ').split('\t')
                synonyms = spl[1]

                synonyms = synonyms.split('|')

                new_synonyms_set = set()
                for synonym in synonyms:
                    tokens = km_util.get_tokens(synonym)
                    sanitized_name = str.join(' ', tokens)
                    new_synonyms_set.add(sanitized_name)

                new_format = str.join('/', new_synonyms_set)
                f_out.write(new_format)
                f_out.write('\n')

if __name__ == '__main__':
    main()