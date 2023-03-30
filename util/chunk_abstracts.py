import json
import gzip
import os
import sys

from os.path import dirname, join, abspath
parent_path = abspath(join(dirname(__file__), '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

import util.km_util as km_util
from util.abstract_catalog import AbstractCatalog

def main():
    out_folder = '/Users/rmillikin/Desktop/abstract_chunks'

    abs_cat = AbstractCatalog('/Users/rmillikin/PubmedAbstracts')
    chunks = 600 # approx
    abstracts_per_chunk = 1000#int(23000000 / chunks)

    json_chunk = []
    chunk_n = 0
    for i, abstr in enumerate(abs_cat.stream_existing_catalog('/Users/rmillikin/PubmedAbstracts/Index/abstracts.txt.gzip')):
        txt = abstr.text.strip()
        if not txt:
            continue

        the_json = {"title": abstr.title, "text": txt, "pmid": abstr.pmid}
        json_chunk.append(the_json)

        if (i + 1) % abstracts_per_chunk == 0:
            write_chunk(os.path.join(out_folder, 'small_chunk_' + str(chunk_n) + '.jsonl.gzip'), json_chunk)
            json_chunk.clear()
            chunk_n += 1

            if chunk_n == 3:
                break

    write_chunk(os.path.join(out_folder, 'chunk_' + str(chunk_n) + '.jsonl.gzip'), json_chunk)

def write_chunk(path: str, data):
    with gzip.open(path, 'wt', encoding=km_util.encoding) as gzip_file:
        for item in data:
            gzip_file.write(json.dumps(item))
            gzip_file.write('\n')

if __name__ == '__main__':
    main()