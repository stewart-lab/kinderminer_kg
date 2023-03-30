import random
import requests
import time
import util.km_util as util

port = '5099'
api_url = 'http://localhost:' + port + '/km/api/jobs'
skim_api_url = 'http://localhost:' + port + '/skim/api/jobs'
the_auth = ('username', 'password')

def run_km_query(the_json: list, wait = True, url=api_url):
    # queue a new KM job for the server to perform
    response = requests.post(url, json=the_json, auth=the_auth)

    # get the job's ID
    json_post_response = response.json()
    job_id = json_post_response['id']

    # get the status of the job
    get_response = requests.get(url + "?id=" + job_id, auth=the_auth)
    json_get_response = get_response.json()
    job_status = json_get_response['status']
    #print('job status is: ' + job_status)

    if not wait:
        return get_job(json_get_response['id'])

    # wait for job to complete, sleep
    while job_status == 'queued' or job_status == 'started':
        time.sleep(1)

        # get the status of the job
        get_response = requests.get(url + "?id=" + job_id, auth=the_auth)
        json_get_response = get_response.json()
        job_status = json_get_response['status']

    # if the job's status is 'finished', print out the results
    if job_status == 'finished':
        res = json_get_response['result']
        return res

def get_job(job_id):
    get_response = requests.get(api_url + "?id=" + job_id, auth=the_auth)
    json_get_response = get_response.json()
    return json_get_response

def textfile_km_query(a_terms: str, b_terms: str, out_file: str):
    print(a_terms)
    print(b_terms)

    excluded_terms = set()
    significant_relations = []

    with open(a_terms, 'r') as f:
        a_terms = [x.strip().lower() for x in f.readlines()]
    with open(b_terms, 'r') as f:
        b_terms = [x.strip().lower() for x in f.readlines()]

    # run KM
    n_rels_desired = 100
    while len(significant_relations) < n_rels_desired:
        query = []

        # get random A-terms that have not yet been part of a significant relationship
        random_a_subset = []
        random_a_len = 1

        if not a_terms or not b_terms:
            break

        while len(random_a_subset) < random_a_len:
            random_sample = random.sample(a_terms, random_a_len)
            for item in random_sample:
                if item in excluded_terms:
                    continue
                if len(random_a_subset) >= random_a_len:
                    break
                random_a_subset.append(item)

        for i, a_term in enumerate(random_a_subset):
            # query A-terms with all B-terms that have not yet been part of a significant relationship
            for b_term in b_terms:

                if b_term in excluded_terms:
                    continue

                if a_term != b_term:
                    query.append({'a_term': a_term, 'b_term': b_term, 'return_pmids': True})

        result = run_km_query(query, True, api_url)

        for res in result:
            a_count = int(res['len(a_term_set)'])
            b_count = int(res['len(b_term_set)'])

            if a_count < 10 or a_count > 100000:
                excluded_terms.add(res['a_term'])
            if b_count < 10 or b_count > 100000:
                excluded_terms.add(res['b_term'])

        # get significant relations with at least 5 article counts
        query_sig_rel = sorted([x for x in result if float(x['pvalue']) < 0.01 and int(x['len(a_b_intersect)'] > 5)], key=lambda x: random.random())

        for res in query_sig_rel:
            a_term = res['a_term']
            b_term = res['b_term']

            if a_term in excluded_terms or b_term in excluded_terms:
                continue

            excluded_terms.add(a_term)
            excluded_terms.add(b_term)

            # TODO: use the ML models here to figure out context for the a_term and b_term? putative relationship?

            significant_relations.append(res)

        util.report_progress(len(significant_relations), n_rels_desired)

        if significant_relations:
            write_km_result(significant_relations, out_file)

def write_km_result(result, out_file):
    with open(out_file, 'w') as f:
        f.write(str.join('\t', [str(x) for x in result[0].keys()]) + '\n')

        result = sorted(result, key=lambda x: float(x['pvalue']))

        for item in result:
            vals = [str(x) for x in item.values()]
            vals = [x[:30000] for x in vals] # each cell will have max 30000 characters
            f.write(str.join('\t', vals) + '\n')