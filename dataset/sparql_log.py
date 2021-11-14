from typing import List
import argparse
from tqdm import tqdm
import json
from urllib.parse import unquote_plus
from SPARQLWrapper import SPARQLWrapper, JSON


class SparqlLog(object):
  def __init__(self, filename: str):
    self.endpoint = SPARQLWrapper(
      'https://query.wikidata.org/sparql',
      agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko)'
            ' Chrome/50.0.2661.102 Safari/537.36')
    self.queries: List[str] = self.load_log(filename)

  def load_log(self, filename):
    queries: List[str] = []
    with open(filename, 'r') as fin:
      _ = fin.readline()  # skip header
      for line in fin:
        query, timestamp, category, user_agents = line.split('\t')
        query = unquote_plus(query)
        queries.append(query)
    return queries

  def get_result(self, query: str, timeout: int = 1):
    if timeout:
      self.endpoint.setTimeout(timeout)
    self.endpoint.setQuery(query)
    self.endpoint.setReturnFormat(JSON)
    return self.endpoint.query().convert()

  def execute(self,
              output: str,
              rank: int = 0,
              world_size: int = 1):
    local_batch = self.queries[rank::world_size]
    print(f'#queires to execute {len(local_batch)}')
    suc_count = 0
    with open(output, 'w') as fout, tqdm(total=len(local_batch)) as pbar:
      for query in local_batch:
        pbar.update(1)
        try:
          result = self.get_result(query)
          num_ans = len(result['results']['bindings'])
          if 1 <= num_ans <= 100:
            suc_count += 1
            pbar.set_postfix_str(f'suc: {suc_count}')
            fout.write(json.dumps({'query': query, 'result': result}) + '\n')
        except Exception as e:
          pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', type=str, choices=['execute'], default='execute')
  parser.add_argument('--world_size', type=int, default=1)
  parser.add_argument('--rank', type=int, default=0)
  parser.add_argument('--output', type=str)
  args = parser.parse_args()
  wikidata_log = 'wikidata_sparql/2018-02-26_2018-03-25_organic.tsv'
  sparql_log = SparqlLog(wikidata_log)
  sparql_log.execute(output=args.output, rank=args.rank, world_size=args.world_size)
