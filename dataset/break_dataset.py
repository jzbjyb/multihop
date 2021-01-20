from typing import Dict, List, Iterable
import os
import csv
from collections import defaultdict
from .hotpotqa import HoptopQA


class Break(object):
  def __init__(self, root_dir: str):
    print('loading Break ...')
    self.ops2count = defaultdict(lambda: 0)
    self.train = self.load_split(os.path.join(root_dir, 'train.csv'))
    self.dev = self.load_split(os.path.join(root_dir, 'dev.csv'))


  def load_split(self, filename) -> List[Dict]:
    data = []
    with open(filename, 'r') as bfin:
      bfin_csv = csv.DictReader(bfin)
      for row in bfin_csv:
        decomp = list(map(lambda x: x.strip(), row['decomposition'].split(';')))
        ops = '-'.join(eval(row['operators']))
        self.ops2count[ops] += 1
        entry = {
          'question_id': row['question_id'],
          'question_text': row['question_text'],
          'decomposition': decomp,
          'operators': ops,
        }
        data.append(entry)
    return data


  def get_hotpotqa(self, hotpotqa: HoptopQA, split: str) -> Iterable[Dict]:
    for entry in getattr(self, split):
      origin, _split, id = entry['question_id'].split('_', 2)
      if origin != 'HOTPOT' or _split != split:
        continue
      if entry['operators'] != 'select-project':
        continue
      de = hotpotqa.decompose(id, split, entry)
      if de is None:
        continue
      yield de
