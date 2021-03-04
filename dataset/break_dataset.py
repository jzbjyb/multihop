from typing import Dict, List, Iterable, Set
import os
import csv
import json
import random
from collections import defaultdict
from .hotpotqa import HoptopQA


class Break(object):
  BUGS = {'DROP_dev_history_2086_ae4f0fc9-a3a6-4f96-9329-e25d16f0b15c'}

  def __init__(self, root_dir: str):
    print('loading Break ...')
    self.ops2count = defaultdict(lambda: defaultdict(lambda: 0))
    self.max_hop = 0
    self.train = self.load_split(os.path.join(root_dir, 'train.csv'))
    self.dev = self.load_split(os.path.join(root_dir, 'dev.csv'))


  def load_split(self, filename) -> Dict[str, Dict]:
    data = {}
    with open(filename, 'r') as bfin:
      bfin_csv = csv.DictReader(bfin)
      for row in bfin_csv:
        id = row['question_id']
        if id in self.BUGS:  # skip bugs
          continue
        decomp = list(map(lambda x: x.strip(), row['decomposition'].split(';')))
        ops = '-'.join(eval(row['operators']))
        domain = row['question_id'].split('_', 1)[0].lower()
        self.ops2count[domain][ops] += 1
        entry = {
          'question_id': id,
          'question_text': row['question_text'],
          'decomposition': decomp,
          'decomposition_instantiated': [decomp[0]] + [None] * (len(decomp) - 1),
          'decomposition_prediction': [None] * len(decomp),
          'operators': ops,
          'prediction': None,
        }
        data[id] = entry
        self.max_hop = max(self.max_hop, len(decomp))
    return data


  def parse_id(self, id):
    origin, split, id = id.split('_', 2)
    return origin, split, id


  def get_origin_from_id(self, id):
    return self.parse_id(id)[0]


  def downsample(self, split: str, num: int):
    data = getattr(self, split)
    keys = list(data.keys())
    random.shuffle(keys)
    new_data = {}
    for k in keys[:num]:
      new_data[k] = data[k]
    setattr(self, split, new_data)


  def get_hop_n(self, hop: int, split: str, origins: Set[str]=None):
    id2q = {}
    for id, entry in getattr(self, split).items():
      origin, _split, _id = self.parse_id(id)
      if origins and origin not in origins:
        continue
      if hop == -1:  # multihop
        id2q[id] = entry['question_text']
      else:
        if len(entry['decomposition_instantiated']) > hop and entry['decomposition_instantiated'][hop] is not None:
          q = entry['decomposition_instantiated'][hop]
          assert '#' not in q.replace('# ', ''), '{} IN HOP {} with id {}'.format(q, hop, id)
          id2q[id] = q
    return id2q


  def instantiate_hop_n(self, id2ans: Dict[str, str], hop: int, split: str):
    data = getattr(self, split)
    for id, ans in id2ans.items():
      if hop == -1:  # multihop
        data[id]['prediction'] = ans
      else:
        data[id]['decomposition_prediction'][hop] = ans
        for i in range(hop + 1, len(data[id]['decomposition'])):
          if data[id]['decomposition_instantiated'][i] is None:
            data[id]['decomposition_instantiated'][i] = data[id]['decomposition'][i]
          data[id]['decomposition_instantiated'][i] = \
            data[id]['decomposition_instantiated'][i].replace('#{}'.format(hop + 1), ans)


  def save(self, split: str, output: str):
    with open(output, 'w') as fout:
      json.dump(getattr(self, split), fout, indent=2)


  def get_hotpotqa(self, hotpotqa: HoptopQA, split: str) -> Iterable[Dict]:
    for id, entry in getattr(self, split).items():
      origin, _split, _id = self.parse_id(id)
      if origin != 'HOTPOT' or _split != split:
        continue
      if entry['operators'] != 'select-project':
        continue
      de = hotpotqa.decompose(id, split, entry)
      if de is None:
        continue
      yield de
