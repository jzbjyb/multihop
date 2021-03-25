from typing import Dict, List, Iterable, Set, Tuple
import os
import csv
import json
import random
from collections import defaultdict
from .hotpotqa import HoptopQA
from .webquestion import ComplexWebQuestion


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


  def add_answers(self, cwq: ComplexWebQuestion=None, hotpotqa: HoptopQA=None):
    count = 0
    for split in [self.train, self.dev]:
      for id in split:
        _origin, _split, _id = self.parse_id(id)
        if cwq is not None and _origin == 'CWQ':
          count += 1
          split[id]['answers'] = cwq[_id]['answers']
        if hotpotqa is not None and _origin == 'HOTPOT':
          count += 1
          split[id]['answers'] = hotpotqa[_id]['answer']
    print('add answers for {} questions'.format(count))


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


  def get_hop_n(self, hop: int, split: str, use_prediction: bool=False, has_ret: bool=False, origins: Set[str]=None):
    if use_prediction:
      raise NotImplementedError
    if has_ret:
      raise NotImplementedError
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


  def get_hotpotqa(self, hotpotqa: HoptopQA, split: str, use_ph: bool=False) -> Iterable[Dict]:
    for id, entry in getattr(self, split).items():
      origin, _split, _id = self.parse_id(id)
      if origin != 'HOTPOT' or _split != split:
        continue
      if entry['operators'] != 'select-project':
        continue
      de = hotpotqa.decompose(_id, split, entry, use_ph=use_ph)
      if de is None:
        continue
      yield de


  def __getitem__(self, item: str):
    for split in ['train', 'dev']:
      if item in getattr(self, split):
        return getattr(self, split)[item]
    raise KeyError(item)


class PseudoBreak(object):
  def __init__(self, domains: List[str], source_target_files: List[Tuple[str, str]], num_hop: int=2, has_multihop: bool=False):
    print('loading PseudoBreak ...')
    self.domains = domains
    self.max_hop = num_hop
    self.has_multihop = has_multihop
    self.data = self.load_source_target(domains, source_target_files, num_hop=num_hop, has_multihop=has_multihop)


  def load_source_target(self, domains: List[str], source_target_files: List[Tuple[str, str]], num_hop: int=2, has_multihop: bool=False) -> Dict:
    data = defaultdict(dict)
    assert len(domains) == len(source_target_files)
    for domain, (sf, tf) in zip(domains, source_target_files):
      with open(sf, 'r') as sfin, open(tf, 'r') as tfin:
        for i, source in enumerate(sfin):
          source = source.rstrip('\n')
          target = tfin.readline().rstrip('\n')
          qid = i // (num_hop + int(has_multihop))
          key = '{}-{}'.format(domain, qid)
          data[key]['question_id'] = key
          data[key]['domain'] = domain
          if has_multihop and i % (num_hop + 1) == num_hop:  # multihop
            data[key]['question_text'] = source
            data[key]['answers'] = target
          else:  # single hop
            if 'decomposition' not in data[key]:
              data[key]['decomposition'] = []
              data[key]['decomposition_prediction'] = []
              data[key]['decomposition_instantiated'] = []
              data[key]['decomposition_answer'] = []
            data[key]['decomposition'].append(source)
            data[key]['decomposition_prediction'].append(None)
            data[key]['decomposition_instantiated'].append(source)
            data[key]['decomposition_answer'].append(target)
    return data


  def get_hop_n(self, hop: int, use_prediction: bool=False, has_ret: bool=False, **kwargs):
    id2q = {}
    for id, entry in self.data.items():
      if hop == -1:  # multihop
        id2q[id] = entry['question_text']
      else:
        if use_prediction and hop > 0:
          if len(entry['decomposition_prediction']) > hop - 1 and entry['decomposition_prediction'][hop - 1] is not None:
            q = entry['decomposition_prediction'][hop - 1]
            if has_ret:
              q = '\t'.join([q] + entry['decomposition_instantiated'][hop].split('\t')[1:])
            id2q[id] = q
        else:
          if len(entry['decomposition_instantiated']) > hop and entry['decomposition_instantiated'][hop] is not None:
            q = entry['decomposition_instantiated'][hop]
            id2q[id] = q
    return id2q


  def instantiate_hop_n(self, id2ans: Dict[str, str], hop: int, **kwargs):
    for id, ans in id2ans.items():
      if hop == -1:  # multihop
        self.data[id]['prediction'] = ans
      else:
        self.data[id]['decomposition_prediction'][hop] = ans
        if hop + 1 < len(self.data[id]['decomposition_instantiated']):
          ph = '#{}'.format(hop + 1)
          if ph in self.data[id]['decomposition_instantiated'][hop + 1]:
            self.data[id]['decomposition_instantiated'][hop + 1] = \
              self.data[id]['decomposition_instantiated'][hop + 1].replace(ph, ans.replace(' # ', ', '))


  def save(self, split: str, output: str):
    with open(output, 'w') as fout:
      json.dump(self.data, fout, indent=2)
