from typing import Dict, Any, Tuple
import os
import json
from .multihop_question import MultihopQuestion


class HoptopQA(object):
  def __init__(self, root_dir):
    print('loading hotpotqa ...')
    for split, file in [('dev', 'hotpot_dev_distractor_v1.json'), ('train', 'hotpot_train_v1.1.json')]:
      file = os.path.join(root_dir, file)
      if not os.path.exists(file):
        continue
      setattr(self, split, self.load_split(file))


  def load_split(self, filename: str) -> Dict[str, Dict]:
    data = {}
    with open(filename, 'r') as fin:
      fin = json.load(fin)
      for entry in fin:
        data[entry['_id']] = {
          'question': entry['question'],
          'answer': entry['answer'],
          'supporting_facts': entry['supporting_facts'],
          'context': {c: p for c, p in entry['context']}  # list to dict
        }
    return data


  def decompose(self, id: str, split: str, break_entry: Dict[str, Any]) -> MultihopQuestion:
    type = break_entry['operators']
    if type == 'select-project':
      if break_entry['decomposition'][1].startswith('return the name of'):
        return None
      entry = getattr(self, split)[id]
      if len(entry['supporting_facts']) != 2:
        return None
      first_entity, first_ind = entry['supporting_facts'][0]
      second_entity, second_ind = entry['supporting_facts'][1]
      answer = entry['answer']
      if len({first_entity.rsplit('(', 1)[0].strip().lower(),
              second_entity.rsplit('(', 1)[0].strip().lower(),
              answer.strip().lower()}) < 3:
        return None
      first, second = break_entry['decomposition']
      first_q, first_c, first_ans = first, (first_entity, entry['context'][first_entity][first_ind]), second_entity
      second_q, second_c, second_ans = second.replace('#1', second_entity), (second_entity, entry['context'][second_entity][second_ind]), entry['answer']
      mh_q, mh_c, mh_ans = break_entry['question_text'], [first_c, second_c], answer
      return MultihopQuestion(
        single_hops=[{'q': first_q, 'c': first_c, 'a': first_ans}, {'q': second_q, 'c': second_c, 'a': second_ans}],
        multi_hop={'q': mh_q, 'c': mh_c, 'a': mh_ans})
    else:
      raise NotImplementedError


  def __getitem__(self, item: str):
    for split in ['train', 'dev']:
      if item in getattr(self, split):
        return getattr(self, split)[item]
    raise KeyError(item)
