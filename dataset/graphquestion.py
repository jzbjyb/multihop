from typing import Dict, Any
import os
import json
from collections import defaultdict


class GraphQuestion(object):
  def __init__(self, root_dir: str):
    print('loading GraphQuestion ...')
    self.numans2count = defaultdict(lambda: 0)
    for split, file in [('train', 'GraphQuestions/freebase13/graphquestions.training.json'),
                        ('test', 'GraphQuestions/freebase13/graphquestions.testing.json')]:
      file = os.path.join(root_dir, file)
      if not os.path.exists(file):
        continue
      setattr(self, split, self.load_split(file))


  def load_split(self, filename: str) -> Dict[str, Any]:
    with open(filename, 'r') as fin:
      result = {}
      data = json.load(fin)
      for ex in data:
        id = str(ex['qid'])
        result[id] = {
          'id': id,
          'question': ex['question'],
          'answers': ex['answer']
        }
        self.numans2count[len(ex['answer'])] += 1
      return result


  def __getitem__(self, item: str):
    item = str(item)
    for split in ['train', 'test']:
      if item in getattr(self, split):
        return getattr(self, split)[item]
    raise KeyError(item)
