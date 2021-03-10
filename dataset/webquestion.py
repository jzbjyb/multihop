from typing import Dict, Any
import os
import json
from collections import defaultdict
from .multihop_question import MultihopQuestion


class WebQuestion(object):
  def __init__(self, root_dir: str):
    print('loading WebQuestion ...')
    self.numparses2count = defaultdict(lambda: 0)
    for split, file in [('train', 'WebQSP/data/WebQSP.train.json'), ('test', 'WebQSP/data/WebQSP.test.json')]:
      file = os.path.join(root_dir, file)
      if not os.path.exists(file):
        continue
      setattr(self, split, self.load_split(file))


  def load_split(self, filename: str) -> Dict[str, Any]:
    with open(filename, 'r') as fin:
      result = {}
      data = json.load(fin)['Questions']
      for ex in data:
        self.numparses2count[len(ex['Parses'])] += 1
        result[ex['QuestionId']] = {
          'id': ex['QuestionId'],
          'question': ex['ProcessedQuestion'],
          'answers': [a['EntityName'] or a['AnswerArgument'] for a in ex['Parses'][0]['Answers']]
        }
      return result


  def __getitem__(self, item: str):
    for split in ['train', 'test']:
      if item in getattr(self, split):
        return getattr(self, split)[item]
    raise KeyError


class ComplexWebQuestion(object):
  def __init__(self, root_dir: str, webq: WebQuestion=None):
    print('loading ComplexWebQuestion ...')
    self.webq = webq
    for split, file in [('train', '1_1/ComplexWebQuestions_train.json'),
                        ('dev', '1_1/ComplexWebQuestions_dev.json')]:
      file = os.path.join(root_dir, file)
      if not os.path.exists(file):
        continue
      setattr(self, split, self.load_split(file))


  def load_split(self, filename: str) -> Dict[str, Any]:
    with open(filename, 'r') as fin:
      result = {}
      data = json.load(fin)
      for ex in data:
        result[ex['ID']] = {
          'id': ex['ID'],
          'type': ex['compositionality_type'],
          'question': ex['question'],
          'machine_question': ex['machine_question'],
          'answers': list(set(aa for a in ex['answers'] for aa in [a['answer'] or a['answer_id']] + a['aliases'])),
          'composition_answer': ex['composition_answer'],
          'webqsp_ID': ex['webqsp_ID'],
        }
      return result


  def decompose_composition(self, cwq: Dict, wq: Dict) -> MultihopQuestion:
    sec_q = wq['question']
    sec_a = wq['answers']
    multi_q = cwq['question']
    multi_a = cwq['answers']
    for i in range(len(sec_q)):
      if sec_q[i] != cwq['machine_question'][i]:
        break
    start = i
    for i in range(len(sec_q)):
      if sec_q[-i - 1] != cwq['machine_question'][-i - 1]:
        break
    end = len(cwq['machine_question']) - i
    fir_q = 'return ' + cwq['machine_question'][start:end].strip()
    fir_a = [cwq['composition_answer']]
    return MultihopQuestion([{'q': fir_q, 'a': fir_a}, {'q': sec_q, 'a': sec_a}],
                            {'q': multi_q, 'a': multi_a}, ind=cwq['id'])


  def decompose_conjunction(self, cwq: Dict, wq: Dict) -> MultihopQuestion:
    fir_q = wq['question']
    fir_a = wq['answers']
    multi_q = cwq['question']
    multi_a = cwq['answers']
    assert cwq['machine_question'].startswith(wq['question']), 'no substring in conjunction'
    sec_q = cwq['machine_question'][len(wq['question']):].strip()
    for rem in ['and', 'is', 'was', 'are', 'were']:
      sec_q = sec_q.lstrip(rem).strip()
    sec_q = 'return ' + sec_q + ' from ' + ', '.join(fir_a)
    sec_q = sec_q.replace(' from after ', ' after ').replace(' from before ', ' before ')
    sec_a = cwq['answers']
    return MultihopQuestion([{'q': fir_q, 'a': fir_a}, {'q': sec_q, 'a': sec_a}],
                            {'q': multi_q, 'a': multi_a}, ind=cwq['id'])


  def decompose_superlative(self, cwq: Dict, wq: Dict) -> MultihopQuestion:
    return self.decompose_conjunction(cwq, wq)


  def decompose_comparative(self, cwq: Dict, wq: Dict) -> MultihopQuestion:
    return self.decompose_conjunction(cwq, wq)


  def decompose(self, split: str):
    if self.webq is None:
      raise Exception('load webquestion first')
    for k, v in getattr(self, split).items():
      yield getattr(self, 'decompose_{}'.format(v['type']))(v, self.webq[v['webqsp_ID']])


  def __getitem__(self, item: str):
    for split in ['train', 'dev']:
      if item in getattr(self, split):
        return getattr(self, split)[item]
    raise KeyError
