from typing import List, Dict, Any
import json
import truecase


class MultihopQuestion(object):
  def __init__(self, single_hops: List[Dict], multi_hop: Dict, ind: Any, **kwargs):
    self.single_hops = single_hops
    self.multi_hop = multi_hop
    self.ind = ind
    self.kwargs = kwargs
    for sh in self.single_hops:
      sh['q'] = self.format_question(sh['q'])
    self.multi_hop['q'] = self.format_question(self.multi_hop['q'])


  def format_question(self, question: str):
    if not question.strip().lower().startswith('return'):
      question = question.strip().rstrip('?') + '?'
    question = truecase.get_true_case(question)
    return question


  def __str__(self):
    out = {'single': self.single_hops, 'multi': self.multi_hop, 'id': self.ind}
    out.update(self.kwargs)
    return json.dumps(out)


  @classmethod
  def fromstr(cls, text):
    d = json.loads(text)
    single_hops = d['single']
    multi_hop = d['multi']
    ind = d['id']
    del d['single']
    del d['multi']
    del d['id']
    return cls(single_hops, multi_hop, ind, **d)
