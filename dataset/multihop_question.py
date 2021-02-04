from typing import List, Dict, Any


class MultihopQuestion(object):
  def __init__(self, single_hops: List[Dict], multi_hop: Dict, ind: Any):
    self.single_hops = single_hops
    self.multi_hop = multi_hop
    self.ind = ind
    for sh in self.single_hops:
      sh['q'] = self.format_question(sh['q'])


  def format_question(self, question: str):
    if question.strip().lower().startswith('return'):
      return question
    return question.strip().rstrip('?') + '?'


  def __str__(self):
    return self.ind + '\n' + str(self.single_hops) + '\n' + str(self.multi_hop)
