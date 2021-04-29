from typing import List, Dict, Any, Union
import json
import truecase


class MultihopQuestion(object):
  def __init__(self, single_hops: List[Dict], multi_hop: Dict, ind: Any=None, **kwargs):
    self.single_hops = single_hops
    self.multi_hop = multi_hop
    self.ind = ind
    self.kwargs = kwargs
    for sh in self.single_hops:
      sh['q'] = self.format_question(sh['q'])
      sh['a'] = self.format_answer(sh['a'])
    self.multi_hop['q'] = self.format_question(self.multi_hop['q'])
    self.multi_hop['a'] = self.format_answer(self.multi_hop['a'])


  def format_question(self, question: str):
    if not question.strip().lower().startswith('return'):
      question = question.strip().rstrip('?') + '?'
    question = question.replace('\t', ' ')
    #question = truecase.get_true_case(question)
    return question


  def format_answer(self, answer: Union[str, List[str], List[List[str]]]):
    if type(answer) is str:
      return answer.strip()
    else:
      return [[a.strip() for a in anss] if type(anss) is list else anss.strip() for anss in answer]


  @staticmethod
  def format_multi_answers_with_alias(answers: Union[List[List[str]], List[str]],
                                      only_first_alias: bool=False,
                                      ans_sep: str='\t\t',
                                      alias_sep: str='\t'):
    if only_first_alias:
      return ans_sep.join(anss[0] if type(anss) is list else anss for anss in answers)
    else:
      return ans_sep.join(alias_sep.join(anss) if type(anss) is list else anss for anss in answers)


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
