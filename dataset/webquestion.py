from typing import Dict, Any, List, Tuple
import os
import json
import numpy as np
from collections import defaultdict
from .multihop_question import MultihopQuestion


class CWQSnippet(object):
  def __init__(self, root_dir: str):
    print('loading snippets ...')
    for split, file in [('train', '1_1/web_snippets_train.json'), ('dev', '1_1/web_snippets_dev.json')]:
      file = os.path.join(root_dir, file)
      if not os.path.exists(file):
        continue
      setattr(self, split, self.load_split(file))


  @staticmethod
  def merge(list1, list2):
    result = []
    merge_inds = []
    min_len = min(len(list1), len(list2))
    for i in range(min_len):
      if np.random.rand() > 0.5:
        result.append(list1[i])
        result.append(list2[i])
        merge_inds.extend([1, 2])
      else:
        result.append(list2[i])
        result.append(list1[i])
        merge_inds.extend([2, 1])
    result = result + list1[min_len:] + list2[min_len:]
    assert len(result) == len(list1) + len(list2)
    return result


  def load_split(self, file):
    id2split2source2qs: Dict[str, Dict[str, Dict[str, Tuple[str, List]]]] = defaultdict(lambda: defaultdict(lambda: {}))
    with open(file, 'r') as fin:
      data = json.load(fin)
      for q in data:
        qid = q['question_ID']
        sources = q['split_source']
        split = q['split_type']
        snippets = q['web_snippets']
        question = q['web_query']
        for s in sources:
          id2split2source2qs[qid][split][s] = (question, snippets)
    return id2split2source2qs


  def get_context(self, split, qid, max_num_words_per_split: int=200, shuffle: bool=True) -> Tuple[str, str]:
    source_li = ['ptrnet split', 'noisy supervision split']
    id2split2source2qs = getattr(self, split)
    three_splits = []
    num_splits = []
    for split in ['split_part1', 'split_part2', 'full_question']:
      mnwps = max_num_words_per_split
      if split == 'full_question':
        mnwps = max_num_words_per_split * 2
      titles = []
      sps = []
      wc = 0
      for source in source_li:
        if source not in id2split2source2qs[qid][split]:
          continue
        question, snippets = id2split2source2qs[qid][split][source]
        for snippet in snippets:
          sp = snippet['snippet']
          title = snippet['title']
          cc = len(sp.split()) + len(title.split())
          if wc + cc > mnwps:
            break
          titles.append(title)
          sps.append(sp)
          wc += cc
        if len(titles) > 0:
          break
      num_splits.append(len(titles) > 0)
      three_splits.append((titles, sps))
    is_split = True
    is_empty = False
    if num_splits[0] and num_splits[1]:
      if shuffle:
        titles = self.merge(three_splits[0][0], three_splits[1][0])
        sps = self.merge(three_splits[0][1], three_splits[1][1])
      else:
        titles = three_splits[0][0] + three_splits[1][0]
        sps = three_splits[0][1] + three_splits[1][1]
    elif num_splits[2]:
      is_split = False
      titles = three_splits[2][0]
      sps = three_splits[2][1]
    else:
      is_empty = True
      titles = []
      sps = []
    return ' '.join(titles), ' '.join(sps), is_split, is_empty, len(titles)


class WebQuestion(object):
  def __init__(self, root_dir: str):
    print('loading WebQuestion ...')
    self.numparses2count = defaultdict(lambda: 0)
    self.dup_count = 0
    for split, file in [('train', 'WebQSP/data/WebQSP.train.json'), ('test', 'WebQSP/data/WebQSP.test.json')]:
      file = os.path.join(root_dir, file)
      if not os.path.exists(file):
        continue
      setattr(self, split, self.load_split(file))
    print('dup count {}'.format(self.dup_count))

  def load_split(self, filename: str, dedup_ans: bool=True) -> Dict[str, Any]:
    with open(filename, 'r') as fin:
      result = {}
      data = json.load(fin)['Questions']
      for ex in data:
        self.numparses2count[len(ex['Parses'])] += 1
        answers = [a['EntityName'] or a['AnswerArgument'] for a in ex['Parses'][0]['Answers']]
        if dedup_ans:
          new_answers = []
          for a in answers:
            if a not in new_answers:
              new_answers.append(a)
          self.dup_count += int(len(new_answers) < len(answers))
          answers = new_answers
        result[ex['QuestionId']] = {
          'id': ex['QuestionId'],
          'question': ex['ProcessedQuestion'],
          'answers': answers
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
    self.dup_count = 0
    for split, file in [('train', '1_1/ComplexWebQuestions_train.json'),
                        ('dev', '1_1/ComplexWebQuestions_dev.json')]:
      file = os.path.join(root_dir, file)
      if not os.path.exists(file):
        continue
      setattr(self, split, self.load_split(file))
    print('dup count {}'.format(self.dup_count))


  def load_split(self, filename: str, dedup_ans: bool=True) -> Dict[str, Any]:
    with open(filename, 'r') as fin:
      result = {}
      data = json.load(fin)
      for ex in data:
        dedup_answers = []
        answer_ids = set()
        answer_sets = set()
        for anss in ex['answers']:
          if dedup_ans and anss['answer_id'] in answer_ids:
            continue
          else:
            answer_ids.add(anss['answer_id'])
          ans: str = anss['answer'] or anss['answer_id']
          alias: List[str] = anss['aliases']
          all_ans: List[str] = [ans] + list(set(alias) - {ans})
          key = all_ans[0]
          if dedup_ans and key in answer_sets:
            continue
          else:
            answer_sets.add(key)
          dedup_answers.append(all_ans)
        self.dup_count += int(len(dedup_answers) < len(ex['answers']))
        result[ex['ID']] = {
          'id': ex['ID'],
          'type': ex['compositionality_type'],
          'question': ex['question'],
          'machine_question': ex['machine_question'],
          'answers': dedup_answers,
          'composition_answer': ex['composition_answer'],
          'webqsp_ID': ex['webqsp_ID'],
        }
      return result


  def decompose_composition(self, cwq: Dict, wq: Dict, use_ph: bool=False) -> MultihopQuestion:
    sec_q = wq['question']
    sec_a = wq['answers']
    multi_q = cwq['question']
    multi_a = cwq['answers']
    sec_a_f = MultihopQuestion.format_multi_answers_with_alias(sec_a, only_first_alias=True)
    multi_a_f = MultihopQuestion.format_multi_answers_with_alias(multi_a, only_first_alias=True)
    assert sec_a_f == multi_a_f, '{} answer inconsistent:\n{}\n{}'.format(cwq['id'], sec_a_f, multi_a_f)
    sec_a = multi_a  # use alias
    for i in range(len(sec_q)):
      if sec_q[i] != cwq['machine_question'][i]:
        break
    start = i
    for i in range(len(sec_q)):
      if sec_q[-i - 1] != cwq['machine_question'][-i - 1]:
        break
    end = len(cwq['machine_question']) - i
    sec_q_ph = cwq['machine_question'][:start].strip() + ' #1 ' + cwq['machine_question'][end:].strip()
    if use_ph:
      sec_q = sec_q_ph
    fir_q = 'return ' + cwq['machine_question'][start:end].strip()
    fir_a = [[cwq['composition_answer']]]
    return MultihopQuestion([{'q': fir_q, 'a': fir_a}, {'q': sec_q, 'a': sec_a}],
                            {'q': multi_q, 'a': multi_a}, ind=cwq['id'])


  def decompose_conjunction(self, cwq: Dict, wq: Dict, use_ph: bool=False) -> MultihopQuestion:
    fir_q = wq['question']
    fir_a = wq['answers']
    multi_q = cwq['question']
    multi_a = cwq['answers']
    assert cwq['machine_question'].startswith(wq['question']), 'no substring in conjunction'
    sec_q = cwq['machine_question'][len(wq['question']):].strip()
    sec_q = sec_q.lstrip('and').strip()
    #for rem in ['is', 'was', 'are', 'were']:
    #  sec_q = sec_q.lstrip(rem).strip()
    #sec_q = 'return ' + sec_q + ' from ' + ', '.join(fir_a)
    sec_q_ph = 'Which one of the following {}: #1?'.format(sec_q)
    sec_q = 'Which one of the following {}: {}?'.format(sec_q, MultihopQuestion.format_multi_answers_with_alias(fir_a, only_first_alias=True, ans_sep=', '))
    if use_ph:
      sec_q = sec_q_ph
    sec_q = sec_q.replace(' from after ', ' after ').replace(' from before ', ' before ')
    sec_a = cwq['answers']
    return MultihopQuestion([{'q': fir_q, 'a': fir_a}, {'q': sec_q, 'a': sec_a}],
                            {'q': multi_q, 'a': multi_a}, ind=cwq['id'])


  def decompose_superlative(self, cwq: Dict, wq: Dict, use_ph: bool=False) -> MultihopQuestion:
    return self.decompose_conjunction(cwq, wq, use_ph=use_ph)


  def decompose_comparative(self, cwq: Dict, wq: Dict, use_ph: bool=False) -> MultihopQuestion:
    return self.decompose_conjunction(cwq, wq, use_ph=use_ph)


  def decompose(self, split: str, use_ph: bool=False):
    if self.webq is None:
      raise Exception('load webquestion first')
    for k, v in getattr(self, split).items():
      yield getattr(self, 'decompose_{}'.format(v['type']))(v, self.webq[v['webqsp_ID']], use_ph=use_ph)


  def __getitem__(self, item: str):
    for split in ['train', 'dev']:
      if item in getattr(self, split):
        return getattr(self, split)[item]
    raise KeyError(item)
