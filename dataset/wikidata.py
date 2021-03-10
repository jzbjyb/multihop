from typing import Set, Dict, List, Tuple
import sling
import time
import os
import json
import random
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from .multihop_question import MultihopQuestion
from .stanford import StanfordNLP
from .POSTree import POSTree


random.seed(2021)
np.random.seed(2021)


class SlingExtractor(object):
  WH_WORDS = {'what', 'when', 'who', 'whom', 'why', 'where', 'which'}
  WH2THAT = {
    'what': 'one',
    'when': 'time',
    'who': 'person',
    'whom': 'person',
    'why': 'reason',
    'where': 'place',
    'which': 'one'
  }
  TYPE2THAT = {
    'CARDINAL': 'number',
    'DATE': 'date',
    'EVENT': 'event',
    'FAC': 'facility',
    'GPE': 'place',
    'LANGUAGE': 'language',
    'LAW': 'law',
    'LOC': 'location',
    'MONEY': 'money',
    'NORP': 'group',
    'ORDINAL': 'rank',
    'ORG': 'organization',
    'PERCENT': 'percent',
    'PERSON': 'person',
    'PRODUCT': 'product',
    'QUANTITY': 'quantity',
    'TIME': 'time',
    'WORK_OF_ART': 'work'
  }
  STAT_PH = '**blank**'

  def load_kb(self, root_dir: str='local/data/e/wiki'):
    print('loading and indexing kb ...')
    start = time.time()
    self.kb = sling.Store()
    self.kb.load(os.path.join(root_dir, 'kb.sling'))
    self.phrase = sling.PhraseTable(self.kb, os.path.join(root_dir, 'en', 'phrase-table.repo'))
    self.kb.freeze()
    self.extract_property_names()
    print('loading took', (time.time() - start), 'secs')


  def extract_property_names(self):
    print('storing property names ...')
    start = time.time()
    self.property_names = defaultdict(list)
    for frame in self.kb:
      if 'id' in frame and frame.id.startswith('P'):
        self.property_names[frame.id].append(frame.name)
    print('found', str(len(self.property_names)), 'properties')
    print('took', (time.time() - start), 'sec')


  def load_stanford_nlp(self):
    self.stanford = StanfordNLP()


  def load_filter(self, filename: str):
    with open(filename, 'r') as fin:
      self.property_names = json.load(fin)
      self.filter = set(self.property_names.keys())


  def load_single_hop_questions(self, filename: str, max_count: int=None):
    self.qa_pairs: List[Tuple[str, List[str]]] = []
    self.ans2qa: Dict[str, Set[int]] = defaultdict(set)
    self.ansent2qa: Dict[str, Set[int]] = defaultdict(set)
    with open(filename, 'r') as fin:
      for l in tqdm(fin):
        l = json.loads(l)
        self.qa_pairs.append((l['question'], l['answer']))
        ind = len(self.qa_pairs) - 1
        for a in l['answer']:
          self.ans2qa[a].add(ind)
          for e in self.phrase.lookup(a):
            self.ansent2qa[e.id].add(ind)
            break  # only use the first entity linking
        if max_count and len(self.qa_pairs) >= max_count:
          break


  @staticmethod
  def get_frame_id(frame):
    if 'id' in frame:
      return frame.id
    if 'is' in frame:
      if type(frame['is']) != sling.Frame:
        return None
      if 'id' in frame['is']:
        return frame['is'].id
    return None


  @staticmethod
  def get_date_property(prop, tail):
    if 'target' not in prop:
      return None
    if prop.target.id != '/w/time':
      return None
    prop_id = SlingExtractor.get_frame_id(prop)
    if type(tail) == int:
      return (prop_id, tail)
    elif type(tail) == sling.Frame and 'is' in tail and type(tail['is']) == int:
      return (prop_id, tail['is'])
    return None


  @staticmethod
  def get_canonical_property(prop, tail):
    if type(prop) != sling.Frame or type(tail) != sling.Frame:
      return None
    prop_id = SlingExtractor.get_frame_id(prop)
    tail_id = SlingExtractor.get_frame_id(tail)
    if prop_id is None:
      return None
    if tail_id is None:
      return None
    if not prop_id.startswith('P') or not tail_id.startswith('Q'):
      return None
    return (prop_id, tail_id)


  def get_type(self, wid) -> str:
    for type_prop in ['P31', 'P279']:
      try:
        return self.kb[self.kb[wid][type_prop].id].name
      except:
        pass
    return None


  def get_name(self, wid) -> str:
    return self.kb[wid].name


  def iter_property(self, wid: str, type: str='can', shuffle: bool=False):
    tup_li: List[Tuple] = []
    for prop, tail in self.kb[wid]:
      tup = self.get_canonical_property(prop, tail)
      if tup is not None and type == 'can':
        if not hasattr(self, 'filter') or tup[0] in self.filter:
          tup_li.append(tup)
        continue
      if tup is None:
        tup = self.get_date_property(prop, tail)
        if tup is not None and type == 'time':
          if not hasattr(self, 'filter') or tup[0] in self.filter:
            tup_li.append(tup)
          continue
    group = defaultdict(list)
    for k, v in tup_li:
      group[k].append(v)
    result = list(group.items())
    if shuffle:
      np.random.shuffle(result)
    return result


  def recursive_iter_property(self, wid: str, hop: int=2):
    for pid, eids in self.iter_property(wid, type='can', shuffle=True):
      if len(eids) > 1:
        continue
      for ppid, eeids in self.iter_property(eids[0], type='can', shuffle=True):
        if len(eeids) > 1:
          continue
        return pid, eids[0], ppid, eeids[0]
    return (None,) * 4


  def question2statement(self, question: str) -> str:
    raw_q = question
    nq = None
    question = question.split(' ')
    first_word = question[0]
    if first_word not in self.WH_WORDS:
      return None
    for i, w in enumerate(question[1:]):
      if w in {'is', 'are', 'was', 'were', 'am'}:
        if i == 0:
          nq = ' '.join([self.WH2THAT[first_word], 'that'] + question[1:])
          break
        else:
          nq = ' '.join(question[1:i + 1] + ['that'] + question[i + 1:])
          break
      elif w in {'do', 'did', 'does'}:
        if i == 0:
          nq = ' '.join([self.WH2THAT[first_word], 'that'] + question[2:])
          break
        else:
          nq = ' '.join(question[1:i + 1] + ['that'] + question[i + 2:])
          break
      elif w in {'that', 'which', 'who', 'where'}:  # another clause
        break
    if nq is None:
      nq = ' '.join([self.WH2THAT[first_word], 'that'] + question[1:])
    return nq


  def question2statement_parse(self, question: str, which: bool=False, keep_ph: bool=False, entity_type: str=None) -> str:
    question = question.strip()
    if not question.endswith('?'):
      question += '?'
    q_toks = question.split(' ')
    first_word = q_toks[0].lower()
    if first_word not in self.WH_WORDS:
      return None
    parse = self.stanford.annotate(question)
    try:
      stat = POSTree(parse).adjust_order().replace('-lrb-', '(').replace('-rrb-', ')')
    except:
      return None
    stat = stat.strip()
    if keep_ph:
      return stat
    if not stat.startswith(self.STAT_PH):
      return None
    if stat.startswith(self.STAT_PH) or stat.endswith(self.STAT_PH):
      stat = stat.replace(self.STAT_PH, '').strip()
    else:
      stat = stat.replace(' ' + self.STAT_PH + ' ', ' ').strip()
    assert self.STAT_PH not in stat
    if entity_type is None:
      before_that = self.WH2THAT[first_word]
    elif entity_type in self.TYPE2THAT:
      before_that = self.TYPE2THAT[entity_type]
    else:
      before_that = entity_type
    if which:
      nq = '{} {} {} {}'.format('which', before_that, 'that', stat)
    else:
      nq = '{} {} {}'.format(before_that, 'that', stat)
    return nq


  def get_ner(self, ners: List, sent: str=None, last: bool=False, only_one: bool=False) -> List:
    ners = [ner for ner in ners if ner[-1] not in {'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'}]
    if only_one and len(ners) > 1:  # only allow sentences with only one named entities
      return []
    if last:
      ners = [ner for ner in ners if ner[2] == len(sent)]
    return ners


  def extend_project_in(self, question: Dict, sample_n: int=1, use_qa_pairs: bool=False, run_el: bool=True) -> List[MultihopQuestion]:
    mqs = []
    # replace entities in the question
    if run_el:
      qes = self.get_ner(question['question_entity'], sent=question['question'], last=True, only_one=True)
    else:
      qes = question['question_entity']
    for qe in qes:
      qe_mention, qe_start, qe_end, qe_type = qe
      if run_el:
        qe_wikis = self.phrase.lookup(qe_mention)
        if len(qe_wikis) <= 0:
          return mqs
        qe_wid, qe_wname = qe_wikis[0].id, qe_wikis[0].name  # only the first of entity linking
      else:
        qe_wid = qe_type
        qe_type = self.get_type(qe_wid)
        if qe_type is None:
          return mqs
      if use_qa_pairs:
        if len(self.ansent2qa[qe_wid]) <= 0:
          return mqs
        for sn in np.random.choice(list(self.ansent2qa[qe_wid]), min(sample_n, len(self.ansent2qa[qe_wid])), replace=False):
          fir_q, fir_a = self.qa_pairs[sn]
          fir_stat = self.question2statement_parse(fir_q, entity_type=qe_type)
          if fir_stat is None:
            continue
          sec_q = question['question']
          sec_a = question['answers']
          multi_q = ' '.join([question['question'][:qe_start].rstrip(), fir_stat, question['question'][qe_end:].lstrip()])
          multi_a = sec_a
          mq = MultihopQuestion([{'q': fir_q, 'a': fir_a}, {'q': sec_q, 'a': sec_a}],
                                {'q': multi_q, 'a': multi_a}, ind=question['id'], op='project_in')
          mqs.append(mq)
      else:
        ps = self.iter_property(qe_wid, type='can')
        if len(ps) <= 0:
          return mqs
        for sn in np.random.choice(len(ps), min(sample_n, len(ps)), replace=False):
          pid, tailid = ps[sn]
          qe_type = self.get_type(qe_wid)
          if qe_type is None:
            continue
          if len(tailid) > 1:
            continue
          tailid = tailid[0]
          insert = '{} that {} {}'.format(qe_type, self.property_names[pid], self.get_name(tailid))  # TODO: this might not be unique
          fir_q = 'return ' + insert
          fir_a = [qe_mention]
          sec_q = question['question']
          sec_a = question['answers']
          multi_q = ' '.join([question['question'][:qe_start].rstrip(), insert, question['question'][qe_end:].lstrip()])
          multi_a = question['answers']
          mq = MultihopQuestion([{'q': fir_q, 'a': fir_a}, {'q': sec_q, 'a': sec_a}],
                                {'q': multi_q, 'a': multi_a}, ind=question['id'], op='project_in')
          mqs.append(mq)
    return mqs


  def extend_project_out(self, question: Dict, sample_n: int=1) -> List[MultihopQuestion]:
    mqs = []
    # replace entities in the answer
    if not (len(question['answers']) == 1 and len(question['answers_entity']) == 1 and len(question['answers_entity'][0]) == 1):  # only for sinlge-answer questions
      return mqs
    aes = self.get_ner(question['answers_entity'][0])
    if len(aes) <= 0:
      return mqs
    ae_mention, ae_start, ae_end, _ = aes[0]
    if ae_mention != question['answers'][0]:  # the whole answer is an entity
      return mqs
    ae_wikis = self.phrase.lookup(ae_mention)
    if len(ae_wikis) <= 0:
      return mqs
    ae_wid, ae_wname = ae_wikis[0].id, ae_wikis[0].name  # only the first of entity linking
    ps = self.iter_property(ae_wid, type='can')
    ae_type = self.get_type(ae_wid)
    if ae_type is None or len(ps) <= 0:
      return mqs
    for sn in np.random.choice(len(ps), min(sample_n, len(ps)), replace=False):
      pid, tailids = ps[sn]
      fir_q = question['question']
      fir_a = question['answers']
      sec_q = 'return {} {}?'.format(ae_mention, self.property_names[pid])
      sec_a = [self.get_name(tailid) for tailid in tailids]
      multi_q = 'return {} {} {}?'.format(ae_type, fir_q, self.property_names[pid])
      multi_a = sec_a
      mq = MultihopQuestion([{'q': fir_q, 'a': fir_a}, {'q': sec_q, 'a': sec_a}],
                            {'q': multi_q, 'a': multi_a}, ind=question['id'], op='project_out')
      mqs.append(mq)
    return mqs


  def extend_filter(self, question: Dict, sample_n: int=1, ques2stat: bool=False, run_el: bool=True) -> List[MultihopQuestion]:
    def find_common_filter(prop_dict_li: List[Dict[str, List]]) -> List[Tuple[str, str, List[int]]]:
      com_pid_tailids = []
      if len(prop_dict_li) <= 0:
        return []
      com_keys = set.intersection(*[set(pd.keys()) for pd in prop_dict_li])
      for key in com_keys:
        v = prop_dict_li[0][key]
        if len(v) != 1:  # only focus on single-value properties
          continue
        i2v = {0: v[0]}
        for i, pd in enumerate(prop_dict_li[1:]):
          if len(pd[key]) != 1:
            break
          i2v[i + 1] = pd[key][0]
        if len(i2v) != len(prop_dict_li):
          continue
        vs = set(i2v.values())  # with multiple values
        if len(vs) <= 1:
          continue
        v = np.random.choice(list(vs), 1)[0]
        com_pid_tailids.append((key, v, [i for i, _v in i2v.items() if _v == v]))
      return com_pid_tailids

    mqs = []
    if len(question['answers']) <= 1:  # more than one answer
      return mqs
    if len(question['answers_entity']) != len(question['answers']):
      return mqs
    for ae in question['answers_entity']:  # each answer correspond to one entity
      if run_el and len(self.get_ner(ae)) != 1:
        return mqs
      elif not run_el and len(ae) != 1:
        return mqs
    if run_el:
      aes = [self.get_ner(ae)[0] for ae in question['answers_entity']]
      aes_wiki = [self.phrase.lookup(ae[0]) for ae in aes]
      aes_wiki_id = [ae[0].id for ae in aes_wiki if len(ae) > 0]
      aes_wiki_name = [ae[0].name for ae in aes_wiki if len(ae) > 0]
    else:
      aes_wiki_id = [ae[0][-1] for ae in question['answers_entity']]
      aes_wiki_name = [self.get_name(aei) for aei in aes_wiki_id if self.get_name(aei)]
    if len(set(aes_wiki_id)) <= 1:  # only one entity
      return mqs
    if len(aes_wiki_id) != len(question['answers_entity']) or len(aes_wiki_id) != len(aes_wiki_name):  # all answers have entity linking ids and names
      return mqs
    pss_wiki = [dict(self.iter_property(ae, type='can')) for ae in aes_wiki_id]
    com_pid_tailids = find_common_filter(pss_wiki)
    if len(com_pid_tailids) <= 0:
      return mqs
    fir_q = question['question']
    fir_a = question['answers']
    fir_a_dedup = list(set(ae for ae in aes_wiki_name))
    for sn in np.random.choice(len(com_pid_tailids), min(sample_n, len(com_pid_tailids)), replace=False):
      pid, tailid, sub_inds = com_pid_tailids[sn]
      sec_q = 'Which one of the following {} {}: {}'.format(self.property_names[pid], self.get_name(tailid), ', '.join(fir_a_dedup))
      sec_a = [fir_a[sub] for sub in sub_inds]
      if ques2stat:
        stat = self.question2statement_parse(fir_q, which=True)  # use wh word as type
        if stat is None:
          continue
        multi_q = '{} {} {}'.format(stat, self.property_names[pid], self.get_name(tailid))
      else:
        multi_q = '{} and {} {}'.format(fir_q, self.property_names[pid], self.get_name(tailid))
      multi_a = sec_a
      mq = MultihopQuestion([{'q': fir_q, 'a': fir_a}, {'q': sec_q, 'a': sec_a}],
                            {'q': multi_q, 'a': multi_a}, ind=question['id'], op='filter')
      mqs.append(mq)
    return mqs


  def extend_agg(self, question: Dict, sample_n: int = 1) -> List[MultihopQuestion]:
    mqs = []
    if len(question['answers']) <= 1:
      return mqs
    if len(question['answers_entity']) != len(question['answers']):
      return mqs
    for ae in question['answers_entity']:
      if len(self.get_ner(ae)) <= 0:
        return mqs
    aes = [self.get_ner(ae)[0] for ae in question['answers_entity']]
    aes_wiki = [self.phrase.lookup(ae[0]) for ae in aes]
    aes_wiki = [ae[0] for ae in aes_wiki if len(ae) > 0]  # some mentions might not have linking
    if len(aes_wiki) != len(aes):
      return mqs
    aes_dedup = set(ae.id for ae in aes_wiki)
    aes_dedup = [self.get_name(ae) for ae in aes_dedup]
    fir_q = question['question']
    fir_a = question['answers']
    sec_q = 'return the number of the following: {}'.format(', '.join(aes_dedup))
    sec_a = str(len(aes_dedup))
    multi_q = 'return the number of {}'.format(fir_q)
    multi_a = sec_a
    mq = MultihopQuestion([{'q': fir_q, 'a': fir_a}, {'q': sec_q, 'a': sec_a}],
                          {'q': multi_q, 'a': multi_a}, ind=question['id'], op='agg')
    mqs.append(mq)
    return mqs


  def extend_superlative(self, question: Dict, sample_n: int = 1, ques2stat: bool=False, run_el: bool=True) -> List[MultihopQuestion]:
    sup2word = {'max': 'latest', 'min': 'earliest'}

    def find_superlative(prop_dict_li: List[Dict[str, List]]) -> List[Tuple[str, str, List[int]]]:
      com_pid_tailids = []
      if len(prop_dict_li) <= 0:
        return []
      com_keys = set.intersection(*[set(pd.keys()) for pd in prop_dict_li])
      for key in com_keys:
        vs = []
        for i, pd in enumerate(prop_dict_li):
          if len(pd[key]) != 1:
            break
          vs.append(pd[key][0])
        if len(vs) != len(prop_dict_li):
          continue
        maxv = np.max(vs)
        maxind = [i for i in range(len(vs)) if vs[i] == maxv]
        com_pid_tailids.append((key, 'max', maxind))
        minv = np.min(vs)
        minind = [i for i in range(len(vs)) if vs[i] == minv]
        com_pid_tailids.append((key, 'min', minind))
      return com_pid_tailids

    mqs = []
    if len(question['answers']) <= 1:
      return mqs
    if len(question['answers_entity']) != len(question['answers']):
      return mqs
    for ae in question['answers_entity']:  # each answer correspond to one entity
      if run_el and len(self.get_ner(ae)) != 1:
        return mqs
      elif not run_el and len(ae) != 1:
        return mqs
    if run_el:
      aes = [self.get_ner(ae)[0] for ae in question['answers_entity']]
      aes_wiki = [self.phrase.lookup(ae[0]) for ae in aes]
      aes_wiki_id = [ae[0].id for ae in aes_wiki if len(ae) > 0]
      aes_wiki_name = [ae[0].name for ae in aes_wiki if len(ae) > 0]
    else:
      aes_wiki_id = [ae[0][-1] for ae in question['answers_entity']]
      aes_wiki_name = [self.get_name(aei) for aei in aes_wiki_id if self.get_name(aei)]
    if len(set(aes_wiki_id)) <= 1:  # only one entity
      return mqs
    if len(aes_wiki_id) != len(question['answers_entity']) or len(aes_wiki_id) != len(aes_wiki_name):  # all answers have entity linking ids and names
      return mqs
    pss_wiki = [dict(self.iter_property(ae, type='time')) for ae in aes_wiki_id]
    com_pid_tailids = find_superlative(pss_wiki)
    if len(com_pid_tailids) <= 0:
      return mqs
    fir_q = question['question']
    fir_a = question['answers']
    fir_a_dedup = list(set(ae for ae in aes_wiki_name))
    for sn in np.random.choice(len(com_pid_tailids), min(sample_n, len(com_pid_tailids)), replace=False):
      pid, sup, sub_inds = com_pid_tailids[sn]
      sec_q = 'Which one of the following {} {}: {}'.format(self.property_names[pid], sup2word[sup], ', '.join(fir_a_dedup))
      sec_a = [fir_a[sub] for sub in sub_inds]
      if ques2stat:
        stat = self.question2statement_parse(fir_q, which=True)  # use wh word as type
        if stat is None:
          continue
        multi_q = '{} {} {}'.format(stat, self.property_names[pid], sup2word[sup])
      else:
        multi_q = '{} and {} {}'.format(fir_q, self.property_names[pid], sup2word[sup])
      multi_a = sec_a
      mq = MultihopQuestion([{'q': fir_q, 'a': fir_a}, {'q': sec_q, 'a': sec_a}],
                            {'q': multi_q, 'a': multi_a}, ind=question['id'], op='superlative')
      mqs.append(mq)
    return mqs


  def extend_intersection(self, question1: Dict, question2: Dict, ques2stat: bool=False) -> List[MultihopQuestion]:
    fir_q = question1['question']
    fir_a = question1['answers']
    sec_q = question2['question']
    sec_a = question2['answers']
    if set(fir_a) == set(sec_a):  # potentially same question
      return []
    if ques2stat:
      fir_stat = self.question2statement_parse(fir_q, which=True)
      sec_stat = self.question2statement_parse(sec_q, keep_ph=True)
      if fir_stat is None or sec_stat is None:
        return []
      if not sec_stat.startswith(self.STAT_PH):
        return []
      multi_q = sec_stat.replace(self.STAT_PH, fir_stat)
    else:
      multi_q = '{} and {}'.format(fir_q, sec_q)
    multi_a = list(set(fir_a) & set(sec_a))
    mq = MultihopQuestion([{'q': fir_q, 'a': fir_a}, {'q': sec_q, 'a': sec_a}],
                          {'q': multi_q, 'a': multi_a}, ind=question1['id'] + '&' + question2['id'], op='intersection')
    return [mq]


  def extend_union(self, question1: Dict, question2: Dict) -> List[MultihopQuestion]:
    fir_q = question1['question']
    fir_a = question1['answers']
    sec_q = question2['question']
    sec_a = question2['answers']
    multi_q = '{} or {}'.format(fir_q, sec_q)
    multi_a = list(set(fir_a) | set(sec_a))
    mq = MultihopQuestion([{'q': fir_q, 'a': fir_a}, {'q': sec_q, 'a': sec_a}],
                          {'q': multi_q, 'a': multi_a}, ind=question1['id'] + '|' + question2['id'], op='union')
    return [mq]


  def add_another_project_in(self, question: Dict, sample_n: int=1) -> List[MultihopQuestion]:
    mqs = []
    # replace entities in the question
    for qe in self.get_ner(question['question_entity']):
      qe_mention, qe_start, qe_end, _ = qe
      qe_wikis = self.phrase.lookup(qe_mention)
      if len(qe_wikis) <= 0:
        return mqs
      qe_wid, qe_wname = qe_wikis[0].id, qe_wikis[0].name  # only the first of entity linking
      if len(self.ansent2qa[qe_wid]) <= 0:
        return mqs
      for sn in np.random.choice(list(self.ansent2qa[qe_wid]), min(sample_n, len(self.ansent2qa[qe_wid])), replace=False):
        fir_q, fir_a = self.qa_pairs[sn]
        sec_q = question['question']
        sec_a = question['answers']
        mq = MultihopQuestion([{'q': fir_q, 'a': fir_a}, {'q': sec_q, 'a': sec_a}],
                              {'q': None, 'a': None}, ind=question['id'], op='project_in')
        mqs.append(mq)
    return mqs


  def add_another_union(self, question1: Dict, question2: Dict) -> List[MultihopQuestion]:
    fir_q = question1['question']
    fir_a = question1['answers']
    sec_q = question2['question']
    sec_a = question2['answers']
    mq = MultihopQuestion([{'q': fir_q, 'a': fir_a}, {'q': sec_q, 'a': sec_a}],
                          {'q': None, 'a': None}, ind=question1['id'] + '|' + question2['id'], op='union')
    return [mq]


  def add_another_intersection(self, question1: Dict, question2: Dict) -> List[MultihopQuestion]:
    fir_q = question1['question']
    fir_a = question1['answers']
    sec_q = question2['question']
    sec_a = question2['answers']
    mq = MultihopQuestion([{'q': fir_q, 'a': fir_a}, {'q': sec_q, 'a': sec_a}],
                          {'q': None, 'a': None}, ind=question1['id'] + '&' + question2['id'], op='union')
    return [mq]


  def extend(self, question: Dict, op: str, question2: Dict=None) -> List:
    if op == 'project_in':
      return self.extend_project_in(question, sample_n=5, use_qa_pairs=True, run_el=False)
    if op == 'project_out':
      return self.extend_project_out(question, sample_n=2)
    if op == 'filter':
      return self.extend_filter(question, sample_n=5, ques2stat=True, run_el=False)
    if op == 'agg':
      return self.extend_agg(question, sample_n=1)
    if op == 'superlative':
      return self.extend_superlative(question, sample_n=2, ques2stat=True, run_el=False)
    if op == 'union':
      return self.extend_union(question, question2)
    if op == 'intersection':
      return self.extend_intersection(question, question2, ques2stat=True)
    raise NotImplementedError


  def add_another(self, question: Dict, op: str, question2: Dict=None) -> List:
    if op == 'project_in':
      return self.add_another_project_in(question, sample_n=1)
    if op == 'union':
      return self.add_another_union(question, question2)
    if op == 'intersection':
      return self.add_another_intersection(question, question2)
    raise NotImplementedError
