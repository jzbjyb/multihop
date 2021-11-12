from typing import Dict, Any, List, Tuple, Union, Set
import os
import json
import numpy as np
from collections import defaultdict
import requests
import re
from tqdm import tqdm
import spacy
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
    for split, file in [('train', 'data/WebQSP.train.json'), ('test', 'data/WebQSP.test.json')]:
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
        sql = ex['Parses'][0]['Sparql']
        if dedup_ans:
          new_answers = []
          for a in answers:
            if a not in new_answers:
              new_answers.append(a)
          self.dup_count += int(len(new_answers) < len(answers))
          answers = new_answers
        mid2name: Dict[str, str] = {}
        topic_mid = ex['Parses'][0]['TopicEntityMid']
        if topic_mid and topic_mid.startswith('m.'):
          mid2name[topic_mid] = ex['Parses'][0]['TopicEntityName']
        for c in ex['Parses'][0]['Constraints']:
          if c['Argument'].startswith('m.'):
            mid2name[c['Argument']] = c['EntityName']
        result[ex['QuestionId']] = {
          'id': ex['QuestionId'],
          'question': ex['ProcessedQuestion'],
          'answers': answers,
          'sql': ComplexWebQuestion.clean_sparql(sql),
          'mid2name': mid2name,
        }
      return result


  def __getitem__(self, item: str):
    for split in ['train', 'test']:
      if item in getattr(self, split):
        return getattr(self, split)[item]
    raise KeyError


class ComplexWebQuestion(object):
  NS = 'ns:'
  NS_MID = 'ns:m.'
  nlp = spacy.load('en_core_web_sm')
  BREAK2CWQ = {'project_in': 'composition', 'filter': 'conjunction', 'superlative': 'superlative'}

  def __init__(self, root_dir: str, webq: WebQuestion=None):
    print('loading ComplexWebQuestion ...')
    self.webq = webq
    self.dup_count = 0
    self.type2count = defaultdict(lambda: 0)
    self.splits = ['train', 'dev']
    for split, file in [('train', '1_1/ComplexWebQuestions_train.json'),
                        ('dev', '1_1/ComplexWebQuestions_dev.json')]:
      file = os.path.join(root_dir, file)
      if not os.path.exists(file):
        continue
      setattr(self, split, self.load_split(file))
    self.fix_sql_by_comparing()
    print('dup count {}'.format(self.dup_count))
    self.mid2wid: Dict[str, str] = self.load_mid2wid(os.path.join(root_dir, 'fb2w.nt'))
    self.mid2name: Dict[str, str] = self.load_mid2name_en(os.path.join(root_dir, 'freebase-mid-name.map.en'), filter=True)

  @staticmethod
  def load_mid2wid(filename: str) -> Dict[str, str]:
    with open(filename, 'r') as fin:
      mid2wid: Dict[str, str] = {}
      for i in range(4):
        fin.readline()
      for l in fin:
        mid, _, wid = l.strip().split('\t')[:3]
        mid = mid[:-1].rsplit('/', 1)[1]
        wid = wid.strip('.').strip()[:-1].rsplit('/', 1)[1]
        mid2wid[mid] = wid
    return mid2wid

  @staticmethod
  def load_mid2name(filename: str, language: Set[str] = {'en'}) -> Dict[str, str]:
    print(f'load mid2name from {filename}')
    mid2name: Dict[str, str] = {}
    with open(filename, 'r') as fin:
      for l in tqdm(fin):
        mid, name = l.strip().split('\t')
        mid = mid.rsplit('/', 1)[1][:-1]
        name, lang = name.rsplit('@', 1)
        name = name.strip('"')
        if lang not in language:
          continue
        mid2name[mid] = name
    return mid2name

  @staticmethod
  def get_mids_in_sql(sql: str) -> Set[str]:
    mids: Set[str] = set()
    for tok in sql.split():
      if tok.startswith(ComplexWebQuestion.NS_MID):
        mid = re.split(r"[^A-Za-z0-9_\.]+", tok[len(ComplexWebQuestion.NS):], 1)[0]
        mids.add(mid)
    return mids

  def load_mid2name_en(self, filename: str, filter: bool = False) -> Dict[str, str]:
    if filter and os.path.exists(filename + '.filter'):
      return self.load_mid2name_en(filename + '.filter', filter=False)
    print(f'load mid2name from {filename}')
    mid2name: Dict[str, str] = {}
    with open(filename, 'r') as fin:
      for l in tqdm(fin):
        mid, name = l.rstrip('\n').split('\t')
        mid2name[mid] = name
    if filter:
      used_mids: Set[str] = set()
      for split in self.splits:
        for k, v in getattr(self, split).items():
          for sql in [v['sql'], self.webq[v['webqsp_ID']]['sql']]:
            used_mids.update(self.get_mids_in_sql(sql))
      print(f'totally {len(used_mids)} mids involved in sql')
      mid2name = {mid: mid2name[mid] for mid in used_mids}
      with open(filename + '.filter', 'w') as fout:
        for mid, name in mid2name.items():
          fout.write(f'{mid}\t{name}\n')
    return mid2name

  @staticmethod
  def clean_sparql(sql: str):
    '''
    simple cleaning heursitcs that do not change the actual content
    '''
    sql = sql.strip().replace('\t', '')

    # remove date mark
    sql = sql.replace('^^xsd:dateTime', '')
    sql = sql.replace('xsd:datetime', '')

    # split by newline
    lines = sql.split('\n')

    # remove comment
    if lines[0].startswith('#MANUAL SPARQL'):  lines = lines[1:]
    if lines[0].startswith('#MANUAL ANSWERS'):  lines = lines[1:]

    # remove the prefix
    lines = lines[1:]

    # remove en filter
    isliteral = None
    for i in range(len(lines)):
      if 'isLiteral' in lines[i]:
        isliteral = i
        break
    if isliteral is not None:
      del lines[isliteral]

    # remove distinct because it's almost always included
    if lines[0].startswith('SELECT DISTINCT'):
      lines[0] = lines[0].replace(' DISTINCT', '').strip()

    # split lines where multile conditions are in the same line and accidentally connected
    _lines = []
    for l in lines:
      c = l.count(' .')
      if c > 1 and ' .?' in l:
        for _l in l.split(' .'):
          if len(_l.strip()) <= 0:
            continue
          _l = _l + ' .'
          _lines.append(_l)
      else:
        _lines.append(l)
    lines = _lines

    # remove comments
    _lines = []
    for l in lines:
      cc = l.count('#')
      assert cc <= 1
      _lines.append(l.split('#', 1)[0])

    # clear leading and ending spaces and remove empty line
    lines = [l.strip() for l in lines if len(l.strip()) > 0]
    return '\n'.join(lines)

  @staticmethod
  def post_clean_sparql(sql: str, simplify: bool = False):
    '''
    cleaning heursitcs that might change the actual content
    '''
    ns = ComplexWebQuestion.NS
    if simplify:
      # remove the first "?x !="
      lines = sql.split('\n')
      to_rm = None
      for i, l in enumerate(lines):
        if l.startswith('FILTER (?x !='):
          to_rm = i
          break
      if to_rm is not None:
        del lines[to_rm]

      # shorten relation id
      toks: List[str] = []
      for i, tok in enumerate(sql.split(' ')):
        if tok.startswith(ns):  # relation
          tok = tok[len(ns):].split('.')[-1]   # use the last part of the relation id
          toks.append(tok)
        else:
          toks.append(tok)
      sql = ' '.join(toks)
    sql = sql.replace('\n', ' ')
    return sql

  def fix_sql_by_comparing(self):
    for split in self.splits:
      for k, v in getattr(self, split).items():
        if v['type'] not in {'conjunction'}:
          continue
        sh_lines = self.webq[v['webqsp_ID']]['sql'].split('\n')
        mh_lines = v['sql'].split('\n')
        _mh_lines = []
        assert len(sh_lines) <= len(mh_lines)
        for i, (sl, ml) in enumerate(zip(sh_lines, mh_lines)):
          sl = sl.strip()
          ml = ml.strip()
          if ml != sl and ml.startswith(sl):  # multiple lines accidentally connected in the same line
            _mh_lines.extend([sl, ml[len(sl):]])
          else:
            _mh_lines.append(ml)
        v['sql'] = '\n'.join(_mh_lines + mh_lines[len(sh_lines):])

  def resolve_sparql(self, sql: str, mid2name: Dict[str, str]) -> str:
    ns = ComplexWebQuestion.NS
    ns_mid = ComplexWebQuestion.NS_MID
    for mid, name in mid2name.items():
      sql = sql.replace(ns + mid, name)
    if ns_mid in sql:  # if still has unresolved mids, use mid2name
      for mid in self.get_mids_in_sql(sql):
        if mid in self.mid2name:
          sql = sql.replace(ns + mid, self.mid2name[mid])
    assert ns_mid not in sql, f'{sql} has unresolved entities'
    return sql

  def get_name_by_request(self, mid: str, use_wikidata: bool = True) -> Union[str, None]:
    if use_wikidata:
      wid = self.mid2wid[mid]
      url = f'https://www.wikidata.org/w/api.php?action=wbgetentities&props=labels&ids={wid}&languages=en&format=json'
      name = json.loads(requests.get(url).text)['entities'][wid]['labels']['en']['value']
      return name
    else:
      url = f'http://sameas.org/store/freebase/json?uri=http://rdf.freebase.com/ns/{mid}'
      dups = json.loads(requests.get(url).text)[0]['duplicates']
      for dup in dups:
        if dup.startswith('http://dbpedia.org/resource'):
          return dup.rsplit('/', 1)[1].replace('_', ' ')
        if dup.startswith('http://rdf.freebase.com/ns/en.'):
          return dup.rsplit('.', 1)[1].replace('_', ' ')
      return None

  def extract_mid2name(self, sql: str, question: str) -> Dict[str, str]:
    mid2name: Dict[str, str] = {}
    for mid in self.get_mids_in_sql(sql):
      if mid in self.mid2name:
        name = self.mid2name[mid]
      else:
        try:
          name = self.get_name_by_request(mid=mid)  # 1. use wikidata
        except:
          name = self.get_name_by_request(mid=mid, use_wikidata=False)  # 2. use sameas
        if name is None:  # 3. use sentence
          question = ComplexWebQuestion.nlp(question)
          if len(question.ents) == 1:
            name = question.ents[0].text
        assert name, f'cant find name for {mid} in {question}'
      mid2name[mid] = name
    return mid2name

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
        self.type2count[ex['compositionality_type']] += 1
        result[ex['ID']] = {
          'id': ex['ID'],
          'type': ex['compositionality_type'],
          'question': ex['question'],
          'machine_question': ex['machine_question'],
          'answers': dedup_answers,
          'composition_answer': ex['composition_answer'],
          'webqsp_ID': ex['webqsp_ID'],
          'sql': ComplexWebQuestion.clean_sparql(ex['sparql'])
        }
      return result


  def get_dev_wq_ids(self):
    return [id.split('_', 1)[0] for id in self.dev.keys()]


  def get_train_wq_ids(self, with_type: bool=False):
    if with_type:
      return [id.split('_', 1)[0] + '#' + self.train[id]['type'] for id in self.train.keys()]
    return [id.split('_', 1)[0] for id in self.train.keys()]


  def extract_sub_sql(self,
                      multihop_sql: str,
                      singlehop_sql: str,
                      comp_type: str,
                      intermediate_answers: str = None) -> str:
    mh_lines = multihop_sql.split('\n')
    sh_lines = singlehop_sql.split('\n')
    if comp_type == 'composition':
      num_add_rules = len(mh_lines) - len(sh_lines)
      # TODO: remoce dup code
      filter_ind = None
      left_bracket_ind = None
      for i, l in enumerate(mh_lines):
        if filter_ind is None and l.startswith('FILTER') and '!=' in l:
          filter_ind = i
        if left_bracket_ind is None and '{' in l:
          left_bracket_ind = i
      if filter_ind is not None and left_bracket_ind is not None:  # choose the one appears earlier
        filter_ind = max(filter_ind, left_bracket_ind)
      else:
        filter_ind = filter_ind or left_bracket_ind
      if num_add_rules > 0:
        assert '?c' in ' '.join(mh_lines[filter_ind + 1:filter_ind + 1 + num_add_rules]), \
          f'?c is not used in \n{singlehop_sql}\n\n{multihop_sql}'
      another_lines = ['SELECT ?x WHERE {'] + \
                      [l.replace('?c', '?x') for l in mh_lines[filter_ind + 1:filter_ind + 1 + num_add_rules]] + \
                      ['}']
    elif comp_type in {'conjunction', 'superlative', 'comparative'}:
      num_add_rules = len(mh_lines) - len(sh_lines)
      filter_ind = None
      left_bracket_ind = None
      diff_ind = None
      for i, (ml, sl) in enumerate(zip(mh_lines, sh_lines)):
        ml = ml.strip()
        sl = sl.strip()
        if filter_ind is None and ml.startswith('FILTER') and '!=' in ml:
          filter_ind = i
        if left_bracket_ind is None and '{' in ml:
          left_bracket_ind = i
        if ml != sl:
          diff_ind = i
          break
      if filter_ind is not None and left_bracket_ind is not None:  # choose the one appears earlier
        filter_ind = min(filter_ind, left_bracket_ind)
      else:
        filter_ind = filter_ind or left_bracket_ind
      assert diff_ind > filter_ind, f'{diff_ind} > {filter_ind}?'
      another_lines = mh_lines[:filter_ind + 1] + [f'FILTER (?x in ({intermediate_answers}))'] + mh_lines[diff_ind:]
    else:
      raise NotImplementedError
    return '\n'.join(another_lines)


  def decompose_composition(self,
                            cwq: Dict,
                            wq: Dict,
                            use_ph: bool=False,
                            simplify_sql: bool=False) -> MultihopQuestion:
    sec_q = wq['question']
    sec_mid2name = wq['mid2name']
    sec_q_sql = self.resolve_sparql(wq['sql'], sec_mid2name)
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
    fir_q_sql = self.extract_sub_sql(cwq['sql'], wq['sql'], comp_type=cwq['type'])
    fir_mid2name = self.extract_mid2name(fir_q_sql, fir_q)
    fir_q_sql = self.resolve_sparql(fir_q_sql, {**fir_mid2name, **sec_mid2name})
    multi_q_sql = self.resolve_sparql(cwq['sql'], {**fir_mid2name, **sec_mid2name})

    fir_q_sql = self.post_clean_sparql(fir_q_sql, simplify=simplify_sql)
    sec_q_sql = self.post_clean_sparql(sec_q_sql, simplify=simplify_sql)
    multi_q_sql = self.post_clean_sparql(multi_q_sql, simplify=simplify_sql)

    return MultihopQuestion([{'q': fir_q, 'a': fir_a, 'sql': fir_q_sql},
                             {'q': sec_q, 'a': sec_a, 'sql': sec_q_sql}],
                            {'q': multi_q, 'a': multi_a, 'sql': multi_q_sql}, ind=cwq['id'])


  def decompose_conjunction(self,
                            cwq: Dict,
                            wq: Dict,
                            use_ph: bool=False,
                            simplify_sql: bool=False) -> MultihopQuestion:
    fir_q = wq['question']
    fir_a = wq['answers']
    fir_mid2name = wq['mid2name']
    fir_q_sql = self.resolve_sparql(wq['sql'], fir_mid2name)
    multi_q = cwq['question']
    multi_a = cwq['answers']
    assert cwq['machine_question'].startswith(wq['question']), 'no substring in conjunction'
    sec_q = cwq['machine_question'][len(wq['question']):].strip()
    sec_q = sec_q.lstrip('and').strip()
    #for rem in ['is', 'was', 'are', 'were']:
    #  sec_q = sec_q.lstrip(rem).strip()
    #sec_q = 'return ' + sec_q + ' from ' + ', '.join(fir_a)
    sec_q_ph = 'Which one of the following {}: #1?'.format(sec_q)
    choices = MultihopQuestion.format_multi_answers_with_alias(fir_a, only_first_alias=True, ans_sep=', ')
    sec_q = 'Which one of the following {}: {}?'.format(sec_q, choices)
    if use_ph:
      sec_q = sec_q_ph
    sec_q = sec_q.replace(' from after ', ' after ').replace(' from before ', ' before ')
    sec_a = cwq['answers']
    sec_q_sql = self.extract_sub_sql(cwq['sql'], wq['sql'], comp_type=cwq['type'], intermediate_answers=choices)
    sec_mid2name = self.extract_mid2name(sec_q_sql, sec_q)
    sec_q_sql = self.resolve_sparql(sec_q_sql, {**fir_mid2name, **sec_mid2name})
    multi_q_sql = self.resolve_sparql(cwq['sql'], {**fir_mid2name, **sec_mid2name})

    fir_q_sql = self.post_clean_sparql(fir_q_sql, simplify=simplify_sql)
    sec_q_sql = self.post_clean_sparql(sec_q_sql, simplify=simplify_sql)
    multi_q_sql = self.post_clean_sparql(multi_q_sql, simplify=simplify_sql)

    return MultihopQuestion([{'q': fir_q, 'a': fir_a, 'sql': fir_q_sql},
                             {'q': sec_q, 'a': sec_a, 'sql': sec_q_sql}],
                            {'q': multi_q, 'a': multi_a, 'sql': multi_q_sql}, ind=cwq['id'])


  def decompose_superlative(self, cwq: Dict, wq: Dict, use_ph: bool=False, simplify_sql: bool=False) -> MultihopQuestion:
    return self.decompose_conjunction(cwq, wq, use_ph=use_ph, simplify_sql=simplify_sql)


  def decompose_comparative(self, cwq: Dict, wq: Dict, use_ph: bool=False, simplify_sql: bool=False) -> MultihopQuestion:
    return self.decompose_conjunction(cwq, wq, use_ph=use_ph, simplify_sql=simplify_sql)


  def decompose(self, split: str, use_ph: bool=False, simplify_sql: bool=False):
    if self.webq is None:
      raise Exception('load webquestion first')
    for k, v in getattr(self, split).items():
      yield getattr(self, 'decompose_{}'.format(v['type']))(
        v, self.webq[v['webqsp_ID']], use_ph=use_ph, simplify_sql=simplify_sql)


  def __getitem__(self, item: str):
    for split in ['train', 'dev']:
      if item in getattr(self, split):
        return getattr(self, split)[item]
    raise KeyError(item)
