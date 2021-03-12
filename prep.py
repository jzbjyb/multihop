from typing import Dict, List, Set, Tuple, Union
from collections import defaultdict
import argparse
import json
import urllib
from random import shuffle
import numpy as np
import spacy
import truecase
from tqdm import tqdm
import os
import csv
import matplotlib.pyplot as plot
from dataset import Break, HoptopQA, WebQuestion, ComplexWebQuestion, SlingExtractor, MultihopQuestion, GraphQuestion
from rag.utils_rag import exact_match_score, f1_score
from rag.eval_rag import get_scores


numhops2temps: Dict[int, List[str]] = {
  #2: ['n-*', 'p-*', '*-n', '*-p', 'n-n', 'n-p', 'p-n', 'p-p']
  2: ['n-*', '*-n', 'n-n']
}
i2ph = {0: 'XXX', 1: 'YYY', 2: 'ZZZ', 3: 'AAA', 4: 'BBB', 5: 'CCC', 6: 'DDD', 7: 'EEE', 8: 'FFF', 9: 'GGG', 10: 'HHH'}


def get_se():
  se = SlingExtractor()
  se.load_kb(root_dir='/home/zhengbaj/tir4/sling/local/data/e/wiki')
  se.load_filter('wikidata_property_template.json')
  se.load_single_hop_questions('/home/zhengbaj/tir4/exp/PAQ/PAQ/PAQ.filtered.jsonl')
  os.environ['STANFORD_HOME'] = '/home/zhengbaj/tir4/stanford'
  se.load_stanford_nlp()
  return se


def nline_to_cate(nline: int, num_hops: int):
  return numhops2temps[num_hops][nline % len(numhops2temps[num_hops])]


def adaptive(pred1: str, pred2: str, gold_file: str, thres: float=0.0):
  em = f1 = total = 0
  uses = []
  with open(pred1, 'r') as p1fin, open(pred2, 'r') as p2fin, open(gold_file, 'r') as gfin:
    for line in p1fin:
      pred1, prob1 = line.rstrip('\n').split('\t')
      pred2, prob2 = p2fin.readline().rstrip('\n').split('\t')
      prob1, prob2 = float(prob1), float(prob2)
      golds = gfin.readline().rstrip('\n').split('\t')
      pred = pred1 if prob1 >= thres else pred2
      uses.append(int(prob1 >= thres))
      em += max(exact_match_score(pred, g) for g in golds)
      f1 += max(f1_score(pred, g) for g in golds)
      total += 1
  print('em {:.2f} f1 {:.2f} total {}, no ret {:.2f}'.format(em / total * 100, f1 / total * 100, total, np.mean(uses) * 100))


def traverse(se: SlingExtractor, outdir: str, max_id: int=50000, train_sample_num: int=10000, test_sample_num: int=2000):
  used_entities = set()
  train: List[Tuple] = []
  test: List[Tuple] = []
  pid2count: Dict[str, int] = defaultdict(lambda: 0)
  os.makedirs(outdir, exist_ok=True)
  with open(os.path.join(outdir, 'train.jsonl'), 'w') as trfout, open(os.path.join(outdir, 'test.jsonl'), 'w') as tefout:
    for id in tqdm(np.random.choice(list(range(1, max_id)), train_sample_num, replace=False)):
      id = 'Q{}'.format(id)
      if se.kb[id] is None:
        continue
      pid1, eid1, pid2, eid2 = se.recursive_iter_property(id)
      if pid1 is None:
        continue
      train.append((id, pid1, eid1, pid2, eid2))
      used_entities.update({id, eid1, eid2})
      pid2count[pid1] += 1
      pid2count[pid2] += 1
      j = {
        'step_id': [id, pid1, eid1, pid2, eid2],
        'step_name': [se.kb[id].name, se.property_names[pid1], se.kb[eid1].name, se.property_names[pid2], se.kb[eid2].name]}
      trfout.write(json.dumps(j) + '\n')
    for id in tqdm(np.random.choice(list(range(1, max_id)), test_sample_num, replace=False)):
      id = 'Q{}'.format(id)
      if se.kb[id] is None:
        continue
      pid1, eid1, pid2, eid2 = se.recursive_iter_property(id)
      if pid1 is None:
        continue
      if len({id, eid1, eid2} & used_entities) > 0:
        continue
      test.append((id, pid1, eid1, pid2, eid2))
      pid2count[pid1] += 1
      pid2count[pid2] += 1
      j = {
        'step_id': [id, pid1, eid1, pid2, eid2],
        'step_name': [se.kb[id].name, se.property_names[pid1], se.kb[eid1].name, se.property_names[pid2], se.kb[eid2].name]}
      tefout.write(json.dumps(j) + '\n')
    print('#train', len(train), '#test', len(test))
    print(sorted(pid2count.items(), key=lambda x: -x[1])[:10])


def to_multihop(question_file: str, output_file: str, se: SlingExtractor, ops: List[str], action: str='extend'):
  set_ops = {'union', 'intersection'}
  count = 0
  build_index = len(set_ops & set(ops)) > 0
  ans2ques: Dict[str, List[Dict]] = defaultdict(list)

  # no set ops
  no_ops = set(ops) - set_ops
  # set ops
  ops = set(ops) & set_ops
  with open(question_file, 'r') as fin, open(output_file, 'w') as fout:
    for l in tqdm(fin):
      question = json.loads(l)
      if build_index:
        for a in question['answers']:
          ans2ques[a].append(question)
      for op in no_ops:
        if action == 'extend':
          eqs = se.extend(question, op)
        elif action == 'add_another':
          eqs = se.add_another(question, op)
        else:
          raise NotImplementedError
        for eq in eqs:
          fout.write(str(eq) + '\n')
          count += 1

    used_question_pairs: Set[Tuple[str, str]] = set()
    if build_index:
      for _, questions in ans2ques.items():
        for i in range(len(questions)):
          for j in range(i + 1, len(questions)):
            q1, q2 = questions[i], questions[j]
            key = tuple(sorted([q1['id'], q2['id']]))
            if key in used_question_pairs:
              continue
            used_question_pairs.add(key)
            for op in ops:
              if action == 'extend':
                eqs = se.extend(q1, op, question2=q2)
              elif action == 'add_another':
                eqs = se.add_another(q1, op, question2=q2)
              else:
                raise NotImplementedError
              for eq in eqs:
                fout.write(str(eq) + '\n')
                count += 1

  print('total count {}'.format(count))


def overlap(pred1_file: str, pred2_file: str, source_file: str, target_file: str, ann_file: str):
  count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
  with open(pred1_file, 'r') as p1fin, open(pred2_file, 'r') as p2fin, \
    open(source_file, 'r') as sfin, open(target_file, 'r') as tfin, open(ann_file, 'r') as afin:
    for line in p1fin:
      pred1, prob1 = line.rstrip('\n').split('\t')
      pred2, prob2 = p2fin.readline().rstrip('\n').split('\t')
      prob1, prob2 = float(prob1), float(prob2)
      golds = tfin.readline().rstrip('\n').split('\t')
      cates = json.loads(afin.readline())['labels']
      em1 = max(exact_match_score(pred1, g) for g in golds)
      em2 = max(exact_match_score(pred2, g) for g in golds)
      for cate in cates:
        count[cate]['{:d}{:d}'.format(int(em1), int(em2))] += 1
      if 'no_question_overlap' in cates and 'no_answer_overlap' in cates:
        count['no_overlap']['{:d}{:d}'.format(int(em1), int(em2))] += 1
  for cate, stat in count.items():
    print(cate)
    total = sum(stat.values())
    stat = sorted(stat.items(), key=lambda x: x[0])
    print(stat)
    stat = ['{:.2f}%'.format(v / total * 100) for k, v in stat]
    for i, v in enumerate(stat):
      if i % 2 == 0:
        print(v, end='')
      else:
        print('\t' + v, end='\n')


def entity_linking_on_elq(input_file: str, output_file: str, dataset: Union[GraphQuestion, WebQuestion], se: SlingExtractor):
  with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
    found = total = 0
    for l in fin:
      l = json.loads(l)
      id = '{}-{}'.format(input_file, l['id'])
      question = l['text']
      question_entity = [(question[m[0]:m[1]], m[0], m[1], e) for m, e in zip(l['mentions'], l['wikidata_id'])]
      answers = dataset[l['id']]['answers']
      answers_entity = []
      for ans in answers:
        total += 1
        answers_entity.append([])
        ae_wikis = se.phrase.lookup(ans)  # match against the whole answer
        if len(ae_wikis) <= 0:
          continue
        found += 1
        answers_entity[-1].append((ans, 0, len(ans), ae_wikis[0].id))
      fout.write(json.dumps({'id': id,
                             'question': question,
                             'question_entity': question_entity,
                             'answers': answers,
                             'answers_entity': answers_entity}) + '\n')
  print('find {} among {} answer entities'.format(found, total))


def find_gold_retrieval(source_file: str, target_file: str, ret_file: str, output_file: str, num_gold: int=1):
  pos2gold: Dict[int, int] = defaultdict(lambda: 0)
  hasgold2count: Dict[int, int] = defaultdict(lambda: 0)
  fould_gold = total = 0
  with open(source_file, 'r') as sfin, \
    open(target_file, 'r') as tfin, \
    open(ret_file, 'r') as rfin, \
    open(output_file, 'w') as fout, \
    open(output_file + '.id', 'w') as ifout:
    for id, l in enumerate(sfin):
      question = l.strip().split('\t')[0]
      answers = tfin.readline().strip().split('\t')
      docs = rfin.readline().strip().split('\t')[:-1]
      docs = [doc.split(' || ') for doc in docs]
      has_gold = 0
      for i, doc in enumerate(docs):
        for answer in answers:
          if answer.lower() in (doc[1] + ' ' + doc[2]).lower():
            pos2gold[i] += 1
            has_gold += 1
            if has_gold <= num_gold:
              fout.write('{}\t{}\t{}\n'.format(question, doc[1], doc[2]))
              ifout.write('{}\n'.format(id))
            break
      fould_gold += int(has_gold > 0)
      hasgold2count[has_gold] += 1
      total += 1
  print('found {} among {} that have gold'.format(fould_gold, total))
  print('position -> portion with gold')
  for k, v in sorted(pos2gold.items(), key=lambda x: x[0]):
    print('{}\t{:.3f}'.format(k, v / total))
  print('#gold -> portion')
  for k, v in sorted(hasgold2count.items(), key=lambda x: x[0]):
    print('{}\t{:.3f}'.format(k, v / total))


def gold_retrieval_compare(source_file: str, target_file: str, pred_file: str,
                           ret_pred_file: str, ret_file_id: str,
                           gold_pred_file: str=None,
                           use_first_ret: bool=False, num_hop: int=2):
  id2preds: Dict[int, List[str]] = defaultdict(list)
  with open(ret_file_id, 'r') as rfin, open(ret_pred_file, 'r') as rpfin:
    for l in rfin:
      id = int(l.strip())
      pred = rpfin.readline().rstrip('\n').split('\t')[0]
      id2preds[id].append(pred)

  is_multi_li = []
  raw_em_li = []
  new_em_li = []
  with open(source_file, 'r') as sfin, \
    open(target_file, 'r') as tfin, \
    open(pred_file, 'r') as pfin:
    for i, l in enumerate(sfin):
      is_multi = (i + 1) % (num_hop + 1) == 0
      question = l.strip().split('\t')[0]
      targets = tfin.readline().strip().split('\t')
      pred = pfin.readline().rstrip('\n').split('\t')[0]
      ret_preds = id2preds[i]
      if len(ret_preds) <= 0:
        continue
      raw_em = max(exact_match_score(pred, target) for target in targets)
      new_ems = [max(exact_match_score(ret_pred, target) for target in targets) for ret_pred in ret_preds]
      new_em = new_ems[0] if use_first_ret else max(new_ems)
      raw_em_li.append(raw_em)
      new_em_li.append(new_em)
      is_multi_li.append(is_multi)

  # only work for num_hop = 2
  gold_em_li = []
  if gold_pred_file is not None:
    with open(gold_pred_file, 'r') as fin, open(target_file, 'r') as tfin:
      for _i, l in enumerate(fin):
        i = _i // 8
        ni = _i % 8
        i = i * 3 + ni // 2
        if ni not in {0, 2, 4}:
          continue
        targets = tfin.readline().strip().split('\t')
        if len(id2preds[i]) <= 0:
          continue
        pred = l.rstrip('\n').split('\t')[0]
        gold_em_li.append(max(exact_match_score(pred, target) for target in targets))
  if len(gold_em_li) <= 0:
    gold_em_li = [0] * len(new_em_li)

  assert len(gold_em_li) == len(new_em_li)
  raw_em_li = np.array(raw_em_li)
  new_em_li = np.array(new_em_li)
  gold_em_li = np.array(gold_em_li)
  is_multi_li = np.array(is_multi_li)
  print('ret {:.2f} pseudo gold {:.2f} gold {:.2f}'.format(
    np.mean(raw_em_li) * 100, np.mean(new_em_li) * 100, np.mean(gold_em_li) * 100))
  print('[MULTI] ret {:.2f} pseudo gold {:.2f} gold {:.2f}'.format(
    np.mean(raw_em_li[is_multi_li]) * 100, np.mean(new_em_li[is_multi_li]) * 100, np.mean(gold_em_li[is_multi_li]) * 100))
  print('[SINGLE] ret {:.2f} pseudo gold {:.2f} gold {:.2f}'.format(
    np.mean(raw_em_li[~is_multi_li]) * 100, np.mean(new_em_li[~is_multi_li]) * 100, np.mean(gold_em_li[~is_multi_li]) * 100))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', type=str, choices=[
    'eval', 'hotpotqa', 'convert_hotpotqa', 'comqa', 'cwq', 'ana', 'ner', 'ner_replace',
    'ner_fill', 'nq', 'ada', 'same', 'overlap', 'to_multihop', 'format',
    'format_sh_mh', 'dict2csv', 'format_traverse', 'combine_para', 'break_ana', 'el', 'load',
    'combine_tomultihop', 'gold_ret', 'gold_ret_compare'], default='hotpotqa')
  parser.add_argument('--input', type=str, nargs='+')
  parser.add_argument('--prediction', type=str, nargs='+')
  parser.add_argument('--output', type=str)
  parser.add_argument('--split', type=str, default='dev')
  parser.add_argument('--num_hops', type=int, default=2)
  parser.add_argument('--num_para', type=int, default=1)
  parser.add_argument('--thres', type=float, default=.0)
  parser.add_argument('--no_context', action='store_true')
  args = parser.parse_args()

  if args.task == 'eval':
    get_scores(None, preds_path=args.input[0], gold_data_path=args.input[1], question_data_path=args.input[2])

  elif args.task == 'convert_hotpotqa':
    hotpotqa = HoptopQA('../Break/break_dataset/QDMR-high-level/hotpotqa')
    with open(args.output + '.id', 'w') as fout, open(args.output + '.source', 'w') as sfout, open(args.output + '.target', 'w') as tfout:
      for id, item in getattr(hotpotqa, args.split).items():
        fout.write('{}\n'.format(id))
        sfout.write('{}\n'.format(item['question']))
        tfout.write('{}\n'.format(item['answer']))

  elif args.task == 'hotpotqa':
    break_dataset = Break('../Break/break_dataset/QDMR-high-level')
    print(sorted(break_dataset.ops2count['hotpot'].items(), key=lambda x: -x[1]))
    hotpotqa = HoptopQA('../Break/break_dataset/QDMR-high-level/hotpotqa')

    with open(args.output, 'w') as fout, open(args.output + '.source', 'w') as sfout, open(args.output + '.target', 'w') as tfout:
      for de in break_dataset.get_hotpotqa(hotpotqa, split=args.split):
        fout.write(json.dumps(de) + '\n')
        for sh in de['single-hop']:
          if not args.no_context:  # use retrieval
            sfout.write('{}\t{}\t{}\n'.format(sh['q'], sh['c'][0], sh['c'][1]))
            tfout.write('{}\n'.format(sh['a']))
          # no retrieval
          sfout.write('{}\n'.format(sh['q']))
          tfout.write('{}\n'.format(sh['a']))
        mh = de['multi-hop']
        if not args.no_context:
          # use all retrieval
          sfout.write('{}\t{}\t{}\n'.format(mh['q'], ' '.join([c[0] for c in mh['c']]), ' '.join([c[1] for c in mh['c']])))
          tfout.write('{}\n'.format(mh['a']))
          # use one retrieval
          sfout.write('{}\t{}\t{}\n'.format(mh['q'], mh['c'][0][0], mh['c'][0][1]))
          tfout.write('{}\n'.format(mh['a']))
          sfout.write('{}\t{}\t{}\n'.format(mh['q'], mh['c'][1][0], mh['c'][1][1]))
          tfout.write('{}\n'.format(mh['a']))
        # use no retrieval
        sfout.write('{}\n'.format(mh['q']))
        tfout.write('{}\n'.format(mh['a']))

  elif args.task == 'cwq':
    wq = WebQuestion('../Break/break_dataset/QDMR/webqsp')
    cwq = ComplexWebQuestion('../Break/break_dataset/QDMR/complexwebq', webq=wq)
    with open(args.output + '.id', 'w') as ifout,\
      open(args.output + '.source', 'w') as sfout, \
      open(args.output + '.single.target', 'w') as stfout, \
      open(args.output + '.multi.target', 'w') as mtfout:
      for de in cwq.decompose(split=args.split):
        for sh in de.single_hops:
          ifout.write(de.ind + '\n')
          sfout.write('{}\n'.format(sh['q']))
          stfout.write('{}\n'.format(sh['a'][0]))
          mtfout.write('{}\n'.format('\t'.join(sh['a'])))
        mh = de.multi_hop
        ifout.write(de.ind + '\n')
        sfout.write('{}\n'.format(mh['q']))
        stfout.write('{}\n'.format(mh['a'][0]))
        mtfout.write('{}\n'.format('\t'.join(mh['a'])))

  elif args.task == 'comqa':
    def parse_answer(answer, id):
      if answer.startswith('http'):
        answer = urllib.parse.unquote_plus(answer).rsplit('/', 1)[-1]
      answer = answer.replace('\n', ' ').replace('\t', ' ')
      return answer

    with open(args.input[0], 'r') as fin, open(args.output + '.source', 'w') as sfout, open(args.output + '.target', 'w') as tfout, open(args.output + '.cluster', 'w') as cfout:
      data = json.load(fin)
      for ex in data:
        answers = [parse_answer(a, ex['cluster_id']) for a in ex['answers']]
        answers = '\t'.join(answers)
        for q in ex['questions']:
          cfout.write(ex['cluster_id'] + '\n')
          sfout.write(q + '\n')
          tfout.write(answers + '\n')

  elif args.task == 'ana':
    def printify(case):
      print('---')
      for s, t, p in case:
        s = s.split('\t')[0]
        print(s, t, p)

    def printstat(stats: Dict, norm: bool=False, intable: bool=True):
      for cate, stat in stats.items():
        print('-> {}'.format(cate))
        t = sum(stat.values())
        if norm:
          stat = {k: '{:.2f}%'.format(v / t * 100) for k, v in stat.items()}
        if intable:
          sh, mh = list(stat.keys())[0].split('-')
          sh, mh = len(sh), len(mh)
          for i in np.array(np.meshgrid(*[[0, 1] for _ in range(sh + mh)])).T.reshape(-1, sh + mh):
            i = ''.join(map(str, i[:sh])) + '-' + ''.join(map(str, i[sh:]))
            if i not in stat:
              stat[i] = 0
          stat = sorted(stat.items())
          print(stat)
          for i, (k, v) in enumerate(stat):
            if i % 2 == 0:
              print('{}'.format(v), end='')
            else:
              print('\t{}'.format(v), end='\n')
        else:
          stat = sorted(stat.items())
          print(stat)

    pred_file, source_file, target_file, add_file = args.input[:4]
    score_file = args.input[4] if len(args.input) > 4 else add_file
    ems = []
    ems_first = []
    groups = []
    cases = []
    cates = []

    addtion_res = None
    if 'cwq' in pred_file:
      addtion_res = ComplexWebQuestion('../Break/break_dataset/QDMR/complexwebq',
                                       webq=WebQuestion('../Break/break_dataset/QDMR/webqsp'))
    if 'op' in add_file:
      addtion_res = lambda x: x

    with open(pred_file, 'r') as pfin, \
      open(source_file, 'r') as sfin, \
      open(target_file, 'r') as tfin, \
      open(add_file, 'r') as afin, \
      open(score_file, 'r') as scorefin:
      preds = []
      sources = []
      scores = []
      for i, l in enumerate(pfin):
        pred = l.rstrip('\n').split('\t')[0]
        source = sfin.readline().strip()
        score = scorefin.readline().strip().split('\t')
        preds.append(pred)
        sources.append(source)
        scores.append((float(score[0]), float(score[1])) if len(score) == 2 else (0, 0))
        if i % args.num_para == args.num_para - 1:
          pass
        else:
          continue
        targets = tfin.readline().rstrip('\n').split('\t')
        addition = afin.readline().rstrip('\n')
        em_li = [max(exact_match_score(pred, target) for target in targets) for pred in preds]
        em = max(em_li)
        ems.append(em)
        ems_first.append(em_li[0])
        if (i // args.num_para) % len(numhops2temps[args.num_hops]) == 0:
          groups.append([])
          cases.append([])
          if type(addtion_res) is ComplexWebQuestion:
            cates.append(addtion_res[addition]['type'])
          elif callable(addtion_res):
            cates.append(addtion_res(addition))
          else:
            cates.append('')
        groups[-1].append(em)
        if numhops2temps[args.num_hops][i % len(numhops2temps[args.num_hops])] in {'n-*', '*-n', 'n-n'}:
          scores[0] = (0, 0)
          rank = np.argsort([-s1 - s2 for s1, s2 in scores])
          sources = np.array(sources)[rank]
          preds = np.array(preds)[rank]
          em_li = np.array(em_li)[rank]
          scores = np.array(scores)[rank]
          cases[-1].append((sources, targets, preds, em_li, scores))
        preds = []
        sources = []
        scores = []
    print('em {:.2f}, only first {:.2f}'.format(np.mean(ems) * 100, np.mean(ems_first) * 100))
    temps = numhops2temps[args.num_hops]
    groups = [dict(zip(temps, group)) for group in groups]

    non_cate: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
    non_cate_case: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    para_cate: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
    np_cate: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))

    for group, case, cate in zip(groups, cases, cates):
      key = '{:d}{:d}-{:d}'.format(group['n-*'], group['*-n'], group['n-n'])
      non_cate[cate][key] += 1
      non_cate_case[cate][key].append(case)
      non_cate['*'][key] += 1
      non_cate_case['*'][key].append(case)

      if 'p-*' in group:
        para_cate[cate]['{:d}{:d}-{:d}'.format(group['p-*'], group['*-p'], group['p-p'])] += 1

        if group['n-*'] and group['*-p']:
          np_cate[cate]['np-{:d}'.format(group['n-p'])] += 1
        if group['p-*'] and group['*-n']:
          np_cate[cate]['pn-{:d}'.format(group['p-n'])] += 1

    printstat(non_cate)
    printstat(non_cate, norm=True)
    printstat(para_cate)
    printstat(np_cate)

    with open(args.output, 'w') as fout:
      for cate, cases in non_cate_case.items():
        fout.write('<h1>{}</h1>\n'.format(cate))
        for key, case in cases.items():
          fout.write('<h2>{}</h2>\n'.format(key))
          shuffle(case)
          fout.write('<h3>random cases</h3>\n')
          for c in case[:5]:
            fout.write('<br>\n')
            for s, t, p, e, scores in c:
              s = '<br>'.join(['{} ({:.2f}, {:.2f})'.format(_s.split('\t')[0], score[0], score[1]) for _s, score in zip(s, scores)])
              p = '&nbsp;&nbsp;&nbsp;'.join('{} ({})'.format(_p, _e) for _p, _e in zip(p, e))
              t = '&nbsp;&nbsp;&nbsp;'.join(t)
              fout.write('<div><div>Q: {}</div>\n<div style="padding-left: 80px;">G: {}</div>\n<div style="padding-left: 80px;">P: {}</div></div>'.format(s, t, p))
          if args.num_para > 1:
            fout.write('<h3>cases improved by paraphrases</h3>\n')
            use_count = 0
            for c in case:
              use = False
              for s, t, p, e, scores in c:
                if not e[0] and np.max(e[1:]):
                  use = True
              if not use:
                continue
              use_count += 1
              fout.write('<br>\n')
              for s, t, p, e, scores in c:
                s = '<br>'.join(['{} ({:.2f}, {:.2f})'.format(_s.split('\t')[0], score[0], score[1]) for _s, score in zip(s, scores)])
                p = '&nbsp;&nbsp;&nbsp;'.join('{} ({})'.format(_p, _e) for _p, _e in zip(p, e))
                t = '&nbsp;&nbsp;&nbsp;'.join(t)
                fout.write('<div><div>Q: {}</div>\n<div style="padding-left: 80px;">G: {}</div>\n<div style="padding-left: 80px;">P: {}</div></div>'.format(s, t, p))
              if use_count >= 5:
                break
          fout.write('<hr>\n')

  elif args.task == 'ner_fill':
    ori_file, para_file = args.input
    num_para = 5
    with open(ori_file, 'r') as ofin, open(para_file, 'r') as pfin, open(args.output, 'w') as fout:
      for l in ofin:
        ents = l.strip().split('\t')[1:]
        for i in range(num_para):
          p, s = pfin.readline().strip().split('\t')
          for j in range(len(ents)):
            p = p.replace(i2ph[j], ents[j])
          fout.write('{}\t{}\n'.format(p, s))

  elif args.task == 'ner_replace':
    nlp = spacy.load('en_core_web_sm')
    batch_size = 5000
    skip_ner_types = {'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'}

    def process_batch(li, fout):
      docs = list(nlp.pipe(li, disable=['parser']))
      for text, doc in zip(li, docs):
        ents = [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents if ent.label_ not in skip_ner_types]
        ents = sorted(ents, key=lambda x: x[1])
        new_text = []
        ent_text = []
        prev_ind = 0
        for i, ent in enumerate(ents):
          assert ent[1] >= prev_ind
          new_text.append(text[prev_ind:ent[1]])
          new_text.append(i2ph[i])
          ent_text.append(ent[0])
          prev_ind = ent[2]
        new_text.append(text[prev_ind:])
        new_text = ''.join(new_text)
        fout.write('{}\t{}\n'.format(new_text, '\t'.join(ent_text)))

    li = []
    with open(args.input[0], 'r') as fin, open(args.output, 'w') as fout:
      for l in tqdm(fin):
        li.append(l.strip())
        if len(li) < batch_size:
          continue
        process_batch(li, fout)
        li = []
      if len(li) > 0:
        process_batch(li, fout)

  elif args.task == 'ner':
    nlp = spacy.load('en_core_web_sm')
    batch_size = 5000

    def get_entities(doc):
      return [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

    def process_batch(id_li, q_li, a_li, fout):
      q_docs = list(nlp.pipe(q_li, disable=['parser']))
      a_docs = list(nlp.pipe([a for anss in a_li for a in anss], disable=['parser']))
      prev_ind = 0
      for j, (id, q, anss, q_doc) in enumerate(zip(id_li, q_li, a_li, q_docs)):
        q_es = get_entities(q_doc)
        ans_es = []
        for a in anss:
          a_doc = a_docs[prev_ind]
          prev_ind += 1
          ans_es.append(get_entities(a_doc))
        fout.write(json.dumps({
          'id': id,
          'question': q, 'question_entity': q_es,
          'answers': anss, 'answers_entity': ans_es}) + '\n')

    def ner_file(in_fname, dir, split, format='jsonl'):
      with open(in_fname) as fin, open(f'{dir}/{split}.ner.txt', 'w') as fout:
        if format == 'json':
          fin = json.load(fin)
        id_li: List[str] = []
        q_li: List[str] = []
        a_li: List[List[str]] = []
        for i, item in tqdm(enumerate(fin)):
          if format == 'jsonl':
            item = json.loads(item)
          id = item['id'] if 'id' in item else str(i)
          question = item['question']
          answers = item['answer']
          id_li.append(id)
          q_li.append(truecase.get_true_case(question))
          a_li.append(answers)
          if len(id_li) < batch_size:
            continue
          process_batch(id_li, q_li, a_li, fout)
          id_li = []
          q_li = []
          a_li = []
        if len(id_li) > 0:
          process_batch(id_li, q_li, a_li, fout)

    ner_file('rag/nq/nqopen/nqopen-test.json', 'rag/nq/nqopen', 'test', format='json')
    ner_file('rag/nq/nqopen/nqopen-dev.json', 'rag/nq/nqopen', 'val', format='json')
    ner_file('rag/nq/nqopen/nqopen-train.json', 'rag/nq/nqopen', 'train', format='json')
    ner_file('../PAQ/PAQ/PAQ.filtered.jsonl', '../PAQ/PAQ', 'all', format='jsonl')

  elif args.task == 'nq':
    def read_file(in_fname, dir, split, ans_format='first'):
      with open(in_fname) as fin, open(f'{dir}/{split}.source', 'w') as sfile, open(f'{dir}/{split}.target', 'w') as tfile:
        json_file = json.load(fin)
        size = 0
        for i, item in enumerate(json_file):
          id = item['id']
          question = item['question'].replace('\t', ' ').replace('\n', ' ')
          if '?' not in question:
            question += '?'
          answers = [answer.replace('\t', ' ').replace('\n', ' ') for answer in item['answer']]
          if ans_format == 'first':
            sfile.write(f'{question}\n')
            tfile.write(f'{answers[0]}\n')
            size += 1
          elif ans_format == 'multi':
            for answer in answers:
              sfile.write(f'{question}\n')
              tfile.write(f'{answer}\n')
              size += 1
          elif ans_format == 'single':
            sfile.write(f'{question}\n')
            tfile.write('\t'.join(answers) + '\n')
            size += len(answers)
      return size

    count_test = read_file('rag/nq_raw/nqopen/nqopen-test.json', 'rag/nq_raw', 'test', ans_format='single')
    count_dev = read_file('rag/nq_raw/nqopen/nqopen-dev.json', 'rag/nq_raw', 'val', ans_format='single')
    count_train = read_file('rag/nq_raw/nqopen/nqopen-train.json', 'rag/nq_raw', 'train', ans_format='first')
    print('train {} val {} test {}'.format(count_train, count_dev, count_test))

  elif args.task == 'ada':
    adaptive(pred1=args.input[0], pred2=args.input[1], gold_file=args.input[2], thres=args.thres)

  elif args.task == 'same':
    pred_file, source_file, target_file, cluster_file = args.input
    ems = []
    cases = []
    prev_cluster = None
    with open(pred_file, 'r') as pfin, open(source_file, 'r') as sfin, open(target_file, 'r') as tfin, open(cluster_file, 'r') as cfin:
      for l in pfin:
        pred = l.rstrip('\n').split('\t')[0]
        question = sfin.readline().rstrip('\n')
        golds = tfin.readline().rstrip('\n').split('\t')
        em = max(exact_match_score(pred, gold) for gold in golds)
        cluster = cfin.readline().rstrip('\n')
        if cluster != prev_cluster:
          ems.append([])
          cases.append([])
          prev_cluster = cluster
        ems[-1].append(em)
        cases[-1].append((pred, golds, question))
    cc = cw = ic = 0
    ic_cases = []
    for c, case in zip(ems, cases):
      l = len(c)
      c = sum(c)
      if l <= 1:
        continue
      if c == l:
        cc += 1
      elif c == 0:
        cw += 1
      else:
        ic += 1
        ic_cases.append(case)
    print('consistently correct {} wrong {}, inconsistent {}'.format(cc, cw, ic))
    shuffle(ic_cases)
    for case in ic_cases[:5]:
      print('================')
      p, g, q = case[0]
      print(g)
      for p, g, q in case:
        print(q, p, sep='\t')

  elif args.task == 'overlap':
    pred1, pred2, source_file, target_file, ana_file = args.input
    overlap(pred1, pred2, source_file, target_file, ana_file)

  elif args.task == 'to_multihop':
    question_file = args.input
    se = get_se()
    to_multihop(question_file, args.output, se,
                ops=['project_in', 'project_out', 'filter', 'agg', 'superlative', 'union', 'intersection'],
                action='extend')
    to_multihop(question_file, args.output, se,
                ops=['project_in', 'union', 'intersection'],
                action='add_another')

  elif args.task == 'format':
    with open(args.input[0], 'r') as fin, \
      open(args.output + '.source', 'w') as sfout, \
      open(args.output + '.target', 'w') as tfout, \
      open(args.output + '.op', 'w') as ofout:
      maxu = maxi = 0
      for l in fin:
        mhq = MultihopQuestion.fromstr(l)
        op = mhq.kwargs['op']
        if op == 'union':
          if maxu >= 500:
            continue
          maxu += 1
        if op == 'intersection':
          if maxi >= 500:
            continue
          maxi += 1
        for sh in mhq.single_hops:
          sfout.write(sh['q'] + '\n')
          tfout.write('\t'.join(sh['a']) + '\n')
          ofout.write(op + '\n')
        mh = mhq.multi_hop
        sfout.write(mh['q'] + '\n')
        tfout.write('\t'.join(mh['a']) + '\n')
        ofout.write(op + '\n')

  elif args.task == 'format_sh_mh':
    tomultihop, addanother = args.input
    with open(tomultihop, 'r') as tfin, \
      open(addanother, 'r') as afin, \
      open(args.output + '.sh', 'w') as sfout, \
      open(args.output + '.op', 'w') as ofout:
      for l in afin:
        mhq = MultihopQuestion.fromstr(l)
        op = mhq.kwargs['op']
        sfout.write(mhq.single_hops[0]['q'] + ' ' + mhq.single_hops[1]['q'] + '\n')
        ofout.write(op + '\n')
      for l in tfin:
        mhq = MultihopQuestion.fromstr(l)
        op = mhq.kwargs['op']
        if op not in {'filter', 'superlative'}:
          continue
        sfout.write(mhq.single_hops[0]['q'] + ' ' + mhq.single_hops[1]['q'] + '\n')
        ofout.write(op + '\n')

  elif args.task == 'dict2csv':
    num_docs = 5233329
    with open(args.input[0], 'r') as fin, open(args.output, 'w', newline='') as fout:
      csvwriter = csv.writer(fout, delimiter='\t')
      data = json.load(fin)
      for i in range(num_docs):
        d = data[str(i)]
        title, text = d['title'], d['text']
        csvwriter.writerow([title, text])

  elif args.task == 'format_traverse':
    with open(args.input[0], 'r') as fin, \
      open(args.output + '.source', 'w') as sfout, \
      open(args.output + '.target', 'w') as tfout:
      for l in fin:
        l = json.loads(l)
        s, p1, e1, p2, e2 = l['step_name']
        sfout.write('{}: {}\n'.format(s, p1))
        tfout.write('{}\n'.format(e1))
        sfout.write('{}: {}\n'.format(e1, p2))
        tfout.write('{}\n'.format(e2))
        sfout.write('{}: {}, {}\n'.format(s, p1, p2))
        tfout.write('{}\n'.format(e2))

  elif args.task == 'combine_para':
    fins = [open(i, 'r') for i in args.input]
    sfins = [open(i + '.score', 'r') for i in args.input]
    pfins = [open(i, 'r') for i in args.prediction]
    with open(args.output, 'w') as fout, \
      open(args.output + '.score', 'w') as sfout, \
      open(args.prediction[0] + '.combine', 'w') as pfout:
      for _fins, _fout in [(fins, fout), (sfins, sfout), (pfins, pfout)]:
        while True:
          try:
            for i, fin in enumerate(_fins):
              for j in range(args.num_para):
                l = fin.readline()
                if l == '':
                  raise EOFError
                if i > 0 and j == 0:
                  continue
                _fout.write(l)
          except EOFError:
            break

  elif args.task == 'break_ana':
    hotpotqa = HoptopQA('../Break/break_dataset/QDMR-high-level/hotpotqa')
    cwq = ComplexWebQuestion('../Break/break_dataset/QDMR/complexwebq', webq=None)
    break_dataset = Break('../Break/break_dataset/QDMR-high-level')
    break_dataset.add_answers(cwq=cwq, hotpotqa=hotpotqa)
    consist = {}
    inconsist = {}
    total_count = consist_count = 0
    op2con_incon = defaultdict(lambda: [0, 0])
    op2cor_incor = defaultdict(lambda: {'00': 0, '01': 0, '10': 0, '11': 0})
    hop2con_incon = defaultdict(lambda: [0, 0])
    hop2cor_incor = defaultdict(lambda: {'00': 0, '01': 0, '10': 0, '11': 0})
    hop2count = defaultdict(lambda: 0)
    op2count = defaultdict(lambda: 0)
    with open(args.input[0], 'r') as fin:
      data = json.load(fin)
      for id, entry in data.items():
        if break_dataset.parse_id(id)[0] not in {'CWQ', 'HOTPOT'}:
          continue
        total_count += 1
        ops = set(entry['operators'].split('-'))
        targets = break_dataset[id]['answers']
        multi_pred = entry['prediction'].strip()
        single_pred = entry['decomposition_prediction'][-1].strip()
        nh = len(entry['operators'].split('-'))
        hop2count[nh] += 1
        for op in ops:
          op2count[op] += 1
        if exact_match_score(multi_pred, single_pred):
          consist_count += 1
          consist[id] = entry
          for op in ops:
            op2con_incon[op][0] += 1
            hop2con_incon[nh][0] += 1
        else:
          inconsist[id] = entry
          for op in ops:
            op2con_incon[op][1] += 1
            hop2con_incon[nh][1] += 1
        multi_correct = int(max([exact_match_score(multi_pred, target) for target in targets]))
        single_correct = int(max([exact_match_score(single_pred, target) for target in targets]))
        op2cor_incor['*']['{}{}'.format(single_correct, multi_correct)] += 1
        hop2cor_incor[nh]['{}{}'.format(single_correct, multi_correct)] += 1
        for op in ops:
          op2cor_incor[op]['{}{}'.format(single_correct, multi_correct)] += 1

    print('{}/{}'.format(consist_count, total_count))
    with open(args.output + '.consist', 'w') as fout:
      json.dump(consist, fout, indent=2)
    with open(args.output + '.inconsist', 'w') as fout:
      json.dump(inconsist, fout, indent=2)

    op2con_incon = [(k, (v1 / (v1 + v2) * 100, v2 / (v1 + v2) * 100)) for k, (v1, v2) in op2con_incon.items()]
    op2con_incon = sorted(op2con_incon, key=lambda x: -x[1][0])
    for k, (v1, v2) in op2con_incon:
      print('{}: {:.2f}%'.format(k, v1))

    hop2con_incon = [(k, (v1 / (v1 + v2) * 100, v2 / (v1 + v2) * 100)) for k, (v1, v2) in hop2con_incon.items()]
    hop2con_incon = sorted(hop2con_incon, key=lambda x: x[0])
    for k, (v1, v2) in hop2con_incon:
      print('{}: {:.2f}%'.format(k, v1))

    for k, v in op2cor_incor.items():
      t = sum(v.values())
      new_v = {_k: '{:.2f}%'.format(_v / t * 100) for _k, _v in v.items()}
      print(k, '\t'.join([x[1] for x in sorted(new_v.items())]), sep='\t')

    for k in sorted(hop2cor_incor.keys()):
      v = hop2cor_incor[k]
      t = sum(v.values())
      new_v = {_k: '{:.2f}%'.format(_v / t * 100) for _k, _v in v.items()}
      print(k, '\t'.join([x[1] for x in sorted(new_v.items())]), sep='\t')

    print(sorted(hop2count.items(), key=lambda x: -x[1]))
    op2count = list(sorted(op2count.items(), key=lambda x: -x[1]))
    print('\t'.join([x[0] for x in op2count]))
    print('\t'.join(['{:.2f}'.format(x[1] / total_count * 100) for x in op2count]))


  elif args.task == 'el':
    se = get_se()
    ds = GraphQuestion('elq')
    #ds = WebQuestion('elq')
    entity_linking_on_elq(args.input[0], args.output, dataset=ds, se=se)

  elif args.task == 'load':
    from transformers import RagRetriever
    retriever = RagRetriever.from_pretrained('facebook/rag-sequence-base')

  elif args.task == 'combine_tomultihop':
    sample_count = 500
    op2mhqs = defaultdict(list)
    with open(args.output + '.source', 'w') as sfout, \
      open(args.output + '.target', 'w') as tfout, \
      open(args.output + '.op', 'w') as ofout:
      for dir in args.input:
        for root, dirs, files in os.walk(dir):
          for file in files:
            if not 'tomultihop' in file or not file.endswith('.jsonl'):
              continue
            with open(os.path.join(root, file), 'r') as fin:
              for l in fin:
                mhq = MultihopQuestion.fromstr(l)
                op = mhq.kwargs['op']
                op2mhqs[op].append(mhq)
      print(sorted([(k, len(v)) for k, v in op2mhqs.items()], key=lambda x: -x[1]))
      for op, mhqs in op2mhqs.items():
        mhqs = np.random.choice(mhqs, min(len(mhqs), sample_count), replace=False)
        for mhq in mhqs:
          for sh in mhq.single_hops:
            sfout.write(sh['q'] + '\n')
            tfout.write('\t'.join(sh['a']) + '\n')
            ofout.write(op + '\n')
          mh = mhq.multi_hop
          sfout.write(mh['q'] + '\n')
          tfout.write('\t'.join(mh['a']) + '\n')
          ofout.write(op + '\n')

  elif args.task == 'gold_ret':
    source_file, target_file, ret_file = args.input
    output_file = args.output
    num_gold = 5
    find_gold_retrieval(source_file, target_file, ret_file, output_file, num_gold=num_gold)

  elif args.task == 'gold_ret_compare':
    source_file, target_file, pred_file, ret_pred_file, ret_file_id = args.input[:5]
    gold_pred_file = args.input[5] if len(args.input) > 5 else None
    print('first ret')
    gold_retrieval_compare(source_file, target_file, pred_file, ret_pred_file, ret_file_id,
                           gold_pred_file=gold_pred_file, use_first_ret=True)
    print('all ret')
    gold_retrieval_compare(source_file, target_file, pred_file, ret_pred_file, ret_file_id,
                           gold_pred_file=gold_pred_file, use_first_ret=False)
