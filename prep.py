from typing import Dict, List, Set, Tuple
from collections import defaultdict
import argparse
import json
import urllib
from random import shuffle
import numpy as np
import spacy
import truecase
from tqdm import tqdm
import matplotlib.pyplot as plot
from dataset import Break, HoptopQA, WebQeustion, ComplexWebQuestion, SlingExtractor, MultihopQuestion
from rag.utils_rag import exact_match_score, f1_score
from rag.eval_rag import get_scores


numhops2temps: Dict[int, List[str]] = {
  #2: ['n-*', 'p-*', '*-n', '*-p', 'n-n', 'n-p', 'p-n', 'p-p']
  2: ['n-*', '*-n', 'n-n']
}


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


def to_multihop(question_file: str, output_file: str, se: SlingExtractor, ops: List[str]):
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
        eqs = se.extend(question, op)
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
              eqs = se.extend(q1, op, question2=q2)
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


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', type=str, choices=['eval', 'hotpotqa', 'comqa', 'cwq', 'ana', 'ner', 'nq', 'ada', 'same', 'overlap', 'to_multihop', 'format'], default='hotpotqa')
  parser.add_argument('--input', type=str, nargs='+')
  parser.add_argument('--output', type=str)
  parser.add_argument('--split', type=str, default='dev')
  parser.add_argument('--num_hops', type=int, default=2)
  parser.add_argument('--thres', type=float, default=.0)
  parser.add_argument('--no_context', action='store_true')
  args = parser.parse_args()

  if args.task == 'eval':
    get_scores(None, preds_path=args.input[0], gold_data_path=args.input[1], question_data_path=args.input[2])

  elif args.task == 'hotpotqa':
    break_dataset = Break('/home/jzb/exp/Break/break_dataset/QDMR-high-level')
    print(sorted(break_dataset.ops2count['hotpot'].items(), key=lambda x: -x[1]))
    hotpotqa = HoptopQA('/home/jzb/exp/Break/break_dataset/QDMR-high-level/hotpotqa')

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
    wq = WebQeustion('/home/jzb/exp/Break/break_dataset/QDMR/webqsp')
    cwq = ComplexWebQuestion('/home/jzb/exp/Break/break_dataset/QDMR/complexwebq', webq=wq)
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
          for i, (k, v) in enumerate(stat):
            if i % 2 == 0:
              print('{}'.format(v), end='')
            else:
              print('\t{}'.format(v), end='\n')
        else:
          stat = sorted(stat.items())
          print(stat)

    pred_file, source_file, target_file, add_file = args.input
    ems = []
    groups = []
    cases = []
    cates = []

    addtion_res = None
    if 'cwq' in pred_file:
      addtion_res = ComplexWebQuestion('/home/jzb/exp/Break/break_dataset/QDMR/complexwebq',
                                       webq=WebQeustion('/home/jzb/exp/Break/break_dataset/QDMR/webqsp'))
    if 'op' in add_file:
      addtion_res = lambda x: x

    with open(pred_file, 'r') as pfin, open(source_file, 'r') as sfin, open(target_file, 'r') as tfin, open(add_file, 'r') as afin:
      for i, l in enumerate(pfin):
        pred = l.rstrip('\n').split('\t')[0]
        source = sfin.readline().strip()
        targets = tfin.readline().rstrip('\n').split('\t')
        addition = afin.readline().rstrip('\n')
        em = max(exact_match_score(pred, target) for target in targets)
        ems.append(em)
        if i % len(numhops2temps[args.num_hops]) == 0:
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
          cases[-1].append((source, targets, pred))
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
          fout.write('<div><span>{}</span></div>\n'.format(key))
          shuffle(case)
          for c in case[:5]:
            fout.write('<br>\n')
            for s, t, p in c:
              s = s.split('\t')[0]
              fout.write('<div>{}&nbsp;&nbsp;&nbsp;{}&nbsp;&nbsp;&nbsp;{}</div>'.format(s, t, p))
          fout.write('<hr>\n')

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
    se = SlingExtractor()
    se.load_kb(root_dir='/home/zhengbaj/tir4/sling/local/data/e/wiki')
    se.load_filter('wikidata_property_template.json')
    to_multihop(question_file, args.output, se,
                ops=['project_in', 'project_out', 'filter', 'agg', 'superlative', 'union', 'intersection'])

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
