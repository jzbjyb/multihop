from typing import Dict, List
from collections import defaultdict
import argparse
import json
import urllib
from random import shuffle
import numpy as np
import matplotlib.pyplot as plot
from dataset import Break, HoptopQA
from rag.utils_rag import exact_match_score, f1_score
from rag.eval_rag import get_scores


numhops2temps: Dict[int, List[str]] = {
  2: ['n-*', 'p-*', '*-n', '*-p', 'n-n', 'n-p', 'p-n', 'p-p']
  #2: ['n-*', '*-n', 'n-n']
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


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', type=str, choices=['eval', 'hotpotqa', 'comqa', 'ana', 'nq', 'ada', 'same'], default='hotpotqa')
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

    pred_file, source_file, target_file = args.input
    ems = []
    groups = []
    cases = []

    with open(pred_file, 'r') as pfin, open(source_file, 'r') as sfin, open(target_file, 'r') as tfin:
      for i, l in enumerate(pfin):
        pred = l.rstrip('\n').split('\t')[0]
        source = sfin.readline().strip()
        targets = tfin.readline().rstrip('\n').split('\t')
        em = max(exact_match_score(pred, target) for target in targets)
        ems.append(em)
        if i % len(numhops2temps[args.num_hops]) == 0:
          groups.append([])
          cases.append([])
        groups[-1].append(em)
        if numhops2temps[args.num_hops][i % len(numhops2temps[args.num_hops])] in {'n-*', '*-n', 'n-n'}:
          cases[-1].append((source, targets, pred))
    temps = numhops2temps[args.num_hops]
    groups = [dict(zip(temps, group)) for group in groups]

    non_cate: Dict[str, int] = defaultdict(lambda: 0)
    para_cate: Dict[str, int] = defaultdict(lambda: 0)
    np_cate: Dict[str, int] = defaultdict(lambda: 0)

    cases_show = []

    for group, case in zip(groups, cases):
      key = '{:d}{:d}-{:d}'.format(group['n-*'], group['*-n'], group['n-n'])
      non_cate[key] += 1
      if key == '01-1':
        cases_show.append(case)

      if 'p-*' in group:
        para_cate['{:d}{:d}-{:d}'.format(group['p-*'], group['*-p'], group['p-p'])] += 1

        if group['n-*'] and group['*-p']:
          np_cate['np-{:d}'.format(group['n-p'])] += 1
        if group['p-*'] and group['*-n']:
          np_cate['pn-{:d}'.format(group['p-n'])] += 1

    print(sorted(non_cate.items(), key=lambda x: x[0]))
    print(sorted(para_cate.items(), key=lambda x: x[0]))
    print(sorted(np_cate.items(), key=lambda x: x[0]))

    for case in cases_show[:10]:
      printify(case)

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
