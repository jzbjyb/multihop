from typing import Dict, List
from collections import defaultdict
import argparse
import json
import numpy as np
from dataset import Break, HoptopQA
from rag.utils_rag import exact_match_score


numhops2temps: Dict[int, List[str]] = {
  2: ['n-*', 'p-*', '*-n', '*-p', 'n-n', 'n-p', 'p-n', 'p-p']
}


def nline_to_cate(nline: int, num_hops: int):
  return numhops2temps[num_hops][nline % len(numhops2temps[num_hops])]


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', type=str, choices=['hotpotqa', 'ana', 'nq'], default='hotpotqa')
  parser.add_argument('--input', type=str, nargs='+')
  parser.add_argument('--output', type=str)
  parser.add_argument('--split', type=str, default='dev')
  parser.add_argument('--num_hops', type=int, default=2)
  args = parser.parse_args()

  if args.task == 'hotpotqa':
    break_dataset = Break('/home/jzb/exp/Break/break_dataset/QDMR-high-level')
    hotpotqa = HoptopQA('/home/jzb/exp/Break/break_dataset/QDMR-high-level/hotpotqa')

    with open(args.output, 'w') as fout, open(args.output + '.source', 'w') as sfout, open(args.output + '.target', 'w') as tfout:
      for de in break_dataset.get_hotpotqa(hotpotqa, split=args.split):
        fout.write(json.dumps(de) + '\n')
        for sh in de['single-hop']:
          # use retrieval
          sfout.write('{}\t{}\t{}\n'.format(sh['q'], sh['c'][0], sh['c'][1]))
          tfout.write('{}\n'.format(sh['a']))
          # no retrieval
          sfout.write('{}\n'.format(sh['q']))
          tfout.write('{}\n'.format(sh['a']))
        mh = de['multi-hop']
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

  elif args.task == 'ana':
    pred_file, source_file, target_file = args.input
    ems = []
    groups = []
    cases = []

    with open(pred_file, 'r') as pfin, open(source_file, 'r') as sfin, open(target_file, 'r') as tfin:
      for i, l in enumerate(pfin):
        pred = l.strip()
        source = sfin.readline().strip()
        target = tfin.readline().strip()
        em = exact_match_score(pred, target)
        ems.append(em)
        if i % len(numhops2temps[args.num_hops]) == 0:
          groups.append([])
          cases.append([])
        groups[-1].append(em)
        if i % len(numhops2temps[args.num_hops]) in {0, 2, 4}:
          cases[-1].append((source, target, pred))
    temps = numhops2temps[args.num_hops]
    groups = [dict(zip(temps, group)) for group in groups]

    non_cate: Dict[str, int] = defaultdict(lambda: 0)
    para_cate: Dict[str, int] = defaultdict(lambda: 0)
    np_cate: Dict[str, int] = defaultdict(lambda: 0)

    cases_show = []

    for group, case in zip(groups, cases):
      key = '{:d}{:d}-{:d}'.format(group['n-*'], group['*-n'], group['n-n'])
      non_cate[key] += 1
      if key == '11-0':
        cases_show.append(case)

      para_cate['{:d}{:d}-{:d}'.format(group['p-*'], group['*-p'], group['p-p'])] += 1

      if group['n-*'] and group['*-p']:
        np_cate['np-{:d}'.format(group['n-p'])] += 1
      if group['p-*'] and group['*-n']:
        np_cate['pn-{:d}'.format(group['p-n'])] += 1

    print(sorted(non_cate.items(), key=lambda x: x[0]))
    print(sorted(para_cate.items(), key=lambda x: x[0]))
    print(sorted(np_cate.items(), key=lambda x: x[0]))

    for case in cases_show[:10]:
      print(case)

  elif args.task == 'nq':
    def read_file(in_fname, dir, split, first_ans=False):
      with open(in_fname) as fin, open(f'{dir}/{split}.source', 'w') as sfile, open(f'{dir}/{split}.target', 'w') as tfile:
        json_file = json.load(fin)
        size = 0
        for i, item in enumerate(json_file):
          id = item['id']
          question = item['question'].replace('\t', ' ').replace('\n', ' ')
          if '?' not in question:
            question += '?'
          answers = [answer.replace('\t', ' ').replace('\n', ' ') for answer in item['answer']]
          if first_ans:
            sfile.write(f'{question}\n')
            tfile.write(f'{answers[0]}\n')
            size += 1
          else:
            for answer in answers:
              sfile.write(f'{question}\n')
              tfile.write(f'{answer}\n')
              size += 1
      return size

    count_test = read_file('rag/nq_raw/nqopen/nqopen-test.json', 'rag/nq_raw', 'test', first_ans=True)
    count_dev = read_file('rag/nq_raw/nqopen/nqopen-dev.json', 'rag/nq_raw', 'val', first_ans=True)
    count_train = read_file('rag/nq_raw/nqopen/nqopen-train.json', 'rag/nq_raw', 'train', first_ans=True)
    print('train {} val {} test {}'.format(count_train, count_dev, count_test))
