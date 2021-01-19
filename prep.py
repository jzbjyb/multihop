import argparse
import json
from dataset import Break, HoptopQA


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', type=str)
  args = parser.parse_args()
  break_dataset = Break('/home/jzb/exp/Break/break_dataset/QDMR-high-level')
  hotpotqa = HoptopQA('/home/jzb/exp/Break/break_dataset/QDMR-high-level/hotpotqa')

  with open(args.output, 'w') as fout, open(args.output + '.source', 'w') as sfout, open(args.output + '.target', 'w') as tfout:
    for de in break_dataset.get_hotpotqa(hotpotqa, split='dev'):
      fout.write(json.dumps(de) + '\n')
      for sh in de['single-hop']:
        sfout.write('{}\t{}\t{}\n'.format(sh['q'], sh['c'][0], sh['c'][1]))
        tfout.write('{}\n'.format(sh['a']))
      mh = de['multi-hop']
      sfout.write('{}\t{}\t{}\n'.format(mh['q'], ' '.join([c[0] for c in mh['c']]), ' '.join([c[1] for c in mh['c']])))
      tfout.write('{}\n'.format(mh['a']))
