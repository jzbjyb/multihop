import argparse
import json
from dataset import Break, HoptopQA


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', type=str)
  args = parser.parse_args()
  break_dataset = Break('/home/jzb/exp/Break/break_dataset/QDMR-high-level')
  hotpotqa = HoptopQA('/home/jzb/exp/Break/break_dataset/QDMR-high-level/hotpotqa')

  with open(args.output, 'w') as fout:
    for de in break_dataset.get_hotpotqa(hotpotqa, split='dev'):
      fout.write(json.dumps(de) + '\n')
