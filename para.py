import sys
from pathlib import Path
from typing import List, Dict
import torch.nn as nn
from fairseq.models.bart import BARTModel
from tqdm import tqdm


class Paraphraser(nn.Module):
    def generate(
        self,
        batch_source: List[str],
        beam_size: int,
        **kwargs
    ) -> List[List[Dict]]:
        raise NotImplementedError


class BartParaphraser(Paraphraser):
    def __init__(
        self,
        model_path: str
    ):
        super().__init__()
        model_path = Path(model_path)
        self.model = BARTModel.from_pretrained(
            str(model_path.parent),
            checkpoint_file=str(model_path.name),
        )
        self.model.cuda()
        self.model.eval()


    def generate(
        self,
        batch_source: List[str],
        beam_size: int,
        **kwargs
    ) -> List[List[Dict]]:
        hypotheses_batch = self.model.sample(
            batch_source,
            beam=beam_size,
            **kwargs
        )
        return hypotheses_batch


def write_to_file(batch, paras_li, fout, keep_size: int, dedup: bool=True):
  for source, paras in zip(batch, paras_li):
    if dedup:
      comb = [source]
      comb_score = [0.0]
      for para in paras:
        para_text = para['sentence'].strip('"')  # remove "
        if para_text not in comb:
          comb.append(para_text)
          comb_score.append(para['score'])
      comb = comb * (len(paras) + 1)
      comb_score = comb_score * (len(paras) + 1)
    else:
      comb = [source] + [para['sentence'] for para in paras]
      comb_score = [0.0] + [para['score'] for para in paras]
    dup_count = keep_size - len(set(comb[:keep_size]))
    for i in range(keep_size):
      fout.write('{}\t{}\n'.format(comb[i], comb_score[i]))
  return dup_count


if __name__ == '__main__':
    source_file, target_file, output_file = sys.argv[1:]
    batch_size = 128
    beam_size = 10
    keep_size = 5

    bart = BartParaphraser('/home/jzb/exp/knowlm/rag/models/paraphrase/checkpoint_best.pt')
    batch = []
    dup_count = count = 0
    with open(source_file, 'r') as sfin, open(output_file + '.source', 'w') as fout:
      for l in tqdm(sfin):
        count += 1
        batch.append(l.strip().split('\t')[0])
        if len(batch) >= batch_size:
          paras_li = bart.generate(batch, beam_size=beam_size)
          dup_count += write_to_file(batch, paras_li, fout, keep_size=keep_size, dedup=True)
          batch = []
      if len(batch) > 0:
        paras_li = bart.generate(batch, beam_size=beam_size)
        dup_count += write_to_file(batch, paras_li, fout, keep_size=keep_size, dedup=True)
        batch = []
    print('total {}, dup {}'.format(count, dup_count))

    with open(target_file, 'r') as tfin, open(output_file + '.target', 'w') as fout:
      for l in tqdm(tfin):
        for i in range(keep_size):
          fout.write(l)
