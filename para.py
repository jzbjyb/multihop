import argparse
from pathlib import Path
from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from fairseq.models.bart import BARTModel


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
    self.model = BARTModel.from_pretrained(str(model_path.parent), checkpoint_file=str(model_path.name))
    self.model.cuda()
    self.model.eval()


  def generate(
    self,
    batch_source: List[str],
    beam_size: int,
    **kwargs
  ) -> List[List[Dict]]:
    hypotheses_batch = self.model.sample(batch_source, beam=beam_size, **kwargs)
    return hypotheses_batch


  def eval_perp(self, batch_source: List[str], batch_target: List[str]):
    pad_token_id = 1
    # tokenization
    source = [self.model.encode(s)[0] for s in batch_source]
    source_len = torch.tensor([s.size(0) for s in source])
    source = nn.utils.rnn.pad_sequence(source, batch_first=True, padding_value=pad_token_id)
    target = [self.model.encode(s)[0] for s in batch_target]
    target = nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=pad_token_id)
    target_mask = target.ne(pad_token_id).float()
    # forward
    source = source.cuda()
    source_len = source_len.cuda()
    target = target.cuda()
    target_mask = target_mask.cuda()
    bs, seq_len = target.size()

    logits = self.model.model(source, source_len, prev_output_tokens=target)[0]
    lp = F.log_softmax(logits, dim=-1)
    lp = lp[:, :-1].contiguous()  # remove the last position
    target_shift = target[:, 1:].contiguous()  # remove the first token (bos)
    target_mask_shift = target_mask[:, 1:].contiguous()
    lp = torch.gather(lp.view(-1, lp.size(-1)), 1, target_shift.view(-1, 1)).view(bs, seq_len - 1)
    lp = lp * target_mask_shift
    slp = lp.sum(-1)
    return slp


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


def write_to_file_eval(f2t, t2f, fout):
  for s1, s2 in zip(f2t, t2f):
    fout.write('{}\t{}\n'.format(s1, s2))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', type=str, choices=['para', 'eval'], default='para')
  parser.add_argument('--input', type=str, nargs='+')
  parser.add_argument('--output', type=str)
  args = parser.parse_args()

  model_path = '/home/jzb/exp/knowlm/rag/models/paraphrase/checkpoint_best.pt'

  if args.task == 'para':
    source_file, target_file = args.input
    output_file = args.output
    batch_size = 128
    beam_size = 10
    keep_size = 5

    bart = BartParaphraser(model_path)
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

  elif args.task == 'eval':
    from_file, to_file = args.input
    output_file = args.output
    batch_size = 128
    num_para = 5

    bart = BartParaphraser(model_path)
    batch_source = []
    batch_target = []
    with open(from_file, 'r') as ffin, open(to_file, 'r') as tfin, open(output_file, 'w') as fout:
      for l in tqdm(ffin):
        f = l.strip().split('\t')[0]
        for i in range(num_para):
          t = tfin.readline().strip().split('\t')[0]
          batch_source.append(f)
          batch_target.append(t)
        if len(batch_source) >= batch_size:
          f2t = bart.eval_perp(batch_source, batch_target).detach().cpu().numpy()
          t2f = bart.eval_perp(batch_target, batch_source).detach().cpu().numpy()
          write_to_file_eval(f2t, t2f, fout)
          batch_source = []
          batch_target = []
      if len(batch_source) > 0:
        f2t = bart.eval_perp(batch_source, batch_target).detach().cpu().numpy()
        t2f = bart.eval_perp(batch_target, batch_source).detach().cpu().numpy()
        write_to_file_eval(f2t, t2f, fout)
