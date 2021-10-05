from typing import List, Tuple
from operator import itemgetter
import numpy as np
import torch
from transformers import BartTokenizer


NEED_PREFIX_TOKENIZER = {BartTokenizer}

def get_num_tokens(text: str, tokenizer):
  if type(tokenizer) in NEED_PREFIX_TOKENIZER:
    return len(tokenizer.tokenize(text, add_prefix_space=True))
  return len(tokenizer.tokenize(text))


def decode_keep_mask(ids: List[int], tokenizer) -> str:
  skip = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
  # add a space before mask (mainly for <s> of roberta/bart)
  return ''.join([' {}'.format(tokenizer.mask_token) if id == tokenizer.mask_token_id else tokenizer.decode([id]) for id in ids if id not in skip])


def predict_batch(model, tokenizer, questions, answers, mask_num_hint: bool=False, max_num_mask: int=1, init_mask_token: str='[MASK]'):
  results: List[Tuple[str, float]] = []
  new_questions = []
  for question, answer in zip(questions, answers):
    # make sure that mask is followed by space (to avoid bugs in BART)
    mask_end = question.find(init_mask_token) + len(init_mask_token)
    if mask_end < len(question) and question[mask_end] != ' ':
      question = question.replace(init_mask_token, init_mask_token + ' ')
    if max_num_mask == 1:
      masks = ' '.join([tokenizer.mask_token] * (get_num_tokens(answer, tokenizer) if mask_num_hint else 1))
      new_questions.append(question.replace(init_mask_token, masks))
    else:
      for i in range(max_num_mask):
        masks = ' '.join([tokenizer.mask_token] * (i + 1))
        new_questions.append(question.replace(init_mask_token, masks))
  batch = tokenizer(new_questions, padding=True, return_tensors='pt')
  batch = {k: v.cuda() for k, v in batch.items()}
  mask_bools = batch['input_ids'].eq(tokenizer.mask_token_id)
  with torch.no_grad():
    logprobs = torch.log_softmax(model(**batch).logits, -1)
  for logprob, mask_bool in zip(logprobs, mask_bools):
    top_logprob, pred_ind = logprob[mask_bool].max(-1)
    top_logprob = top_logprob.sum().item()
    results.append((tokenizer.decode(pred_ind), top_logprob))
  _results: List[str] = []
  logprobs: List[float] = []
  for i in range(0, len(results), max_num_mask):
    choices = results[i:i + max_num_mask]
    best = np.argmax(list(map(itemgetter(1), choices)))
    choice = choices[best]
    _results.append(choice[0])
    logprobs.append(choice[1])
  return _results, logprobs
