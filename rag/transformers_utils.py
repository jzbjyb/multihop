from typing import List, Tuple
from operator import itemgetter
import numpy as np
import torch
from transformers import BartTokenizer


def shift_tokens_right(input_ids, pad_token_id):
  '''
  Shift input ids one token to the right,
  and wrap the last non pad token (usually <eos> which is also used as config.decoder_start_token_id).
  '''
  prev_output_tokens = input_ids.clone()
  index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
  prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
  prev_output_tokens[:, 1:] = input_ids[:, :-1]
  return prev_output_tokens


def decode_keep_mask(ids: List[int], tokenizer) -> str:
  skip = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
  # add a space before mask (mainly for <s> of roberta/bart)
  return ''.join([' {}'.format(tokenizer.mask_token) if id == tokenizer.mask_token_id else tokenizer.decode([id]) for id in ids if id not in skip])


NEED_PREFIX_TOKENIZER = {BartTokenizer}

def tokenize_answer(text: str, tokenizer):
  if type(tokenizer) in NEED_PREFIX_TOKENIZER:
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text, add_prefix_space=True))
  return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))


def predict_batch(model,
                  tokenizer,
                  questions: List[str],
                  answers: List[str],
                  mask_num_hint: bool=False,
                  max_num_mask: int=1,
                  init_mask_token: str='[MASK]',
                  seq2seq: bool=False,
                  score_answer: bool=False):
  if seq2seq:
    assert score_answer, 'seq2seq is only used for scoring answers in seq2seq format'
  if score_answer:
    max_num_mask = 1
    mask_num_hint = True
  results: List[Tuple[str, float]] = []
  new_questions = []
  new_answers = []
  answers_toks: List[List[int]] = []
  for question, answer in zip(questions, answers):
    # make sure that mask is followed by space (to avoid bugs in BART/RoBERTa tokenizer)
    mask_end = question.find(init_mask_token) + len(init_mask_token)
    if mask_end < len(question) and question[mask_end] != ' ':
      question = question.replace(init_mask_token, init_mask_token + ' ')
    if max_num_mask == 1:
      answer_toks = tokenize_answer(answer, tokenizer)
      answers_toks.append(answer_toks)
      masks = ' '.join([tokenizer.mask_token] * (len(answer_toks) if mask_num_hint else 1))
      new_questions.append(question.replace(init_mask_token, masks))
      new_answers.append(answer)
    else:
      for i in range(max_num_mask):
        masks = ' '.join([tokenizer.mask_token] * (i + 1))
        new_questions.append(question.replace(init_mask_token, masks))
        new_answers.append(answer)
  batch = tokenizer(new_questions, padding=True, return_tensors='pt')
  batch = {k: v.cuda() for k, v in batch.items()}
  with torch.no_grad():
    if seq2seq:
      tgt_ids = tokenizer(new_answers, padding=True, return_tensors='pt')['input_ids'].cuda()
      mask_bools = tgt_ids.ne(tokenizer.pad_token_id)
      decoder_input_ids = shift_tokens_right(tgt_ids, tokenizer.pad_token_id)
      logprobs = torch.log_softmax(model(**batch, decoder_input_ids=decoder_input_ids).logits, -1)  # e.g., BRAR
    else:
      mask_bools = batch['input_ids'].eq(tokenizer.mask_token_id)
      logprobs = torch.log_softmax(model(**batch).logits, -1)
  for i, (logprob, mask_bool) in enumerate(zip(logprobs, mask_bools)):
    if score_answer:
      if seq2seq:
        logprob = logprob[mask_bool]
        pred_ind = tgt_ids[i][mask_bool]
        top_logprob = torch.gather(logprob, 1, pred_ind.unsqueeze(-1)).view(-1)
      else:
        logprob = logprob[mask_bool]
        pred_ind = torch.tensor(answers_toks[i]).to(logprob.device)
        top_logprob = torch.gather(logprob, 1, pred_ind.unsqueeze(-1)).view(-1)
    else:
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
