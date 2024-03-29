from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
from transformers import RagSequenceForGeneration, AutoModel, AutoConfig, RagRetriever, BatchEncoding
from transformers.models.dpr.modeling_dpr import DPRQuestionEncoderOutput
from distributed_pytorch_retriever import RagPyTorchDistributedRetriever


class RobertaRetriever(nn.Module):
  def __init__(self, model_name):
    super().__init__()

    config = AutoConfig.from_pretrained(model_name)
    self.encoder = AutoModel.from_pretrained(model_name)
    self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                 nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))


  def encode_seq(self, input_ids, mask):
    cls_rep = self.encoder(input_ids, mask)[0][:, 0, :]
    vector = self.project(cls_rep)
    return vector


  def forward(self,
              input_ids: Optional[torch.Tensor] = None,
              attention_mask: Optional[torch.Tensor] = None,
              token_type_ids: Optional[torch.Tensor] = None,
              inputs_embeds: Optional[torch.Tensor] = None,
              output_attentions=None,
              output_hidden_states=None,
              return_dict=None,):
    pooled = self.encode_seq(input_ids, attention_mask)
    if not return_dict:
      return [pooled]
    return DPRQuestionEncoderOutput(pooler_output=pooled, hidden_states=None, attentions=None)  # TODO: replace none


def load_saved(model, path, exact=True):
  try:
    state_dict = torch.load(path)
  except:
    state_dict = torch.load(path, map_location=torch.device('cpu'))

  def filter(x):
    if x.startswith('rag.question_encoder.'):
      return x[21:]
    if x.startswith('module.'):
      return x[7:]
    return x

  if exact:
    state_dict = {filter(k): v for (k, v) in state_dict.items()}
  else:
    state_dict = {filter(k): v for (k, v) in state_dict.items() if filter(k) in model.state_dict()}
  model.load_state_dict(state_dict, strict=False)  # TODO: embeddings.position_ids missing
  return model


class MyRagRetriever(RagRetriever):
  def __call__(
    self,
    question_input_ids: List[List[int]],
    question_hidden_states: np.ndarray,
    question_strings: List[str]=None,
    prefix=None,
    n_docs=None,
    return_tensors=None,
  ) -> BatchEncoding:
    n_docs = n_docs if n_docs is not None else self.n_docs
    prefix = prefix if prefix is not None else self.config.generator.prefix
    retrieved_doc_embeds, doc_ids, docs = self.retrieve(question_hidden_states, n_docs)

    if question_strings is None:
      input_strings = self.question_encoder_tokenizer.batch_decode(question_input_ids, skip_special_tokens=True)
    else:
      input_strings = [s.replace('[MASK]', '<mask>') for s in question_strings]  # TODO: make it more robust to different models
    context_input_ids, context_attention_mask = self.postprocess_docs(
      docs, input_strings, prefix, n_docs, return_tensors=return_tensors
    )

    return BatchEncoding(
      {
        "context_input_ids": context_input_ids,
        "context_attention_mask": context_attention_mask,
        "retrieved_doc_embeds": retrieved_doc_embeds,
        "doc_ids": doc_ids,
      },
      tensor_type=return_tensors,
    )


class MyRagPyTorchDistributedRetriever(RagPyTorchDistributedRetriever):
  def __call__(
    self,
    question_input_ids: List[List[int]],
    question_hidden_states: np.ndarray,
    question_strings: List[str]=None,
    prefix=None,
    n_docs=None,
    return_tensors=None,
  ) -> BatchEncoding:
    n_docs = n_docs if n_docs is not None else self.n_docs
    prefix = prefix if prefix is not None else self.config.generator.prefix
    retrieved_doc_embeds, doc_ids, docs = self.retrieve(question_hidden_states, n_docs)

    if question_strings is None:
      input_strings = self.question_encoder_tokenizer.batch_decode(question_input_ids, skip_special_tokens=True)
    else:
      input_strings = [s.replace('[MASK]', '<mask>') for s in question_strings]  # TODO: make it more robust to different models
    context_input_ids, context_attention_mask = self.postprocess_docs(
      docs, input_strings, prefix, n_docs, return_tensors=return_tensors
    )

    return BatchEncoding(
      {
        "context_input_ids": context_input_ids,
        "context_attention_mask": context_attention_mask,
        "retrieved_doc_embeds": retrieved_doc_embeds,
        "doc_ids": doc_ids,
      },
      tensor_type=return_tensors,
    )


class MemoryBank(nn.Module):
  def __init__(self, bank_size: int, emb_size: int = None):
    super().__init__()
    self.bank_size = bank_size
    self.emb_size = emb_size
    if emb_size is not None:
      self.register_buffer('queue', torch.randn(bank_size, emb_size))
    self.register_buffer('ptr', torch.zeros(1, dtype=torch.long))
    self.register_buffer('is_full', torch.zeros(1, dtype=bool))


  @property
  def is_initialized(self):
    return hasattr(self, 'queue')


  @torch.no_grad()
  def put(self,
          embeddings: torch.FloatTensor):  # (bs, emb_size)
    bs, es = embeddings.size()
    ptr = int(self.ptr)

    if not self.is_initialized:
      self.emb_size = es
      self.register_buffer('queue', torch.randn(self.bank_size, self.emb_size).to(embeddings.device))

    # update queue
    if ptr + bs > self.bank_size:  # reach the end of the queue
      self.queue[ptr:, :] = embeddings[:self.bank_size - ptr]
      self.queue[:ptr + bs - self.bank_size, :] = embeddings[self.bank_size - ptr:]
      self.is_full = self.is_full | True
    else:
      self.queue[ptr:ptr + bs, :] = embeddings

    # move pointer
    self.ptr[0] = (ptr + bs) % self.bank_size
    return


  @torch.no_grad()
  def get(self):
    if not self.is_initialized:
      raise ValueError('queue not initialized')
    if not self.is_full:
      return self.queue[:self.ptr]  # (<bank_size, emb_size)
    return self.queue


class MyRagSequenceForGeneration(RagSequenceForGeneration):
  def set_additional_retrievers(self, retrievers: List):
    self._retrievers = retrievers


  def retrieve_from_multiple(self, *args, **kwargs):
    if not hasattr(self, '_retrievers'):
      return self.retriever(*args, **kwargs)

    all_retrievers = [self.retriever] + self._retrievers
    question_hidden_states = torch.tensor(args[1])
    n_docs = kwargs['n_docs']

    cii_li = []
    cam_li = []
    rde_li = []
    di_li = []
    ds_li = []

    # collect retrieved results from all retrievers
    for retriever in all_retrievers:
      retriever_outputs = retriever(*args, **kwargs)
      cii, cam, rde, di = (
        retriever_outputs['context_input_ids'],  # (batch_size * n_docs, seq_len)
        retriever_outputs['context_attention_mask'],  # (batch_size * n_docs, seq_len)
        retriever_outputs['retrieved_doc_embeds'],  # (batch_size, n_docs, emb_size)
        retriever_outputs['doc_ids']  # (batch_size, n_docs)
      )
      cii_li.append(cii.view(-1, n_docs, cii.size(-1)))
      cam_li.append(cam.view(-1, n_docs, cam.size(-1)))
      rde_li.append(rde)
      di_li.append(di)
      doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), rde.to(question_hidden_states).transpose(1, 2)).squeeze(1)
      ds_li.append(doc_scores)

    # merge results
    cii = torch.cat(cii_li, 1)
    cam = torch.cat(cam_li, 1)
    rde = torch.cat(rde_li, 1)
    di = torch.cat(di_li, 1)
    doc_scores = torch.cat(ds_li, 1)

    # sort by scores and select
    doc_scores, topk_ind = torch.topk(doc_scores, n_docs, dim=1)
    cii = torch.gather(cii, 1, topk_ind.unsqueeze(-1).repeat(1, 1, cii.size(-1))).view(-1, cii.size(-1))
    cam = torch.gather(cam, 1, topk_ind.unsqueeze(-1).repeat(1, 1, cam.size(-1))).view(-1, cam.size(-1))
    rde = torch.gather(rde, 1, topk_ind.unsqueeze(-1).repeat(1, 1, rde.size(-1)))
    di = torch.gather(di, 1, topk_ind)

    return {'context_input_ids': cii, 'context_attention_mask': cam, 'retrieved_doc_embeds': rde, 'doc_ids': di}


  @torch.no_grad()
  def generate(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    context_input_ids=None,
    context_attention_mask=None,
    doc_scores=None,
    do_deduplication=None,  # defaults to True
    num_return_sequences=None,  # defaults to 1
    num_beams=None,  # defaults to 1
    n_docs=None,
    **model_kwargs
  ):
    """
    Implements RAG sequence "thorough" decoding. Read the :meth:`~transformers.PreTrainedModel.generate``
    documentation for more information on how to set other generate input parameters.
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            The sequence used as a prompt for the generation. If :obj:`input_ids` is not passed, then
            :obj:`context_input_ids` has to be provided.
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            `What are attention masks? <../glossary.html#attention-mask>`__
        context_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
            Input IDs post-processed from the retrieved documents and the question encoder input_ids by the
            retriever.
        context_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
            Attention mask post-processed from the retrieved documents and the question encoder :obj:`input_ids` by
            the retriever.
            If the model is not initialized with a ``retriever`` or ``input_ids`` is not given,
            :obj:`context_input_ids` and :obj:`context_attention_mask` have to be provided to the forward pass.
            They are returned by :meth:`~transformers.RagRetriever.__call__`.
        doc_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see :obj:`retrieved_doc_embeds`) and
            :obj:`question_encoder_last_hidden_state`.
            If the model is not initialized with a ``retriever`` or ``input_ids`` is not given, :obj:`doc_scores`
            has to be provided to the forward pass. :obj:`doc_scores` are returned by
            :meth:`~transformers.RagRetriever.__call__`.
        do_deduplication (:obj:`bool`, `optional`):
            Whether or not to deduplicate the generations from different context documents for a given input. Has
            to be set to :obj:`False` if used while training with distributed backend.
        num_return_sequences(:obj:`int`, `optional`, defaults to 1):
            The number of independently computed returned sequences for each element in the batch. Note that this
            is not the value we pass to the ``generator``'s `:func:`~transformers.PreTrainedModel.generate``
            function, where we set ``num_return_sequences`` to :obj:`num_beams`.
        num_beams (:obj:`int`, `optional`, defaults to 1):
            Number of beams for beam search. 1 means no beam search.
        n_docs (:obj:`int`, `optional`, defaults to :obj:`config.n_docs`)
            Number of documents to retrieve and/or number of documents for which to generate an answer.
        kwargs:
            Additional kwargs will be passed to :meth:`~transformers.PreTrainedModel.generate`.
    Return:
        :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`: The generated
        sequences. The second dimension (sequence length) is either equal to :obj:`max_length` or shorter if all
        batches finished early due to the :obj:`eos_token_id`.
    """

    n_docs = n_docs if n_docs is not None else self.config.n_docs
    do_deduplication = do_deduplication if do_deduplication is not None else self.config.do_deduplication
    num_doc_return_sequences = (
      num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
    )
    num_beams = num_beams if num_beams is not None else self.config.num_beams

    assert (
      input_ids is not None or context_input_ids is not None
    ), " At least one of input_ids or context_input_ids must be given"

    if self.retriever is not None and context_input_ids is None:
      question_hidden_states = self.question_encoder(input_ids, attention_mask=attention_mask)[0]
      context_input_ids = self.retriever(
        input_ids,
        question_hidden_states.cpu().detach().to(torch.float32).numpy(),
        prefix=self.generator.config.prefix,
        n_docs=n_docs,
        return_tensors="pt",
      )["context_input_ids"]

      # set to correct device
      context_input_ids = context_input_ids.to(input_ids)

    hypos = []
    logprobs = []
    model_kwargs["num_beams"] = num_beams
    model_kwargs["num_return_sequences"] = num_beams
    model_kwargs["attention_mask"] = None

    batch_size = input_ids.shape[0] if input_ids is not None else context_input_ids.shape[0] // n_docs

    for index in range(batch_size):
      # first, generate beams from documents:
      generator_input_ids = context_input_ids[index * n_docs: (index + 1) * n_docs]  # (n_docs, max_len)

      output_sequences = self.generator.generate(
        generator_input_ids,
        **model_kwargs,
      )  # n_docs * n_beam, tgt_len
      if do_deduplication:
        # do_deduplication, max_output_len
        output_sequences = torch.stack(list({str(k.tolist()): k for k in output_sequences}.values()))

      num_candidates = output_sequences.shape[
        0
      ]  # after deduplication, this number can be less than n_docs*n_beam

      # then, run model forwards to get nll scores:
      if input_ids is not None:
        new_input_ids = input_ids[index: index + 1].repeat(num_candidates, 1)
        outputs = self(new_input_ids, labels=output_sequences, exclude_bos_score=True)
      else:  # input_ids is None, need context_input_ids/mask and doc_scores
        assert (
          context_attention_mask is not None
        ), "Make sure that `context_attention_mask` are passed, if no `input_ids` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function."
        assert (
          doc_scores is not None
        ), "Make sure that `doc_scores` are passed, if no `input_ids` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function."

        individual_input_ids = generator_input_ids.repeat(
          num_candidates, 1
        )  # (num_candidates*n_docs, max_len)

        individual_attention_mask = context_attention_mask[index * n_docs: (index + 1) * n_docs]
        individual_attention_mask = individual_attention_mask.repeat(num_candidates, 1)

        individual_doc_scores = doc_scores[index: (index + 1), :]  # doc_scores.shape = [batch, n_docs]
        individual_doc_scores = individual_doc_scores.repeat(num_candidates, 1)  # [num_candidates, n_docs]

        outputs = self(
          context_input_ids=individual_input_ids,
          context_attention_mask=individual_attention_mask,
          doc_scores=individual_doc_scores,
          labels=output_sequences,
          exclude_bos_score=True,
        )

      lps, top_cand_inds = (-outputs["loss"]).topk(num_doc_return_sequences)

      # add hypothesis
      hypos.append(output_sequences[top_cand_inds])
      logprobs.append(lps)

    logprobs = torch.cat(logprobs, 0)
    return self._cat_and_pad(hypos, pad_token_id=self.config.generator.pad_token_id), logprobs


  @staticmethod
  def load_question_encoder(rag_model, question_encoder_path: str):
    model = RobertaRetriever('roberta-base')
    model = load_saved(model, question_encoder_path, exact=False)
    rag_model.rag.question_encoder = model


  def get_nll(
    self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, exclude_bos_score=False, n_docs=None
  ):
    # shift tokens left
    target = torch.cat(
      [target[:, 1:], target.new(target.shape[0], 1).fill_(self.config.generator.pad_token_id)], 1
    )

    n_docs = n_docs if n_docs is not None else self.config.n_docs

    # bos_token_id is None for T5
    bos_token_id = self.config.bos_token_id or self.config.generator.bos_token_id
    use_bos = bos_token_id is not None and target[:, 0].eq(bos_token_id).all()

    def _mask_pads(ll, smooth_obj):
      pad_mask = target.eq(self.config.generator.pad_token_id)
      if pad_mask.any():
        ll.masked_fill_(pad_mask, 0.0)
        smooth_obj.masked_fill_(pad_mask, 0.0)
      return ll.squeeze(-1), smooth_obj.squeeze(-1)

    # seq_logits dim = (batch*n_docs, tgt_len , #vocabs)
    seq_logprobs = torch.nn.functional.log_softmax(seq_logits, dim=-1).view(
      seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
    )  # batch_size x n_docs x tgt_len x #vocab_size

    # (bs, n_docs) or (bs, bs * n_docs) or (bs, bs * n_docs + memory_bank_size)
    doc_logprobs = torch.nn.functional.log_softmax(doc_scores, dim=1)

    bs = doc_logprobs.size(0)
    if doc_logprobs.size(1) == n_docs:
      pass
    elif doc_logprobs.size(1) >= bs * n_docs:  # contain doc log probs over the whole batch (plus memory bank)
      # (bs, bs, n_docs)
      doc_logprobs = doc_logprobs[:, :bs * n_docs].view(bs, bs, n_docs)  # remove memory bank
      # (bs, n_docs)
      doc_logprobs = torch.masked_select(
        doc_logprobs,
        torch.eye(bs).unsqueeze(-1).bool().to(doc_logprobs.device)).view(bs, n_docs)
    else:
      raise Exception(f'the size of document log prob {doc_logprobs.size()} is unexpected')

    doc_logprobs = doc_logprobs.unsqueeze(-1).unsqueeze(-1)

    # RAG-sequence marginalization
    first_token_scores = seq_logprobs[:, :, :1, :]
    second_token_scores = seq_logprobs[:, :, 1:2, :]
    remainder = seq_logprobs[:, :, 2:, :]
    rag_logprobs = torch.cat([first_token_scores, second_token_scores + doc_logprobs, remainder], dim=2)

    # calculate loss
    target = target.unsqueeze(1).unsqueeze(-1).repeat(1, n_docs, 1, 1)
    assert target.dim() == rag_logprobs.dim()

    ll = rag_logprobs.gather(dim=-1, index=target)
    smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits

    ll, smooth_obj = _mask_pads(ll, smooth_obj)

    # sum over tokens, exclude bos while scoring
    ll = ll[:, :, 1:].sum(2) if exclude_bos_score and use_bos else ll.sum(2)
    smooth_obj = smooth_obj.sum(2)
    ll = ll.logsumexp(1)  # logsumexp over docs
    smooth_obj = smooth_obj.logsumexp(1)

    nll_loss = -ll
    smooth_loss = -smooth_obj

    if reduce_loss:
      nll_loss = nll_loss.sum()
      smooth_loss = smooth_loss.sum()

    eps_i = epsilon / rag_logprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss
