from typing import Optional
import torch
from transformers import RagSequenceForGeneration


class MyRagSequenceForGeneration(RagSequenceForGeneration):
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
