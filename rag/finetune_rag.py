"""Finetuning script for RAG models. Adapted from examples.seq2seq.finetune.py"""

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.distributed as dist
from pytorch_lightning.accelerators.ddp_accelerator import DDPAccelerator
from pytorch_lightning.cluster_environments import TorchElasticEnvironment
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoTokenizer,
    BartForConditionalGeneration,
    BatchEncoding,
    RagConfig,
    RagSequenceForGeneration,
    RagTokenForGeneration,
    RagTokenizer,
    T5ForConditionalGeneration,
)
from transformers import logging as transformers_logging
from transformers.integrations import is_ray_available


if is_ray_available():
    import ray
    from distributed_ray_retriever import RagRayDistributedRetriever, RayRetriever


from callbacks_rag import (  # noqa: E402 # isort:skipq
    get_checkpoint_callback,
    get_early_stopping_callback,
    Seq2SeqLoggingCallback,
)

from distributed_pytorch_retriever import RagPyTorchDistributedRetriever  # noqa: E402 # isort:skip
from utils_rag import (  # noqa: E402 # isort:skip
    calculate_exact_match,
    flatten_list,
    get_git_info,
    is_rag_model,
    lmap,
    pickle_save,
    save_git_info,
    save_json,
    set_extra_model_params,
    Seq2SeqDataset,
)
from rag_model import MyRagSequenceForGeneration

# need the parent dir module
sys.path.insert(2, str(Path(__file__).resolve().parents[1]))
from lightning_base import BaseTransformer, add_generic_args, generic_train  # noqa


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

transformers_logging.set_verbosity_info()
root_to_mdr = '/home/jzb/node09/exp/multihop_dense_retrieval'


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# In PTL >v1.0, `init_ddp_connection` method in the `LightningModule`
# is no longer used, and is moved into DDPAccelerator instead.
# We override DDPAccelerator to add our custom logic for initializing the
# retriever.
# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/tests/backends/test_accelerator_connector.py


class CustomAccel(DDPAccelerator):
    def __init__(self, trainer=None, **kwargs):
        # Trainer is set later.
        super().__init__(trainer, **kwargs)

    def init_ddp_connection(self, global_rank: int, world_size: int, is_slurm_managing_tasks: bool = True):
        logger.info("Custom init_ddp_connection.")
        module = self.trainer.model
        if self.cluster_environment is None:
            self.cluster_environment = TorchElasticEnvironment()
        self.distributed_port = module.hparams.distributed_port
        os.environ["MASTER_PORT"] = str(self.distributed_port)
        super().init_ddp_connection(global_rank, world_size, is_slurm_managing_tasks)
        if module.is_rag_model:
            if module.distributed_retriever == "pytorch":
                module.model.rag.retriever.init_retrieval(self.distributed_port)
            elif module.distributed_retriever == "ray" and global_rank == 0:
                # For the Ray retriever, only initialize it once when global
                # rank is 0.
                module.model.rag.retriever.init_retrieval()
            module.model.rag.retriever.index.dataset._format_type = None  # TODO: avoid bus error


class GenerativeQAModule(BaseTransformer):
    mode = "generative_qa"
    loss_names = ["loss"]
    metric_names = ["em"]
    val_metric = "em"

    def __init__(self, hparams, **kwargs):
        # when loading from a pytorch lightning checkpoint, hparams are passed as dict
        if isinstance(hparams, dict):
            hparams = AttrDict(hparams)
        if hparams.model_type == "rag_sequence":
            self.model_class = RagSequenceForGeneration
        elif hparams.model_type == "rag_token":
            self.model_class = RagTokenForGeneration
        elif hparams.model_type == "bart":
            self.model_class = BartForConditionalGeneration
        else:
            self.model_class = T5ForConditionalGeneration
        self.is_rag_model = is_rag_model(hparams.model_type)
        self.retrieval_mode = hparams.retrieval_mode
        self.retrieval_hop = hparams.retrieval_hop
        self.use_mdr = hparams.use_mdr
        self.fix_retriever = hparams.fix_retriever
        self.fix_generator = hparams.fix_generator
        self.consistency_loss = hparams.consistency_loss
        self.distance = hparams.distance

        config_class = RagConfig if self.is_rag_model else AutoConfig
        config = config_class.from_pretrained(hparams.model_name_or_path)
        config.max_combined_length = hparams.max_combined_length  # need to be smaller for multihop training
        if hparams.n_docs is not None:
            config.n_docs = hparams.n_docs

        # set retriever parameters
        config.index_name = hparams.index_name or config.index_name
        config.passages_path = hparams.passages_path or config.passages_path
        config.index_path = hparams.index_path or config.index_path
        config.use_dummy_dataset = hparams.use_dummy_dataset

        # set extra_model_params for generator configs and load_model
        extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "attention_dropout", "dropout")
        model2 = None
        if self.is_rag_model:
            if hparams.prefix is not None:
                config.generator.prefix = hparams.prefix
            config.label_smoothing = hparams.label_smoothing
            hparams, config.generator = set_extra_model_params(extra_model_params, hparams, config.generator)
            if hparams.distributed_retriever == "pytorch":
                if self.use_mdr:
                    retriever = RagPyTorchDistributedRetriever.from_pretrained(
                        'facebook/rag-sequence-base', index_name='custom',
                        passages_path=os.path.join(root_to_mdr, 'data/hotpot_dataset/my_knowledge_dataset'),
                        index_path=os.path.join(root_to_mdr, 'data/hotpot_dataset/my_knowledge_dataset_hnsw_index.faiss'))
                else:
                    if self.retrieval_mode == 'no':
                        retriever = RagPyTorchDistributedRetriever.from_pretrained('facebook/rag-sequence-nq', index_name="exact", use_dummy_dataset=True)
                    else:
                        retriever = RagPyTorchDistributedRetriever.from_pretrained('facebook/rag-sequence-base', config=RagConfig.from_pretrained('facebook/rag-sequence-base'))
                        #retriever = RagPyTorchDistributedRetriever.from_pretrained('facebook/rag-sequence-base', index_name='exact', use_dummy_dataset=True)
                        #retriever = RagPyTorchDistributedRetriever.from_pretrained(hparams.model_name_or_path, config=config)
            elif hparams.distributed_retriever == "ray":
                # The Ray retriever needs the handles to the retriever actors.
                retriever = RagRayDistributedRetriever.from_pretrained(
                    hparams.model_name_or_path, hparams.actor_handles, config=config
                )
            retriever.config.max_combined_length = hparams.max_combined_length  # need to be smaller for multihop training
            model = self.model_class.from_pretrained(hparams.model_name_or_path, config=config, retriever=retriever)
            if hparams.model_name_or_path2:
                print('---> load model2 {}'.format(hparams.model_name_or_path2))
                model2 = self.model_class.from_pretrained(hparams.model_name_or_path2, config=config, retriever=retriever)
            if self.use_mdr:
                # load question encoder from MDR
                if hparams.model_name_or_path.startswith('facebook'):  # official model
                    MyRagSequenceForGeneration.load_question_encoder(model, os.path.join(root_to_mdr, 'models/q_encoder.pt'))
                else:
                    MyRagSequenceForGeneration.load_question_encoder(model, os.path.join(hparams.model_name_or_path, 'pytorch_model.bin'))
                # load tokenizer from MDR
                model.retriever.question_encoder_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
            prefix = config.question_encoder.prefix
        else:
            if hparams.prefix is not None:
                config.prefix = hparams.prefix
            hparams, config = set_extra_model_params(extra_model_params, hparams, config)
            model = self.model_class.from_pretrained(hparams.model_name_or_path, config=config)
            prefix = config.prefix


        if self.use_mdr:
            tokenizer = (
                RagTokenizer.from_pretrained('facebook/rag-sequence-nq')  # TODO: avoid bug
                if self.is_rag_model
                else AutoTokenizer.from_pretrained(hparams.model_name_or_path)
            )
            tokenizer.question_encoder = AutoTokenizer.from_pretrained('roberta-base')
        else:
            tokenizer = (
                RagTokenizer.from_pretrained(hparams.model_name_or_path)
                if self.is_rag_model
                else AutoTokenizer.from_pretrained(hparams.model_name_or_path)
            )

        super().__init__(hparams, config=config, tokenizer=tokenizer, model=model)
        self.model2 = model2

        #save_git_info(self.hparams.output_dir)  # TODO: debug
        self.output_dir = Path(self.hparams.output_dir)
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            prefix=prefix or "",
        )
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"

        #self.hparams.git_sha = get_git_info()["repo_sha"]  # TODO: debug
        self.num_workers = hparams.num_workers
        self.distributed_port = self.hparams.distributed_port

        # For single GPU training, init_ddp_connection is not called.
        # So we need to initialize the retrievers here.
        if hparams.gpus <= 1:
            if hparams.distributed_retriever == "ray":
                self.model.retriever.init_retrieval()
            elif hparams.distributed_retriever == "pytorch":
                self.model.retriever.init_retrieval(self.distributed_port)
            self.model.retriever.index.dataset._format_type = None  # TODO: avoid bus error

        self.distributed_retriever = hparams.distributed_retriever

    def forward(self, input_ids=None, **kwargs):
        return self.model(input_ids=input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    @staticmethod
    def convert_to_decoder_ids(ids: torch.LongTensor, mask: torch.LongTensor, model, max_source_length: int, use_mdr: bool=False):
        def truncate_multihop_question(question, max_doc_tokens: int = 64):
            doc_sep = model.config.doc_sep
            title_sep = model.config.title_sep
            questions = question.rsplit(doc_sep, 1)
            if len(questions) <= 1:
                return question
            if use_mdr:
                question = (questions[1], questions[0].split(title_sep, 1)[-1])
            else:
                question = ' '.join(questions[0].split(' ')[:max_doc_tokens]) + doc_sep + questions[1]
            return question
        strings = model.retriever.generator_tokenizer.batch_decode(ids, skip_special_tokens=True)
        strings = [truncate_multihop_question(s) for s in strings]
        ids = model.retriever.question_encoder_tokenizer.batch_encode_plus(
            strings,
            max_length=max_source_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        return ids['input_ids'].to(mask), ids['attention_mask'].to(mask)

    def _multihop_step(self, batch: dict, num_hop: int=1) -> Tuple:
        assert num_hop > 0, 'num_hop should be a positive integer'
        # TODO: only for RAG models
        model = self.model
        config = self.model.config

        n_docs = config.n_docs
        exclude_bos_score = config.exclude_bos_score
        reduce_loss = True
        use_cache = False
        output_attentions = config.output_attentions
        output_hidden_states = config.output_hidden_states
        output_retrieved = config.output_retrieved
        past_key_values = None
        epsilon = config.label_smoothing
        encoder_outputs = None

        source_ids, source_mask, target_ids = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
        decoder_input_ids = target_ids
        decoder_input_ids_il = decoder_input_ids.repeat_interleave(n_docs, dim=0)
        lm_labels = decoder_input_ids

        # recursive retrieval
        prev_doc_scores = None
        loss = 0
        for nh in range(num_hop):
            question_enc_outputs = model.question_encoder(source_ids, attention_mask=source_mask, return_dict=True)
            question_encoder_last_hidden_state = question_enc_outputs[0]  # hidden states of question encoder
            if self.fix_retriever:
                question_encoder_last_hidden_state = question_encoder_last_hidden_state.detach()

            retriever_outputs = model.retriever(
                source_ids,
                question_encoder_last_hidden_state.cpu().detach().to(torch.float32).numpy(),
                prefix=model.generator.config.prefix,
                n_docs=n_docs,
                return_tensors='pt',
            )
            context_input_ids, context_attention_mask, retrieved_doc_embeds, retrieved_doc_ids = (
                retriever_outputs['context_input_ids'],
                retriever_outputs['context_attention_mask'],
                retriever_outputs['retrieved_doc_embeds'],
                retriever_outputs['doc_ids'],
            )

            retrieved_doc_embeds = retrieved_doc_embeds.to(question_encoder_last_hidden_state)
            context_input_ids = context_input_ids.to(source_ids)
            context_attention_mask = context_attention_mask.to(source_ids)
            doc_scores = torch.bmm(question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)).squeeze(1)

            if prev_doc_scores is None:
                prev_doc_scores = doc_scores
            else:
                seq_len = context_input_ids.size(1)

                # log prob over previous docs
                prev_doc_scores = F.log_softmax(prev_doc_scores, dim=1).view(-1).unsqueeze(-1)  # SHAPE: (batch_size * ndoc, 1)
                # log prob over current docs
                prev_doc_scores = prev_doc_scores + F.log_softmax(doc_scores, dim=1)  # SHAPE: (batch_size * ndoc, ndoc)
                prev_doc_scores = prev_doc_scores.view(-1, n_docs * n_docs)  # SHAPE: (batch_size, ndoc * ndoc)
                prev_doc_scores, topk_ind = torch.topk(prev_doc_scores, n_docs, dim=1)  # SHAPE: (batch_size, ndoc)
                topk_ind = topk_ind.unsqueeze(-1).expand(-1, -1, seq_len)  # SHAPE: (batch_size, ndoc, seq_len)

                # beam search
                context_input_ids = context_input_ids.view(-1, n_docs * n_docs, seq_len)  # SHAPE: (batch_size, ndoc * ndoc, seq_len)
                context_attention_mask = context_attention_mask.view(-1, n_docs * n_docs, seq_len)  # SHAPE: (batch_size, ndoc * ndoc, seq_len)
                context_input_ids = torch.gather(context_input_ids, 1, topk_ind)  # SHAPE: (batch_size, ndoc, seq_len)
                context_attention_mask = torch.gather(context_attention_mask, 1, topk_ind)  # SHAPE: (batch_size, ndoc, seq_len)
                context_input_ids = context_input_ids.view(-1, seq_len)  # SHAPE: (batch_size * ndoc, seq_len)
                context_attention_mask = context_attention_mask.view(-1, seq_len)  # SHAPE: (batch_size * ndoc, seq_len)

            # "query" for the next iteration
            # TODO: documents too long might truncate questions
            if nh != num_hop - 1:
                source_ids, source_mask = self.convert_to_decoder_ids(
                    context_input_ids, context_attention_mask, self.model, self.hparams.max_source_length, use_mdr=args.use_mdr)

            # loss for the current hop
            gen_outputs = model.generator(
                input_ids=context_input_ids,
                attention_mask=context_attention_mask,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids_il,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_dict=True,
            )
            if self.fix_generator:
                gen_outputs.logits = gen_outputs.logits.detach()

            loss += model.get_nll(
                gen_outputs.logits,
                prev_doc_scores,
                decoder_input_ids,
                reduce_loss=reduce_loss,
                epsilon=epsilon,
                exclude_bos_score=exclude_bos_score,
                n_docs=n_docs,
            )
        return (loss,)

    def compute_consistency_loss(self, logits1, logits2, target, use_consist):
        pad_token_id = self.model.config.generator.pad_token_id
        target = torch.cat([target[:, 1:], target.new(target.shape[0], 1).fill_(pad_token_id)], 1)
        pad_mask = target.eq(pad_token_id)
        if self.distance == 'jsd':
            p1 = F.softmax(logits1, -1)
            p2 = F.softmax(logits2, -1)
            log_avg_dist = torch.log(torch.clamp(p1 / 2 + p2 / 2, min=1e-10))
            kl1 = F.kl_div(log_avg_dist, p1, reduction='none', log_target=False).sum(-1)
            kl2 = F.kl_div(log_avg_dist, p2, reduction='none', log_target=False).sum(-1)
            jsd = kl1 / 2 + kl2 / 2
            if pad_mask.any():
                jsd.masked_fill_(pad_mask, 0.0)
            loss = jsd.sum(-1)
        elif self.distance == 'kl':
            p2 = F.softmax(logits2, -1)
            lp1 = F.log_softmax(logits1, -1)
            kl = F.kl_div(lp1, p2, reduction='none', log_target=False).sum(-1)
            if pad_mask.any():
                kl.masked_fill_(pad_mask, 0.0)
            loss = kl.sum(-1)
        else:
            raise NotImplementedError
        loss = (loss * use_consist.float()).sum()
        return loss

    def _step(self, batch: dict, use_retrieval: bool=True) -> Tuple:
        if use_retrieval:
            return self._multihop_step(batch=batch, num_hop=self.retrieval_hop)

        source_ids, source_mask, target_ids = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
        source_ids_fd, source_mask_fd = batch["input_ids_for_decoder"], batch["attention_mask_for_decoder"]

        rag_kwargs = {}
        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(target_ids)
            lm_labels = target_ids
        elif isinstance(self.model, BartForConditionalGeneration):
            decoder_input_ids = target_ids[:, :-1].contiguous()
            lm_labels = target_ids[:, 1:].clone()
        else:
            assert self.is_rag_model
            generator = self.model.rag.generator
            if isinstance(generator, T5ForConditionalGeneration):
                decoder_start_token_id = generator.config.decoder_start_token_id
                decoder_input_ids = (
                    torch.cat(
                        [torch.Tensor([[decoder_start_token_id]] * target_ids.shape[0]).to(target_ids), target_ids],
                        dim=1,
                    )
                    if target_ids.shape[0] < self.target_lens["train"]
                    else generator._shift_right(target_ids)
                )
            elif isinstance(generator, BartForConditionalGeneration):
                decoder_input_ids = target_ids
            lm_labels = decoder_input_ids
            rag_kwargs["reduce_loss"] = True

        assert decoder_input_ids is not None

        consist_loss = 0
        if self.is_rag_model and not use_retrieval:
            outputs = self(
                context_input_ids=source_ids_fd,
                context_attention_mask=source_mask_fd,
                doc_scores=source_ids.new_zeros(source_ids.size(0), 1).float(),
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
                labels=lm_labels,
                n_docs=1,
                **rag_kwargs,
            )
            if 'input_ids2' in batch:
                source_ids2, source_mask2 = batch["input_ids2"], batch["attention_mask2"]
                source_ids_fd2, source_mask_fd2 = batch["input_ids_for_decoder2"], batch["attention_mask_for_decoder2"]
                if self.model2 is not None:
                    fw_func = self.model2
                else:
                    fw_func = self.model
                outputs2 = fw_func(
                    input_ids=None,
                    context_input_ids=source_ids_fd2,
                    context_attention_mask=source_mask_fd2,
                    doc_scores=source_ids2.new_zeros(source_ids2.size(0), 1).float(),
                    decoder_input_ids=decoder_input_ids,
                    use_cache=False,
                    labels=lm_labels,
                    n_docs=1,
                    **rag_kwargs,
                )
                if self.consistency_loss != 'no':
                    consist_loss = self.compute_consistency_loss(
                        outputs['logits'], outputs2['logits'].detach(), decoder_input_ids, batch['use_consist'])
        else:
            outputs = self(
                source_ids,
                attention_mask=source_mask,
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
                labels=lm_labels,
                **rag_kwargs,
            )

        if self.consistency_loss == 'combine':
            loss = outputs["loss"] + consist_loss
        elif self.consistency_loss == 'only':
            loss = consist_loss
        else:
            loss = outputs["loss"]

        return (loss,)

    @property
    def pad(self) -> int:
        raise NotImplementedError("pad not implemented")

    def training_step(self, batch, batch_idx) -> Dict:
        if self.retrieval_mode == 'ret':
            loss_tensors = self._step(batch, use_retrieval=True)
        elif self.retrieval_mode == 'no':
            loss_tensors = self._step(batch, use_retrieval=False)
        elif self.retrieval_mode == 'combine':
            loss_tensors1 = self._step(batch, use_retrieval=True)
            loss_tensors2 = self._step(batch, use_retrieval=False)
            loss_tensors = (loss_tensors1[0] + loss_tensors2[0],)
        else:
            raise NotImplementedError

        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # tokens per batch
        tgt_pad_token_id = (
            self.tokenizer.generator.pad_token_id
            if isinstance(self.tokenizer, RagTokenizer)
            else self.tokenizer.pad_token_id
        )
        src_pad_token_id = (
            self.tokenizer.question_encoder.pad_token_id
            if isinstance(self.tokenizer, RagTokenizer)
            else self.tokenizer.pad_token_id
        )
        logs["tpb"] = (
            batch["input_ids"].ne(src_pad_token_id).sum() + batch["decoder_input_ids"].ne(tgt_pad_token_id).sum()
        )

        return {"loss": loss_tensors[0], "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict:
        if self.retrieval_mode in {'ret', 'combine'}:
            return self._generative_step(batch, use_retrieval=True)
        if self.retrieval_mode == 'no':
            return self._generative_step(batch, use_retrieval=True)  # TODO: debug
        raise NotImplementedError

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        gen_metrics = {
            k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]
        }
        metrics_tensor: torch.FloatTensor = torch.tensor(gen_metrics[self.val_metric]).type_as(loss)
        gen_metrics.update({k: v.item() for k, v in losses.items()})

        # fix for https://github.com/PyTorchLightning/pytorch-lightning/issues/2424
        if dist.is_initialized():
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            metrics_tensor = metrics_tensor / dist.get_world_size()
            gen_metrics.update({self.val_metric: metrics_tensor.item()})

        losses.update(gen_metrics)
        metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        metrics["step_count"] = self.step_count
        self.save_metrics(metrics, prefix)  # writes to self.metrics_save_path
        preds = flatten_list([x["preds"] for x in outputs])
        return {"log": metrics, "preds": preds, f"{prefix}_loss": loss, f"{prefix}_{self.val_metric}": metrics_tensor}

    def save_metrics(self, latest_metrics, type_path) -> None:
        self.metrics[type_path].append(latest_metrics)
        save_json(self.metrics, self.metrics_save_path)

    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_exact_match(preds, target)

    def _generative_step(self, batch: dict, use_retrieval: bool=True) -> dict:
        start_time = time.time()
        batch = BatchEncoding(batch).to(device=self.model.device)
        if use_retrieval:
            generated_ids = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                do_deduplication=False,  # rag specific parameter
                use_cache=True,
                min_length=1,
                max_length=self.target_lens["val"],
            )
        else:
            generated_ids = self.model.generate(
                context_input_ids=batch["input_ids_for_decoder"],
                context_attention_mask=batch["attention_mask_for_decoder"],
                doc_scores=batch["input_ids_for_decoder"].new_zeros(batch["input_ids_for_decoder"].size(0), 1).float(),
                do_deduplication=False,  # rag specific parameter
                use_cache=True,
                min_length=1,
                max_length=self.target_lens["val"],
            )

        gen_time = (time.time() - start_time) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["decoder_input_ids"])
        loss_tensors = self._step(batch, use_retrieval=use_retrieval)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        gen_metrics: Dict = self.calc_generative_metrics(preds, target)

        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target, **gen_metrics)
        return base_metrics

    def test_step(self, batch, batch_idx):
        if self.retrieval_mode in {'ret', 'combine'}:
            return self._generative_step(batch, use_retrieval=True)
        if self.retrieval_mode == 'no':
            return self._generative_step(batch, use_retrieval=True)  # TODO: debug
        raise NotImplementedError

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_dataset(self, type_path) -> Seq2SeqDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset = Seq2SeqDataset(
            self.tokenizer,
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("checkpoint{}".format(self.step_count))
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)
        parser.add_argument(
            "--max_source_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=25,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=25,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=25,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument(
            "--prefix",
            type=str,
            default=None,
            help="Prefix added at the beginning of each text, typically used with T5-based models.",
        )
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )
        parser.add_argument(
            "--distributed-port", type=int, default=-1, required=False, help="Port number for distributed training."
        )
        parser.add_argument(
            "--model_type",
            choices=["rag_sequence", "rag_token", "bart", "t5"],
            type=str,
            help="RAG model type: sequence or token, if none specified, the type is inferred from the model_name_or_path",
        )
        return parser

    @staticmethod
    def add_retriever_specific_args(parser):
        parser.add_argument(
            "--index_name",
            type=str,
            default=None,
            help="Name of the index to use: 'hf' for a canonical dataset from the datasets library (default), 'custom' for a local index, or 'legacy' for the orignal one)",
        )
        parser.add_argument(
            "--passages_path",
            type=str,
            default=None,
            help="Path to the dataset of passages for custom index. More info about custom indexes in the RagRetriever documentation as well as in `examples/rag/use_own_knowledge_dataset.py`",
        )
        parser.add_argument(
            "--index_path",
            type=str,
            default=None,
            help="Path to the faiss index for custom index. More info about custom indexes in the RagRetriever documentation as well as in `examples/rag/use_own_knowledge_dataset.py`",
        )
        parser.add_argument(
            "--distributed_retriever",
            choices=["ray", "pytorch"],
            type=str,
            default="pytorch",
            help="What implementation to use for distributed retriever? If "
            "pytorch is selected, the index is loaded on training "
            "worker 0, and torch.distributed is used to handle "
            "communication between training worker 0, and the other "
            "training workers. If ray is selected, the Ray library is "
            "used to create load the index on separate processes, "
            "and Ray handles the communication between the training "
            "workers and the retrieval actors.",
        )
        parser.add_argument(
            "--use_dummy_dataset",
            type=bool,
            default=False,
            help="Whether to use the dummy version of the dataset index. More info about custom indexes in the RagRetriever documentation as well as in `examples/rag/use_own_knowledge_dataset.py`",
        )
        parser.add_argument(
            '--retrieval_mode',
            type=str,
            choices=['ret', 'no', 'combine'],
            default='ret'
        )
        parser.add_argument(
            '--retrieval_hop',
            type=int,
            default=1
        )
        parser.add_argument(
            '--n_docs',
            type=int,
            default=None
        )
        parser.add_argument(
            '--max_combined_length',
            type=int,
            default=300
        )
        parser.add_argument('--use_mdr', action='store_true')
        parser.add_argument('--fix_retriever', action='store_true')
        parser.add_argument('--fix_generator', action='store_true')
        parser.add_argument('--consistency_loss', type=str, choices=['no', 'combine', 'only'], default='no')
        parser.add_argument('--distance', type=str, choices=['jsd', 'kl'], default='jsd')
        return parser

    @staticmethod
    def add_ray_specific_args(parser):
        # Ray cluster address.
        parser.add_argument(
            "--ray-address",
            default="auto",
            type=str,
            help="The address of the Ray cluster to connect to. If not "
            "specified, Ray will attempt to automatically detect the "
            "cluster. Has no effect if pytorch is used as the distributed "
            "retriever.",
        )
        parser.add_argument(
            "--num_retrieval_workers",
            type=int,
            default=1,
            help="The number of retrieval actors to use when Ray is selected"
            "for the distributed retriever. Has no effect when "
            "distributed_retriever is set to pytorch.",
        )
        return parser


def main(args=None, model=None) -> GenerativeQAModule:
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GenerativeQAModule.add_model_specific_args(parser, os.getcwd())
    parser = GenerativeQAModule.add_retriever_specific_args(parser)

    args = args or parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    named_actors = []
    if args.distributed_retriever == "ray" and args.gpus > 1:
        if not is_ray_available():
            raise RuntimeError("Please install Ray to use the Ray " "distributed retriever.")
        # Connect to an existing Ray cluster.
        try:
            ray.init(address=args.ray_address)
        except (ConnectionError, ValueError):
            logger.warning(
                "Connection to Ray cluster failed. Make sure a Ray"
                "cluster is running by either using Ray's cluster "
                "launcher (`ray up`) or by manually starting Ray on "
                "each node via `ray start --head` for the head node "
                "and `ray start --address='<ip address>:6379'` for "
                "additional nodes. See "
                "https://docs.ray.io/en/master/cluster/index.html "
                "for more info."
            )
            raise

        # Create Ray actors only for rank 0.
        if ("LOCAL_RANK" not in os.environ or os.environ["LOCAL_RANK"] == 0) and (
            "NODE_RANK" not in os.environ or os.environ["NODE_RANK"] == 0
        ):
            remote_cls = ray.remote(RayRetriever)
            named_actors = [
                remote_cls.options(name="retrieval_worker_{}".format(i)).remote()
                for i in range(args.num_retrieval_workers)
            ]
        else:
            logger.info(
                "Getting named actors for NODE_RANK {}, LOCAL_RANK {}".format(
                    os.environ["NODE_RANK"], os.environ["LOCAL_RANK"]
                )
            )
            named_actors = [ray.get_actor("retrieval_worker_{}".format(i)) for i in range(args.num_retrieval_workers)]
    args.actor_handles = named_actors
    assert args.actor_handles == named_actors

    if model is None:
        model: GenerativeQAModule = GenerativeQAModule(args)

    dataset = Path(args.data_dir).name
    if (
        args.logger_name == "default"
        or args.fast_dev_run
        or str(args.output_dir).startswith("/tmp")
        or str(args.output_dir).startswith("/var")
    ):
        training_logger = True  # don't pollute wandb logs unnecessarily
    elif args.logger_name == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        project = os.environ.get("WANDB_PROJECT", dataset)
        training_logger = WandbLogger(name=model.output_dir.name, project=project)

    elif args.logger_name == "wandb_shared":
        from pytorch_lightning.loggers import WandbLogger

        training_logger = WandbLogger(name=model.output_dir.name, project=f"hf_{dataset}")

    es_callback = (
        get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
        if args.early_stopping_patience >= 0
        else False
    )

    trainer: pl.Trainer = generic_train(
        model,
        args,
        logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback=get_checkpoint_callback(args.output_dir, model.val_metric),
        early_stopping_callback=es_callback,
        logger=training_logger,
        accelerator=CustomAccel() if args.gpus > 1 else None,
        profiler=pl.profiler.AdvancedProfiler() if args.profile else None,
    )
    pickle_save(model.hparams, model.output_dir / "hparams.pkl")

    if not args.do_predict:
        return model

    # test() without a model tests using the best checkpoint automatically
    trainer.test()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GenerativeQAModule.add_model_specific_args(parser, os.getcwd())
    parser = GenerativeQAModule.add_retriever_specific_args(parser)
    parser = GenerativeQAModule.add_ray_specific_args(parser)

    # Pytorch Lightning Profiler
    parser.add_argument(
        "--profile",
        action="store_true",
        help="If True, use pytorch_lightning.profiler.AdvancedProfiler to profile the Trainer.",
    )

    args = parser.parse_args()

    main(args)
