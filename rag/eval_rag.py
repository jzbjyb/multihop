""" Evaluation script for RAG models."""

from typing import List, Callable, Dict

import argparse
import ast
import logging
import os
import sys
import time

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from operator import itemgetter

from transformers import BartForConditionalGeneration, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration, AutoTokenizer
from transformers import logging as transformers_logging


sys.path.append(os.path.join(os.getcwd()))  # noqa: E402 # isort:skip
from utils_rag import exact_match_score, f1_score, truncate_context_with_question  # noqa: E402 # isort:skip
from rag_model import MyRagSequenceForGeneration, MyRagRetriever
from finetune_rag import GenerativeQAModule, root_to_mdr
from transformers_utils import predict_batch, decode_keep_mask
from dataset import Break, PseudoBreak


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

transformers_logging.set_verbosity_info()


def infer_model_type(model_name_or_path):
    if "token" in model_name_or_path:
        return "rag_token"
    if "sequence" in model_name_or_path:
        return "rag_sequence"
    if "bart" in model_name_or_path:
        return "bart"
    return None


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def get_scores(args, preds_path, gold_data_path, question_data_path=None):
    hypos = [line.rstrip('\n').split('\t')[0] for line in open(preds_path, "r").readlines()]
    answers = []

    if args is None or args.gold_data_mode == 'ans_tab':
        answers = [line.rstrip('\n').split('\t') for line in open(gold_data_path, 'r')]
    elif args.gold_data_mode == "ans":
        references = [line.strip() for line in open(gold_data_path, "r").readlines()]
        answers = [[reference] for reference in references]
    elif args.gold_data_mode == "qa":
        data = pd.read_csv(gold_data_path, sep="\t", header=None)
        for answer_list in data[1]:
            ground_truths = ast.literal_eval(answer_list)
            answers.append(ground_truths)

    if question_data_path:
        questions = [line.strip() for line in open(question_data_path, 'r').readlines()]
    else:
        questions = list(range(len(answers)))

    q2em = defaultdict(lambda: 0)
    q2f1 = defaultdict(lambda: 0)
    for prediction, ground_truths, question in zip(hypos, answers, questions):
        em = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        q2em[question] = max(q2em[question], em)
        q2f1[question] = max(q2f1[question], f1)

    em = 100.0 * sum(q2em.values()) / len(q2em)
    f1 = 100.0 * sum(q2f1.values()) / len(q2f1)

    logger.info(f"F1: {f1:.2f}")
    logger.info(f"EM: {em:.2f}")


def get_precision_at_k(args, preds_path, gold_data_path):
    k = args.k
    hypos = [line.rstrip('\n').split('\t')[0] for line in open(preds_path, "r").readlines()]
    references = [line.strip() for line in open(gold_data_path, "r").readlines()]

    em = total = 0
    for hypo, reference in zip(hypos, references):
        hypo_provenance = set(hypo.split("\t")[:k])
        ref_provenance = set(reference.split("\t"))
        total += 1
        em += len(hypo_provenance & ref_provenance) / k

    em = 100.0 * em / total
    logger.info(f"Precision@{k}: {em: .2f}")


def evaluate_batch_retrieval(args, rag_model, questions, **kwargs):
    def strip_title(title):
        if title.startswith('"'):
            title = title[1:]
        if title.endswith('"'):
            title = title[:-1]
        return title

    retriever_input_ids = rag_model.retriever.question_encoder_tokenizer.batch_encode_plus(
        questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_source_length,
    )["input_ids"].to(args.device)

    question_enc_outputs = rag_model.rag.question_encoder(retriever_input_ids)
    question_enc_pool_output = question_enc_outputs[0]

    result = rag_model.retriever(
        retriever_input_ids,
        question_enc_pool_output.cpu().detach().to(torch.float32).numpy(),
        prefix=rag_model.rag.generator.config.prefix,
        n_docs=rag_model.config.n_docs,
        return_tensors="pt",
    )
    all_docs = rag_model.retriever.index.get_doc_dicts(result.doc_ids)
    provenance_strings = []
    for docs in all_docs:
        provenance = [strip_title(title) for title in docs["title"]]
        provenance_strings.append("\t".join(provenance))
    return provenance_strings


def evaluate_batch_retrieval_all(args, rag_model, questions, **kwargs):
    n_docs = args.n_docs
    with torch.no_grad():
        inputs_dict = rag_model.retriever.question_encoder_tokenizer.batch_encode_plus(
            questions, return_tensors='pt', padding=True, truncation=True, max_length=args.max_source_length)
        input_ids = inputs_dict['input_ids'].to(args.device)
        attention_mask = inputs_dict['attention_mask'].to(args.device)
        question_hidden_states = rag_model.question_encoder(input_ids, attention_mask=attention_mask)[0]
        doc_ids = rag_model.retriever(
            input_ids,
            question_hidden_states.cpu().detach().to(torch.float32).numpy(),
            prefix=rag_model.generator.config.prefix,
            n_docs=n_docs,
            return_tensors='pt',
        ).doc_ids
        all_docs = rag_model.retriever.index.get_doc_dicts(doc_ids)
        doc_ids = doc_ids.cpu().numpy()
        all_docs = ['\t'.join(map(lambda x: '{} || {} || {}'.format(x[0], x[1], x[2]), zip(doc_ids[i], docs['title'], docs['text']))) for i, docs in enumerate(all_docs)]
        scores = [0] * len(all_docs)
        return all_docs, scores, None, None, None


def evaluate_batch_e2e_multihop_retrieval(args, rag_model, questions, **kwargs):
    if type(rag_model.retriever) is MyRagRetriever and args.retrieval_hop > 1:
        raise Exception('MyRagRetriever is only used for single-hop retrieval')
    use_eval_target = 'eval_target' in kwargs and bool(kwargs['eval_target'])

    n_docs = rag_model.config.n_docs
    do_deduplication = rag_model.config.do_deduplication
    num_doc_return_sequences = rag_model.config.num_return_sequences
    num_beams = args.num_beams
    questions = [q.split('\t')[0] for q in questions]
    with torch.no_grad():
        inputs_dict = rag_model.retriever.question_encoder_tokenizer.batch_encode_plus(
            questions, return_tensors='pt', padding=True, truncation=True, max_length=args.max_source_length)
        input_ids = inputs_dict['input_ids'].to(args.device)
        attention_mask = inputs_dict['attention_mask'].to(args.device)
        prev_doc_scores = None
        prev_doc_ids = None
        retrieved_docs = []
        retrieved_doc_ids = []

        # retrieve
        for nh in range(args.retrieval_hop):
            question_hidden_states = rag_model.question_encoder(input_ids, attention_mask=attention_mask)[0]
            add_kwargs = {'question_strings': questions} if type(rag_model.retriever) is MyRagRetriever else {}
            retriever_outputs = rag_model.retrieve_from_multiple(
                input_ids,
                question_hidden_states.cpu().detach().to(torch.float32).numpy(),
                prefix=rag_model.generator.config.prefix,
                n_docs=n_docs,
                return_tensors='pt',
                **add_kwargs
            )
            context_input_ids, context_attention_mask, retrieved_doc_embeds, doc_ids = (
                retriever_outputs['context_input_ids'],
                retriever_outputs['context_attention_mask'],
                retriever_outputs['retrieved_doc_embeds'],
                retriever_outputs['doc_ids']
            )
            all_docs = rag_model.retriever.index.get_doc_dicts(doc_ids)
            doc_ids = doc_ids.to(input_ids)
            retrieved_doc_embeds = retrieved_doc_embeds.to(question_hidden_states)
            context_input_ids = context_input_ids.to(input_ids)
            context_attention_mask = context_attention_mask.to(input_ids)
            doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)).squeeze(1)
            if prev_doc_scores is None:
                prev_doc_scores = doc_scores
                doc_ids = doc_ids.view(-1, nh + 1)
            else:
                seq_len = context_input_ids.size(1)

                # log prob over previous docs
                prev_doc_scores = F.log_softmax(prev_doc_scores, dim=1).view(-1).unsqueeze(-1)  # SHAPE: (batch_size * ndoc, 1)
                # log prob over current docs
                prev_doc_scores = prev_doc_scores + F.log_softmax(doc_scores, dim=1)  # SHAPE: (batch_size * ndoc, ndoc)
                prev_doc_scores = prev_doc_scores.view(-1, n_docs * n_docs)  # SHAPE: (batch_size, ndoc * ndoc)
                prev_doc_scores, topk_ind = torch.topk(prev_doc_scores, n_docs, dim=1)  # SHAPE: (batch_size, ndoc)
                _topk_ind = topk_ind
                topk_ind = topk_ind.unsqueeze(-1).expand(-1, -1, seq_len)  # SHAPE: (batch_size, ndoc, seq_len)

                # beam search
                context_input_ids = context_input_ids.view(-1, n_docs * n_docs, seq_len)  # SHAPE: (batch_size, ndoc * ndoc, seq_len)
                context_attention_mask = context_attention_mask.view(-1, n_docs * n_docs, seq_len)  # SHAPE: (batch_size, ndoc * ndoc, seq_len)
                doc_ids = doc_ids.view(-1, n_docs * n_docs).unsqueeze(-1)  # SHAPE: (batch_size, ndoc * ndoc, 1)
                doc_ids = torch.cat([prev_doc_ids.view(-1, n_docs, prev_doc_ids.size(-1)).repeat_interleave(n_docs, dim=1), doc_ids], -1)  # SHAPE: (batch_size, ndoc * ndoc, nh + 1)
                context_input_ids = torch.gather(context_input_ids, 1, topk_ind)  # SHAPE: (batch_size, ndoc, seq_len)
                context_attention_mask = torch.gather(context_attention_mask, 1, topk_ind)  # SHAPE: (batch_size, ndoc, seq_len)
                doc_ids = torch.gather(doc_ids, 1, _topk_ind.unsqueeze(-1).expand(-1, -1, nh + 1))  # SHAPE: (batch_size, ndoc)
                context_input_ids = context_input_ids.view(-1, seq_len)  # SHAPE: (batch_size * ndoc, seq_len)
                context_attention_mask = context_attention_mask.view(-1, seq_len)  # SHAPE: (batch_size * ndoc, seq_len)
                doc_ids = doc_ids.view(-1, nh + 1)  # SHAPE: (batch_size * ndoc, nh + 1)

            prev_doc_ids = doc_ids
            strings = rag_model.retriever.generator_tokenizer.batch_decode(context_input_ids, skip_special_tokens=True)
            retrieved_docs.append([strings[i * n_docs:i * n_docs + n_docs] for i in range(len(strings) // n_docs)])
            retrieved_doc_ids.append([doc_ids[i * n_docs:i * n_docs + n_docs].cpu().numpy().tolist() for i in range(len(strings) // n_docs)])

            if nh != args.retrieval_hop - 1:
                input_ids, attention_mask = GenerativeQAModule.convert_to_decoder_ids(
                    context_input_ids, context_attention_mask, rag_model, max_source_length=args.max_source_length, use_mdr=args.use_mdr)

        # generate
        hypos = []
        logprobs = []
        batch_size = context_input_ids.shape[0] // n_docs

        for index in range(batch_size):
            # first, generate beams from documents:
            generator_input_ids = context_input_ids[index * n_docs : (index + 1) * n_docs]  # (n_docs, max_len)
            if args.generation_method == 'generator':
                if use_eval_target:
                    tokenizer = rag_model.retriever.generator_tokenizer
                    output_sequences: List[str] = kwargs['eval_target'][index]
                    num_doc_return_sequences = len(output_sequences)
                    output_sequences = tokenizer.batch_encode_plus(
                        output_sequences,
                        max_length=rag_model.config.max_combined_length,
                        return_tensors='pt',
                        padding='max_length',
                        truncation=True)['input_ids'].to(generator_input_ids.device)
                else:
                    output_sequences = rag_model.generator.generate(
                        generator_input_ids,
                        attention_mask=None,
                        num_beams=num_beams,
                        num_return_sequences=num_doc_return_sequences,
                        min_length=args.min_length,
                        max_length=args.max_length,
                    )  # n_docs * n_beam, tgt_len
                    if do_deduplication:
                        # do_deduplication, max_output_len
                        output_sequences = torch.stack(list({str(k.tolist()): k for k in output_sequences}.values()))
                num_candidates = output_sequences.shape[0]  # after deduplication, this number can be less than n_docs*n_beam
                individual_input_ids = generator_input_ids.repeat(num_candidates, 1)
                individual_attention_mask = context_attention_mask[index * n_docs : (index + 1) * n_docs]
                individual_attention_mask = individual_attention_mask.repeat(num_candidates, 1)
                individual_doc_scores = prev_doc_scores[index : (index + 1), :]  # doc_scores.shape = [batch, n_docs]
                individual_doc_scores = individual_doc_scores.repeat(num_candidates, 1)  # [num_candidates, n_docs]
                outputs = rag_model(
                    context_input_ids=individual_input_ids,
                    context_attention_mask=individual_attention_mask,
                    doc_scores=individual_doc_scores,
                    labels=output_sequences,
                    exclude_bos_score=True,
                )
                if use_eval_target:  # don't sort
                    lps = -outputs['loss']
                    # add hypothesis
                    hypos.append(output_sequences)
                    logprobs.append(lps)
                else:
                    lps, top_cand_inds = (-outputs['loss']).topk(num_doc_return_sequences)
                    # add hypothesis
                    hypos.append(output_sequences[top_cand_inds])
                    logprobs.append(lps)
            elif args.generation_method == 'mask':
                tokenizer = rag_model.retriever.generator_tokenizer
                context_with_question: List[str] = []
                for gii in generator_input_ids:
                    context_with_question.append(decode_keep_mask(gii.tolist(), tokenizer))
                preds, lps = predict_batch(
                    rag_model.generator, tokenizer, context_with_question, [''] * len(questions),
                    mask_num_hint=False, max_num_mask=5, init_mask_token='<mask>')
                best = np.argmax(lps)
                # TODO: add support for num_doc_return_sequences
                hypos.append(tokenizer([preds[best]], return_tensors='pt')['input_ids'])
                logprobs.append(torch.tensor([lps[best]]))
        outputs = rag_model._cat_and_pad(hypos, pad_token_id=rag_model.config.generator.pad_token_id)
        logprobs = torch.cat(logprobs, 0)
        answers = rag_model.retriever.generator_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if args.print_predictions:
            for q, a in zip(questions, answers):
                logger.info("Q: {} - A: {}".format(q, a))
        return answers, logprobs.cpu().numpy(), retrieved_docs, retrieved_doc_ids, all_docs


def evaluate_batch_e2e(args, rag_model, questions, **kwargs):
    with torch.no_grad():
        inputs_dict = rag_model.retriever.question_encoder_tokenizer.batch_encode_plus(
            questions, return_tensors="pt", padding=True, truncation=True, max_length=args.max_source_length
        )

        input_ids = inputs_dict.input_ids.to(args.device)
        attention_mask = inputs_dict.attention_mask.to(args.device)
        outputs, logprobs = rag_model.generate(  # rag_model overwrites generate
            input_ids,
            attention_mask=attention_mask,
            num_beams=args.num_beams,
            min_length=args.min_length,
            max_length=args.max_length,
            early_stopping=False,
            num_return_sequences=args.num_return_sequences,
            bad_words_ids=[[0, 0]],  # BART likes to repeat BOS tokens, dont allow it to generate more than one
        )
        answers = rag_model.retriever.generator_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if args.print_predictions:
            for q, a in zip(questions, answers):
                logger.info("Q: {} - A: {}".format(q, a))

        return answers, logprobs.cpu().numpy(), None, None


def evaluate_batch_e2e_with_context(args, rag_model, questions: List[str], **kwargs):
    qtds = [q.split('\t') for q in questions]
    def format_qtd(qtd):
        score = 0.0
        if len(qtd) in {3, 4}:
            q, ct, cd = qtd[:3]
            try:
                score = float(qtd[3]) if len(qtd) > 3 else 0.0
                ct = ct.strip('"')
                text = truncate_context_with_question(
                    ct + rag_model.config.title_sep + cd, q, max_length=args.max_source_length) \
                       + rag_model.config.doc_sep + ('A: ' if args.no_question else q)
            except:
                score = 0.0
                add_context = qtd[3]
                ct = ct.strip('"')
                text = truncate_context_with_question(
                    ct + rag_model.config.title_sep + add_context + ' ' + cd, q, max_length=args.max_source_length) \
                       + rag_model.config.doc_sep + ('A: ' if args.no_question else q)
            text = text.replace('[MASK]', '<mask>')
            return text.replace("  ", " "), score
        if len(qtd) == 1:
            return qtd[0].replace("  ", " "), score
        raise NotImplementedError
    with torch.no_grad():
        text_scores = [format_qtd(qtd) for qtd in qtds]
        context_input = rag_model.retriever.generator_tokenizer.batch_encode_plus(
            list(map(itemgetter(0), text_scores)),
            max_length=args.max_source_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )

        cinput_ids = context_input.input_ids.to(args.device)
        cattention_mask = context_input.attention_mask.to(args.device)
        doc_score = torch.tensor(list(map(itemgetter(1), text_scores))).view(
            cinput_ids.size(0) // args.n_docs, args.n_docs).to(args.device)
        assert cinput_ids.size(0) % args.n_docs == 0, 'the batch is incomplete'

        if not kwargs['candidates']:
            outputs, logprobs = rag_model.generate(
                context_input_ids=cinput_ids,
                context_attention_mask=cattention_mask,
                doc_scores=doc_score,
                num_beams=args.num_beams,
                min_length=args.min_length,
                max_length=args.max_length,
                early_stopping=False,
                num_return_sequences=args.num_return_sequences,
                n_docs=args.n_docs,
                bad_words_ids=[[0, 0]],  # BART likes to repeat BOS tokens, dont allow it to generate more than one
            )
            answers = rag_model.retriever.generator_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        else:
            candidates = kwargs['candidates']
            cand_input_ids = rag_model.retriever.generator_tokenizer.batch_encode_plus(
                candidates,
                max_length=args.max_length,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
            ).input_ids.to(args.device)
            outputs = rag_model(
                context_input_ids=cinput_ids,
                context_attention_mask=cattention_mask,
                doc_scores=doc_score,
                labels=cand_input_ids,
                exclude_bos_score=True,
            )
            logprobs = -outputs['loss']
            answers = candidates

    if args.print_predictions:
        for q, a in zip(questions, answers):
            logger.info("Q: {} - A: {}".format(q, a))

    return answers, logprobs.cpu().numpy(), None, None, None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        choices=["rag_sequence", "rag_token", "bart"],
        type=str,
        help="RAG model type: rag_sequence, rag_token or bart, if none specified, the type is inferred from the model_name_or_path",
    )
    parser.add_argument(
        "--index_name",
        default=None,
        choices=["exact", "compressed", "legacy"],
        type=str,
        help="RAG model retriever type",
    )
    parser.add_argument(
        "--index_path",
        default=None,
        type=str,
        help="Path to the retrieval index",
        nargs='*'
    )
    parser.add_argument("--n_docs", default=1, type=int, help="Number of retrieved docs")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained checkpoints or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--eval_mode",
        choices=["e2e", "retrieval", "e2ec", "e2ec_nq", "break", "pseudo_break", "retrieval_all", "e2e_multichoice", "e2ec_multichoice"],
        default="e2e",
        type=str,
        help="Evaluation mode, e2e calculates exact match and F1 of the downstream task, retrieval calculates precision@k.",
    )
    parser.add_argument(
        "--generation_method",
        choices=["generator", "mask"],
        default="generator",
        type=str,
    )
    parser.add_argument(
        '--retrieval_hop',
        type=int,
        default=1
    )
    parser.add_argument('--use_mdr', action='store_true')
    parser.add_argument('--no_question', action='store_true')
    parser.add_argument("--k", default=1, type=int, help="k for the precision@k calculation")
    parser.add_argument(
        "--evaluation_set",
        default=None,
        type=str,
        required=True,
        help="Path to a file containing evaluation samples",
    )
    parser.add_argument(
        "--gold_data_path",
        default=None,
        type=str,
        required=True,
        help="Path to a tab-separated file with gold samples",
    )
    parser.add_argument(
        "--gold_data_mode",
        default="qa",
        type=str,
        choices=["qa", "ans_tab", "ans"],
        help="Format of the gold data file"
        "qa - a single line in the following format: question [tab] answer_list"
        "ans_tab - a single line in the following format: answers separated by tabs"
        "ans - a single line of the gold file contains the expected answer string",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="predictions.txt",
        help="Name of the predictions file, to be stored in the checkpoints directory",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--recalculate",
        help="Recalculate predictions even if the prediction file exists",
        action="store_true",
    )
    parser.add_argument(
        "--num_beams",
        default=4,
        type=int,
        help="Number of beams to be used when generating answers",
    )
    parser.add_argument(
        "--num_return_sequences",
        default=1,
        type=int,
        help="Number of returned answers",
    )
    parser.add_argument("--min_length", default=1, type=int, help="Min length of the generated answers")
    parser.add_argument("--max_length", default=50, type=int, help="Max length of the generated answers")
    parser.add_argument(
        "--max_source_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--print_predictions",
        action="store_true",
        help="If True, prints predictions while evaluating.",
    )
    parser.add_argument(
        "--print_docs",
        action="store_true",
        help="If True, prints docs retried while generating.",
    )
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


def write_html(questions: List[str], answers: List[str], golds: List[str],
               logprobs: List[float], ret_docs: List[List[List[str]]], ret_doc_ids: List[List[List[int]]], vis_file):
    for i, (ques, ans, gold, lp) in enumerate(zip(questions, answers, golds, logprobs)):
        vis_file.write('<div><span>{}</span></div>\n'.format(ques))
        vis_file.write('<div><span>Prediction: {}</span> <span>{:.5f}</span></div>\n'.format(ans, lp))
        vis_file.write('<div><span>Gold: {}</span></div>\n'.format(gold))
        for nh, hop in enumerate(ret_docs):
            vis_file.write('<div>--- HOP {} ---</div>\n'.format(nh + 1))
            for nd, doc in enumerate(hop[i]):
                vis_file.write('<div> * <b>{}</b> {}</div>\n'.format('-'.join(map(str, ret_doc_ids[nh][i][nd])), doc))
        vis_file.write('<hr>\n')


def write_retrieval(doc_ids: List[List[int]], all_docs: List[Dict[str, List]], ret_file):
    for i, docs in enumerate(all_docs):
        combine = '\t'.join(map(lambda x: '{} || {} || {}'.format('-'.join(map(str, x[0])), x[1], x[2]),
                                zip(doc_ids[i], docs['title'], docs['text'])))
        ret_file.write(combine + '\n')


class Args(object):
    def __init__(self, source, target, prediction, bs):
        self.evaluation_set = source
        self.gold_data_path = target
        self.predictions_path = prediction
        self.eval_batch_size = bs
        self.eval_mode = 'retrieval_all'
        self.device = torch.device("cuda")
        self.num_beams = 5
        self.min_length = 1
        self.max_length = 50
        self.n_docs = 100
        self.max_source_length = 128
        self.print_predictions = False


def run_on_dataset(model, args, evaluate_batch_fn: Callable, score_fn: Callable):
    with open(args.evaluation_set, "r") as eval_file, \
      open(args.gold_data_path, 'r') as gold_file, \
      open(args.predictions_path, "w") as preds_file, \
      open(args.predictions_path + '.html', 'w') as vis_file:
        questions = []
        golds = []
        for line in tqdm(eval_file):
            questions.append(line.strip())
            golds.append(gold_file.readline().rstrip('\n'))
            if len(questions) == args.eval_batch_size:
                answers, logprobs, ret_docs, ret_doc_ids = evaluate_batch_fn(args, model, questions)
                preds_file.write('\n'.join('{}\t{:.5f}'.format(a, l) for a, l in zip(answers, logprobs)) + '\n')
                preds_file.flush()
                if args.eval_mode == 'e2e':
                    write_html(questions, answers, golds, logprobs, ret_docs, ret_doc_ids, vis_file)
                questions = []
                golds = []
        if len(questions) > 0:
            answers, logprobs, ret_docs, ret_doc_ids = evaluate_batch_fn(args, model, questions)
            preds_file.write('\n'.join('{}\t{:.5f}'.format(a, l) for a, l in zip(answers, logprobs)) + '\n')
            preds_file.flush()
            if args.eval_mode == 'e2e':
                write_html(questions, answers, golds, logprobs, ret_docs, ret_doc_ids, vis_file)
        score_fn(args, args.predictions_path, args.gold_data_path)


def main(args):
    model_kwargs = {}
    if args.model_type is None:
        args.model_type = infer_model_type(args.model_name_or_path)
        assert args.model_type is not None
    if args.model_type.startswith("rag"):
        model_class = RagTokenForGeneration if args.model_type == "rag_token" else MyRagSequenceForGeneration
        model_kwargs["n_docs"] = args.n_docs
        if args.index_name is not None:
            model_kwargs["index_name"] = args.index_name
        if args.index_path is not None:
            model_kwargs["index_path"] = args.index_path[0]
    else:
        model_class = BartForConditionalGeneration

    checkpoints = (
        [f.path for f in os.scandir(args.model_name_or_path) if f.is_dir()]
        if args.eval_all_checkpoints
        else [args.model_name_or_path]
    )

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    if args.eval_mode in {"e2e", "e2e_multichoice"}:
        evaluate_batch_fn = evaluate_batch_e2e_multihop_retrieval
        score_fn = get_scores
    elif args.eval_mode in {'e2ec', 'e2ec_nq', 'e2ec_multichoice'}:
        evaluate_batch_fn = evaluate_batch_e2e_with_context
        score_fn = get_scores
    elif args.eval_mode == 'break':
        evaluate_batch_fn = evaluate_batch_e2e_multihop_retrieval
        score_fn = get_scores
    elif args.eval_mode == 'pseudo_break':
        evaluate_batch_fn = evaluate_batch_e2e_with_context
        score_fn = get_scores
    elif args.eval_mode == 'retrieval_all':
        evaluate_batch_fn = evaluate_batch_retrieval_all
        score_fn = get_scores
    else:
        evaluate_batch_fn = evaluate_batch_retrieval
        score_fn = get_precision_at_k

    for checkpoint in checkpoints:
        if os.path.exists(args.predictions_path) and (not args.recalculate):
            logger.info("Calculating metrics based on an existing predictions file: {}".format(args.predictions_path))
            score_fn(args, args.predictions_path, args.gold_data_path)
            continue

        logger.info("***** Running evaluation for {} *****".format(checkpoint))
        logger.info("  Batch size = %d", args.eval_batch_size)
        logger.info("  Predictions will be stored under {}".format(args.predictions_path))

        if args.model_type.startswith("rag"):
            retrievers = []
            if args.eval_mode in {'e2ec', 'e2ec_nq', 'pseudo_break'}:
                retriever = RagRetriever.from_pretrained('facebook/rag-sequence-nq', index_name="exact", use_dummy_dataset=True)
            else:
                if args.use_mdr:
                    retriever = RagRetriever.from_pretrained(
                        'facebook/rag-sequence-base', index_name='custom',
                        passages_path=os.path.join(root_to_mdr, 'data/hotpot_dataset/ds'),
                        index_path=os.path.join(root_to_mdr, 'data/hotpot_dataset/ds_hnsw_index.faiss'))
                else:
                    if args.index_path is not None:
                        retrievers = []
                        for ip in args.index_path:  # load multiple retrievers if any
                            if 'custom_index' in ip:  # custom index build from scratch
                                print(f'load custom index {ip}')
                                retriever = MyRagRetriever.from_pretrained(
                                    'facebook/rag-sequence-nq', index_name='custom',
                                    passages_path=os.path.join(ip, 'ds'),
                                    index_path=os.path.join(ip, 'ds_hnsw_index.faiss'))
                            else:
                                retriever = MyRagRetriever.from_pretrained('facebook/rag-sequence-base')
                            retrievers.append(retriever)
                        retriever = retrievers[0]
                    else:
                        retriever = MyRagRetriever.from_pretrained('facebook/rag-sequence-base')

            model = model_class.from_pretrained(checkpoint, retriever=retriever, **model_kwargs)
            if args.index_path is not None and len(retrievers) > 1:
                model.set_additional_retrievers(retrievers[1:])
            model.retriever.init_retrieval()
            model.retriever.index.dataset._format_type = None  # TODO: avoid bus error
            if args.use_mdr:
                # load question encoder from MDR
                if checkpoint.startswith('facebook'):  # official model
                    MyRagSequenceForGeneration.load_question_encoder(model, os.path.join(root_to_mdr, 'models/q_encoder.pt'))
                else:
                    MyRagSequenceForGeneration.load_question_encoder(model, os.path.join(checkpoint, 'pytorch_model.bin'))
                # load tokenizer from MDR
                model.retriever.question_encoder_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        else:
            model = model_class.from_pretrained(checkpoint, **model_kwargs)

        model.to(args.device)

        if args.eval_mode in {'break', 'pseudo_break'}:
            eval_multi = [-1] if args.eval_mode == 'break' else []
            use_prediction = False
            has_ret = True
            split = 'dev'
            batch_size = args.eval_batch_size
            output_file = args.predictions_path
            if args.eval_mode == 'break':
                break_dataset = Break('../../Break/break_dataset/QDMR-high-level')
            elif args.eval_mode == 'pseudo_break':
                num_hop = 2
                multihop_count = int(args.evaluation_set.split('*')[0])
                source_files = args.evaluation_set.split('*')[1].split(':')
                domains = source_files[0:len(source_files):2]
                source_files = source_files[1:len(source_files):2]
                target_files = args.gold_data_path.split(':')
                break_dataset = PseudoBreak(domains, list(zip(source_files, target_files)), num_hop=num_hop, multihop_count=multihop_count)
            for nh in list(range(break_dataset.max_hop)) + eval_multi:
                id2q = break_dataset.get_hop_n(nh, use_prediction=use_prediction, has_ret=has_ret, split=split)
                print('--- {} with {} questions ---'.format(nh, len(id2q)))
                if len(id2q) <= 0:
                    continue
                id2a = {}
                inds, questions = list(zip(*id2q.items()))
                for b in range(0, len(inds), batch_size):
                    ids = inds[b:b + batch_size]
                    qs = questions[b:b + batch_size]
                    answers, logprobs, ret_docs, ret_doc_ids = evaluate_batch_fn(args, model, qs)[:4]
                    for id, a in zip(ids, answers):
                        id2a[id] = a
                break_dataset.instantiate_hop_n(id2a, nh, split=split)
                break_dataset.save(split, output_file)
        else:
            with open(args.evaluation_set, "r") as eval_file, \
              open(args.gold_data_path, 'r') as gold_file, \
              open(args.predictions_path, "w") as preds_file, \
              open(args.predictions_path + '.html', 'w') as vis_file, \
              open(args.predictions_path + '.ret', 'w') as ret_file:
                questions = []
                golds = []
                alias_li: List[List[str]] = []
                candidates: List[str] = []
                for line in tqdm(eval_file):
                    if args.eval_mode == 'e2e_multichoice':
                        alias = gold_file.readline().rstrip('\n').split('\t')
                        questions.append(line.strip())
                        golds.append(alias[0])  # assume the first one is gold
                        alias_li.append(alias)
                    elif args.eval_mode == 'e2ec_multichoice':
                        cands = gold_file.readline().rstrip('\n').split('\t')
                        questions.extend([line.strip()] * len(cands))
                        candidates.extend(cands)
                        golds.append(cands[0])  # assume the first one is gold
                    else:
                        questions.append(line.strip())
                        golds.append(gold_file.readline().rstrip('\n'))
                    if len(questions) >= args.eval_batch_size:
                        answers, logprobs, ret_docs, ret_doc_ids, all_docs = evaluate_batch_fn(args, model, questions, eval_target=alias_li, candidates=candidates)
                        preds_file.write('\n'.join('{}\t{:.5f}'.format(a, l) for a, l in zip(answers, logprobs)) + '\n')
                        preds_file.flush()
                        if args.eval_mode in {'e2e', 'e2e_multichoice'}:
                            write_html(questions, answers, golds, logprobs, ret_docs, ret_doc_ids, vis_file)
                            write_retrieval(ret_doc_ids[0], all_docs, ret_file)  # TODO: only work for hop1
                        questions = []
                        golds = []
                        alias_li = []
                        candidates = []
                if len(questions) > 0:
                    answers, logprobs, ret_docs, ret_doc_ids, all_docs = evaluate_batch_fn(args, model, questions, eval_target=alias_li, candidates=candidates)
                    preds_file.write('\n'.join('{}\t{:.5f}'.format(a, l) for a, l in zip(answers, logprobs)) + '\n')
                    preds_file.flush()
                    if args.eval_mode in {'e2e', 'e2e_multichoice'}:
                        write_html(questions, answers, golds, logprobs, ret_docs, ret_doc_ids, vis_file)
                        write_retrieval(ret_doc_ids[0], all_docs, ret_file)  # TODO: only work for hop1
                score_fn(args, args.predictions_path, args.gold_data_path)


if __name__ == "__main__":
    args = get_args()
    main(args)
