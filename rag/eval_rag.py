""" Evaluation script for RAG models."""

from typing import List

import argparse
import ast
import logging
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm
from collections import defaultdict

from transformers import BartForConditionalGeneration, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration
from transformers import logging as transformers_logging


sys.path.append(os.path.join(os.getcwd()))  # noqa: E402 # isort:skip
from utils_rag import exact_match_score, f1_score  # noqa: E402 # isort:skip
from rag_model import MyRagSequenceForGeneration


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


def evaluate_batch_retrieval(args, rag_model, questions):
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


def evaluate_batch_e2e(args, rag_model, questions):
    with torch.no_grad():
        inputs_dict = rag_model.retriever.question_encoder_tokenizer.batch_encode_plus(
            questions, return_tensors="pt", padding=True, truncation=True
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
            num_return_sequences=1,
            bad_words_ids=[[0, 0]],  # BART likes to repeat BOS tokens, dont allow it to generate more than one
        )
        answers = rag_model.retriever.generator_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if args.print_predictions:
            for q, a in zip(questions, answers):
                logger.info("Q: {} - A: {}".format(q, a))

        return answers, logprobs.cpu().numpy()


def evaluate_batch_e2e_with_context(args, rag_model, questions: List[str]):
    qtds = [q.split('\t') for q in questions]
    def format_qtd(qtd):
        if len(qtd) == 3:
            q, ct, cd = qtd
            return (ct + rag_model.config.title_sep + cd + rag_model.config.doc_sep + q).replace("  ", " ")
        if len(qtd) == 1:
            return qtd[0].replace("  ", " ")
        raise NotImplementedError
    with torch.no_grad():
        context_input = rag_model.retriever.generator_tokenizer.batch_encode_plus(
            [format_qtd(qtd) for qtd in qtds],
            max_length=rag_model.config.max_combined_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )

        cinput_ids = context_input.input_ids.to(args.device)
        cattention_mask = context_input.attention_mask.to(args.device)
        doc_score = torch.zeros(cinput_ids.size(0), 1).to(args.device)
        outputs, logprobs = rag_model.generate(
            context_input_ids=cinput_ids,
            context_attention_mask=cattention_mask,
            doc_scores=doc_score,
            num_beams=args.num_beams,
            min_length=args.min_length,
            max_length=args.max_length,
            early_stopping=False,
            num_return_sequences=1,
            n_docs=1,
            bad_words_ids=[[0, 0]],  # BART likes to repeat BOS tokens, dont allow it to generate more than one
        )
        answers = rag_model.retriever.generator_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    if args.print_predictions:
        for q, a in zip(questions, answers):
            logger.info("Q: {} - A: {}".format(q, a))

    return answers, logprobs.cpu().numpy()


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
    )
    parser.add_argument("--n_docs", default=5, type=int, help="Number of retrieved docs")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained checkpoints or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--eval_mode",
        choices=["e2e", "retrieval", "e2ec"],
        default="e2e",
        type=str,
        help="Evaluation mode, e2e calculates exact match and F1 of the downstream task, retrieval calculates precision@k.",
    )
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
    parser.add_argument("--min_length", default=1, type=int, help="Min length of the generated answers")
    parser.add_argument("--max_length", default=50, type=int, help="Max length of the generated answers")

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
            model_kwargs["index_path"] = args.index_path
    else:
        model_class = BartForConditionalGeneration

    checkpoints = (
        [f.path for f in os.scandir(args.model_name_or_path) if f.is_dir()]
        if args.eval_all_checkpoints
        else [args.model_name_or_path]
    )

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    if args.eval_mode == "e2e":
        evaluate_batch_fn = evaluate_batch_e2e
        score_fn = get_scores
    elif args.eval_mode == 'e2ec':
        evaluate_batch_fn = evaluate_batch_e2e_with_context
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
            if args.eval_mode == 'e2ec':
                retriever = RagRetriever.from_pretrained(checkpoint, index_name="exact", use_dummy_dataset=True)
            else:
                retriever = RagRetriever.from_pretrained('facebook/rag-sequence-base')
            model = model_class.from_pretrained(checkpoint, retriever=retriever, **model_kwargs)
            model.retriever.init_retrieval()
        else:
            model = model_class.from_pretrained(checkpoint, **model_kwargs)
        model.to(args.device)

        with open(args.evaluation_set, "r") as eval_file, open(args.predictions_path, "w") as preds_file:
            questions = []
            for line in tqdm(eval_file):
                questions.append(line.strip())
                if len(questions) == args.eval_batch_size:
                    answers, logprobs = evaluate_batch_fn(args, model, questions)
                    preds_file.write('\n'.join('{}\t{:.5f}'.format(a, l) for a, l in zip(answers, logprobs)) + '\n')
                    preds_file.flush()
                    questions = []
            if len(questions) > 0:
                answers, logprobs = evaluate_batch_fn(args, model, questions)
                preds_file.write('\n'.join('{}\t{:.5f}'.format(a, l) for a, l in zip(answers, logprobs)) + '\n')
                preds_file.flush()

            score_fn(args, args.predictions_path, args.gold_data_path)


if __name__ == "__main__":
    args = get_args()
    main(args)
