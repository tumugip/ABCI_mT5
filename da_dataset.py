import io
import os
import json
import sys
import numpy as np
import csv
import argparse
import random
import logging
from logging import INFO, DEBUG, NOTSET
from logging import StreamHandler, FileHandler, Formatter

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# MULTITASKING_TRANSFORM


def encode_t5(src, tgt, hparams):
    inputs = hparams.tokenizer.batch_encode_plus(
        [src],
        max_length=hparams.max_seq_length,
        truncation=True,
        pad_to_max_length=True,
        padding="max_length", return_tensors="pt")
    targets = hparams.tokenizer.batch_encode_plus(
        [tgt],
        max_length=hparams.target_max_seq_length,
        truncation=True,
        pad_to_max_length=True,
        padding="max_length", return_tensors="pt")

    source_ids = inputs["input_ids"].squeeze()
    source_mask = inputs["attention_mask"].squeeze()

    target_ids = targets["input_ids"].squeeze()
    target_mask = targets["attention_mask"].squeeze()

    return {
        "source_ids": source_ids.to(dtype=torch.long),
        "source_mask": source_mask.to(dtype=torch.long),
        "target_ids": target_ids.to(dtype=torch.long),
        "target_mask": target_mask.to(dtype=torch.long),
    }


def encode_string(src, tgt, hparams):
    return src, tgt


def _setup_extra_id(hparams):
    if '<extra_id_0>' not in hparams.tokenizer.vocab:
        hparams.tokenizer.add_tokens(
            [f'<extra_id_{i}>' for i in range(100)])
        hparams.vocab_size += 100

# argparse


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _add_arguments(parser, args_dict):
    for key in args_dict:
        option_name = f'--{key}'
        default = args_dict[key]
        if isinstance(default, bool):
            if default == False:
                parser.add_argument(
                    option_name, action='store_true', default=default)
            elif default == True:
                parser.add_argument(
                    option_name, action='store_false', default=default)
        elif isinstance(default, int):
            parser.add_argument(option_name, type=int, default=default)
        elif isinstance(default, float):
            parser.add_argument(option_name, type=float, default=default)
        elif isinstance(default, str):
            parser.add_argument(option_name, default=default)

def _setup_logger(hparams):
    log_file = f'log{hparams.suffix}.txt'

    # ストリームハンドラの設定
    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO)
    stream_handler.setFormatter(Formatter("%(message)s"))

    # ファイルハンドラの設定
    file_handler = FileHandler(log_file)

    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(
        Formatter(
            "%(asctime)s@ %(name)s [%(levelname)s] %(funcName)s: %(message)s")
    )
    # ルートロガーの設定
    logging.basicConfig(level=NOTSET, handlers=[stream_handler, file_handler])
    logging.info(f'PyTorch: {torch.__version__}')
    logging.info(f'hparams: {hparams}')



def init_hparams(init_dict, description='Trainer of mT5 on ABCI', Tokenizer=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('files', nargs='+', help='files')
    parser.add_argument('--name', type=str, default='', help='project name')
    _add_arguments(parser, init_dict)
    # parser.add_argument('-q', '--quantize', action='store_true',
    #                     help='quantize model')
    hparams = parser.parse_args()

    if hparams.name == '':
        hparams.suffix = ''
        hparams.prefix = ''
    else:
        hparams.prefix = hparams.name
        hparams.suffix = f'_{hparams.name}'
        hparams.output_dir = f'{hparams.prefix}/{hparams.output_dir}'

    _set_seed(hparams.seed)

    if not os.path.isdir(hparams.output_dir):
        os.makedirs(hparams.output_dir)

    if hparams.masking or hparams.target_column == -1:
        hparams.masking = True
        hparams.target_column = -1

    if hparams.additional_tokens == '':
        hparams.additional_special_tokens = None
    else:
        hparams.additional_tokens = hparams.additional_tokens.split()

    if Tokenizer is None:
        hparams.encode = encode_string
    else:
        if not hasattr(hparams, 'use_fast_tokenizer'):
            hparams.use_fast_tokenizer = False
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        hparams.tokenizer = Tokenizer.from_pretrained(
            hparams.tokenizer_name_or_path, is_fast=hparams.use_fast_tokenizer)
        hparams.vocab_size = hparams.tokenizer.vocab_size
        if hparams.additional_tokens:
            hparams.tokenizer.add_tokens(hparams.additional_tokens)
            hparams.vocab_size += len(hparams.additional_tokens)
        if hparams.masking:
            _setup_extra_id(hparams)
        hparams.encode = encode_t5
    hparams.data_duplication = True
    # if not hasattr(hparams, 'da_choice'):
    #     hparams.da_choice = 0.5
    # if not hasattr(hparams, 'da_shuffle'):
    #     hparams.da_shuffle = 0.5
    _setup_logger(hparams)
    return hparams