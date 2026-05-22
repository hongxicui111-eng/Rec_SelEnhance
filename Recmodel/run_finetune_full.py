# -*- coding: utf-8 -*-
# @Time    : 2020/4/25 22:59
# @Author  : Hui Wang

import os
import numpy as np
import random
import torch
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SRDataset, DNSDataset
from trainers import FinetuneTrainer
from models import SASRec
from utils import *
import time
time_stmp = time.time()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='Beauty', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckp', default=0, type=int, help="pretrain epochs 10, 20, 30...")
    parser.add_argument('--num_split', type=int, help='number of split', default=6)


    # model args
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)
    parser.add_argument('--backbone', type=str, default="SASRec", help="backbone model name")
    # Hyperparameters for Attention block
    parser.add_argument('--num_attention_heads', default=2, type=int) 
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=1024, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)


    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")


    # Neg sample
    parser.add_argument("--N", type=int, default=200, help="sample_size")
    parser.add_argument("--M", type=int, default=10, help="pool_size")
    parser.add_argument("--neg_sampler", type=str, default="DNS", help="neg_sampler")
    parser.add_argument("--loss_type", type=str, default="BCE", help="BCE,BPR,InfoNCE")
    parser.add_argument("--temperature", type=float, default=0.1, help="temperature for InfoNCE loss (0.05~0.5 recommended)")
    parser.add_argument("--CL_type", type=str, default="Radical", help="Radical, Gentle")
    parser.add_argument("--start_epoch", default=30, type=int)
    parser.add_argument("--K", default=0.05, type=float)




    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)


    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = True

    args.data_file = args.data_dir + args.data_name

    item_size = 0
    args.data_file = args.data_dir + args.data_name
    train_data, max_item, _ = get_user_seqs(args.data_file + "_train.txt")
    item_size = max(item_size, max_item)
    valid_data, max_item, _ = get_user_seqs(args.data_file + "_val.txt")
    item_size = max(item_size, max_item)
    test_data, max_item, _ = get_user_seqs(args.data_file + "_test.txt")
    item_size = max(item_size, max_item)
    args.item_size = item_size + 2
    valid_matrix = generate_rating_matrix(valid_data, args.item_size)
    test_matrix = generate_rating_matrix(test_data, args.item_size)

    # save model args
    args_str = f'{args.backbone}-{args.data_name}-{args.ckp}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # set item score in train set to `0` in validation
    args.train_matrix = valid_matrix

    # save model
    checkpoint = args_str + str(time_stmp) + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    neg_sampler_dict = {
        'Uniform': SRDataset,
        "DNS": DNSDataset
    }

    print("neg_sampler: ", args.neg_sampler)
    train_dataset = neg_sampler_dict[args.neg_sampler](args, train_data)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=8)

    eval_dataset = neg_sampler_dict[args.neg_sampler](args, valid_data)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, num_workers=8)

    test_dataset = neg_sampler_dict[args.neg_sampler](args, test_data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    model = SASRec(args=args)

    trainer = FinetuneTrainer(model, train_dataloader, eval_dataloader,
                              test_dataloader, args)


    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        trainer.args.train_matrix = test_matrix
        scores, result_info = trainer.test(0, full_sort=True)

    else:
        Epoch = 0
        early_stopping = EarlyStopping(args.checkpoint_path, patience=30, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            # evaluate on NDCG@20
            scores, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                Epoch = epoch
                break
            Epoch = epoch
            
        trainer.args.train_matrix = test_matrix
        print('---------------Change to test_rating_matrix!-------------------')
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=True)

    print(args_str)
    print(result_info)
    to_excel(result_info, args, args.start_epoch, Epoch, training_time=0, inference_time=0)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')
main()