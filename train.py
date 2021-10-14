import sys
import random
import re
import argparse
import multiprocessing
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from konlpy.tag import Mecab

from model import PaddingMask, TransformerEncoder
from dataset import *
from loader import *
from encoder import *
from masking import *
from tokenizer import *
from scheduler import *
from preprocessor import *

def progressLearning(value, endvalue, mlm_loss, mlm_acc, sop_loss, sop_acc, bar_length=50):
    percent = float(value + 1) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\r[{0}] {1}/{2} \t MLM Loss : {3:.3f} , MLM Acc : {4:.3f} \t SOP Loss : {5:.3f} , SOP Acc : {6:.3f}".format(arrow + spaces,
        value+1, 
        endvalue, 
        mlm_loss, 
        mlm_acc,
        sop_loss,
        sop_acc)
    )
    sys.stdout.flush()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(args) :
    # -- Seed
    seed_everything(args.seed)

    # -- Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- Data
    print('Load Raw Data')
    data = get_data(args.data_dir, args.file_size)
    
    print('Extract Text Data')
    text_data = []
    for json_data in tqdm(data) :
        text_list = [text for text in preprocess_data(json_data) if len(text) < args.sen_max_size]
        text_data.extend(text_list)
    print('\n')

    # -- Tokenizer & Encoder
    mecab = Mecab()
    sen_preprocessor = SenPreprocessor(mecab)
    tokenizer = get_spm(os.path.join(args.token_dir, 'tokenizer.model'))
    v_size = len(tokenizer)

    print('Make Train Data')
    sop_text = []
    sop_label = []
    for text in tqdm(text_data) :
        sen_list = sent_tokenize(text)
        if len(sen_list) < 2 :
            continue
        size = int(len(sen_list)/2)
        order_flag = np.random.binomial(size=size, n=1, p= 0.5)
        idx = 0
        for i in range(0,len(sen_list),2) :
            if i+1 < len(sen_list) :
                prev = sen_preprocessor(sen_list[i])
                next = sen_preprocessor(sen_list[i+1])
                if order_flag[idx] >= 0.5 :
                    sop_text.append((prev,next))
                    sop_label.append(1)
                else :
                    sop_text.append((next,prev))
                    sop_label.append(0)
                idx += 1
    print('\n')

    # -- Encoder
    encoder = Encoder(tokenizer, args.max_size)
    sop_input_ids = []
    sop_type_ids = []
    print('Encoding Train Data')
    for data in tqdm(sop_text) :
        idx_list, type_list = encoder(data)
        sop_input_ids.append(idx_list)
        sop_type_ids.append(type_list)
    print('\n')

    # -- Masking
    masking = Masking(v_size)
    sop_masked_ids = []
    sop_label_ids = []
    print('Masking Train Data')
    for input_ids in tqdm(sop_input_ids) :
        masked_ids, label_ids = masking(input_ids)
        sop_masked_ids.append(masked_ids)
        sop_label_ids.append(label_ids)
    print('\n')

    # -- Dataset
    dset = BertDataset(sop_masked_ids, sop_type_ids, sop_label_ids, sop_label)
    dset_len = [len(data['input_ids']) for data in dset]
    print('Data Size : %d\n' %len(dset))

    # -- DataLoader
    collator =  BertCollator(dset_len, args.batch_size)
    data_loader = DataLoader(dset,
        num_workers=multiprocessing.cpu_count()//2,
        batch_sampler=collator.sample(),
        collate_fn=collator
    )
   
    # -- Model
    padding_mask = PaddingMask()
    # Transformer Encoder
    model = TransformerEncoder(
        layer_size=args.layer_size, 
        max_size=args.max_size, 
        v_size=v_size, 
        d_model=args.embedding_size,
        num_heads=args.head_size,
        hidden_size=args.hidden_size,
        drop_rate=0.1,
        norm_rate=1e-6,
        cuda_flag=use_cuda
    ).to(device)

    init_lr = 1e-4
    # -- Optimizer
    optimizer = optim.Adam(model.parameters(), 
        lr=init_lr, 
        betas=(0.9,0.98), 
        eps=1e-9,
        weight_decay=args.weight_decay
    )

    # -- Scheduler
    schedule_fn = Scheduler(args.embedding_size, init_lr, args.warmup_steps)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, 
        lr_lambda = lambda epoch: schedule_fn(epoch)
    )
    
    # -- Logging
    writer = SummaryWriter(args.log_dir)

    # -- Criterion 
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100).to(device)
    sop_criterion = nn.BCELoss().to(device)

    # -- Training
    log_count = 0
    for epoch in range(args.epochs) :
        idx = 0
        mean_mlm_loss = 0.0
        mean_mlm_acc = 0.0
        mean_sop_loss = 0.0
        mean_sop_acc = 0.0

        print('Epoch : %d/%d' %(epoch, args.epochs))
        for data in data_loader :
            optimizer.zero_grad()
            writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], idx)

            in_data = data['input_ids'].long().to(device)
            type_data = data['type_ids'].long().to(device)
            mask_data = padding_mask(in_data)

            label_data = data['label_ids'].long().to(device)
            label_data = torch.reshape(label_data, (-1,))
    
            sop_data = data['sop'].float().to(device)
    
            out_sop, out_mlm = model(in_data, type_data, mask_data)
            out_mlm = torch.reshape(out_mlm, (-1,v_size))
            out_sop = torch.sigmoid(out_sop)

            mlm_loss = mlm_criterion(out_mlm, label_data)
            mlm_acc = (torch.argmax(out_mlm, dim=-1) == label_data).float()
            mlm_acc = torch.masked_select(mlm_acc, label_data != -100).mean()

            sop_loss = sop_criterion(out_sop, sop_data)
            sop_acc = (torch.where(out_sop >= 0.5, 1.0 , 0.0) == sop_data).float().mean()

            loss = mlm_loss + sop_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
        
            progressLearning(idx, len(data_loader), mlm_loss.item(), mlm_acc.item(), sop_loss.item(), sop_acc.item())
    
            mean_mlm_loss += mlm_loss
            mean_mlm_acc += mlm_acc
            mean_sop_loss += sop_loss
            mean_sop_acc += sop_acc
    
            if (idx + 1) % 100 == 0 :
                writer.add_scalar('train/mlm_loss', mlm_loss.item(), log_count)
                writer.add_scalar('train/mlm_acc', mlm_acc.item(), log_count)
                writer.add_scalar('train/sop_loss', sop_loss.item(), log_count)
                writer.add_scalar('train/sop_acc', sop_acc.item(), log_count)
                log_count += 1
            idx += 1

        mean_mlm_loss /= len(data_loader)
        mean_mlm_acc /= len(data_loader)
        mean_sop_loss /= len(data_loader)
        mean_sop_acc /= len(data_loader)

        torch.save({'epoch' : (epoch) ,  
            'batch_size' : args.batch_size,
            'model_state_dict' : model.state_dict() , 
            'mlm_loss' : mean_mlm_loss.item() , 
            'mlm_acc' : mean_mlm_acc.item() , 
            'sop_loss' : mean_sop_loss.item() , 
            'sop_acc' : mean_sop_acc.item()} , 
        f'./Model/checkpoint_bert.pt') 

        print('\nMean MLM Loss : %.3f , Mean MLM Accuracy : %.3f \t Mean SOP Loss : %.3f , Mean SOP Accuracy : %.3f' %(mean_mlm_loss.item(), 
            mean_mlm_acc.item(), 
            mean_sop_loss.item(), 
            mean_sop_acc.item())
        )

    print('Training Finished')

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()

    # Training argument
    parser.add_argument('--seed', type=int, default=777, help='random seed (default: 777)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='warmup steps of train (default: 2000)')
    parser.add_argument('--sen_max_size', type=int, default=512, help='max size of sentence (default: 512)')
    parser.add_argument('--max_size', type=int, default=512, help='max size of index sequence (default: 512)')
    parser.add_argument('--layer_size', type=int, default=12, help='layer size of model (default: 12)')
    parser.add_argument('--embedding_size', type=int, default=768, help='embedding size of token (default: 768)')
    parser.add_argument('--hidden_size', type=int, default=3072, help='hidden size of position-wise layer (default: 3072)')
    parser.add_argument('--head_size', type=int, default=12, help='head size of multi head attention (default: 12)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 64)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay of optimizer (default: 1e-4)')

    # Container environment
    parser.add_argument('--file_size', type=int, default=10, help='size of newspaper file')
    parser.add_argument('--data_dir', type=str, default='../GPT1/Data')
    parser.add_argument('--model_dir', type=str, default='./Model')
    parser.add_argument('--token_dir', type=str, default='./Tokenizer')
    parser.add_argument('--log_dir' , type=str , default='./Log')

    args = parser.parse_args()
    train(args)

