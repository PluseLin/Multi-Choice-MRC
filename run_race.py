from tqdm import tqdm, trange
import os
import argparse
import random

from tensorboardX import SummaryWriter
import logging
import time

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig,BertOrigin

from Utils.race_utils import load_data,get_device, classifiction_metric

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", 
                        default="BertOrigin", 
                        type=str, 
                        help="模型的名字")

    # 文件路径：数据目录， 缓存目录
    parser.add_argument("--data_dir",
                        default="./data/sequence",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--output_dir",
                        default=".bertoutput",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument("--config_name", 
                        default="bert_config.json",
                        type=str)
    parser.add_argument("--weights_name",
                        default="pytorch_model.bin",
                        type=str)

    parser.add_argument("--cache_dir",
                        default=".bertcache",
                        type=str,
                        help="缓存目录，主要用于模型缓存")

    parser.add_argument("--log_dir",
                        default=".bertlog",
                        type=str,
                        help="缓存目录，主要用于模型缓存")
    
    parser.add_argument("--vocab_file",
                    default="./blc/vocab.txt",
                    type=str,
                    help="提供给Tokenizer的单词文件,通常名为vocab.txt")

    parser.add_argument("--bert_model_src",
                    default="./blc",
                    type=str,
                    help="BERT预训练模型来源,可以是名称,url或本地目录")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="随机种子 for initialization")

    # 文本预处理参数
    parser.add_argument("--do_lower_case",
                        default=True,
                        type=bool,
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--max_seq_length",
                        default=386,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    # 训练参数
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--dev_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for dev.")
    parser.add_argument("--test_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for test.")

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                        "E.g., 0.1 = 10%% of training.")
    # optimizer 参数
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="Adam 的 学习率"
                        )
    
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run testing.")

    parser.add_argument('--print_step',
                        type=int,
                        default=400,
                        help="多少步进行模型保存以及日志信息写入")


    # 解决gpu不足问题
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
                                  
    parser.add_argument("--gpu_ids", type=str, default="0", help="gpu 的设备id")
    config = parser.parse_args()

    return config

def train(epoch_num, n_gpu, train_dataloader, dev_dataloader, model, optimizer, criterion, gradient_accumulation_steps, device, label_list, output_model_file, output_config_file, log_dir, print_step):

    model.train()

    writer = SummaryWriter(
        log_dir=log_dir + '/' + time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time())))

    best_dev_loss = float('inf')
    global_step = 0

    for epoch in range(int(epoch_num)):
        print(f'---------------- Epoch: {epoch+1:02} ----------')
        epoch_loss = 0
        train_steps = 0

        all_preds = np.array([], dtype=int)
        all_labels = np.array([], dtype=int)

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            logits = model(input_ids, segment_ids, input_mask)
            loss = criterion(logits.view(-1, len(label_list)),
                             label_ids.view(-1))

            if n_gpu > 1:
                loss = loss.mean()
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            
            train_steps += 1
            # 反向传播
            loss.backward()
            
            epoch_loss += loss.item()

            preds = logits.detach().cpu().numpy()
            outputs = np.argmax(preds, axis=1)
            all_preds = np.append(all_preds, outputs)
            label_ids = label_ids.to('cpu').numpy()
            all_labels = np.append(all_labels, label_ids)

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step != 0 and global_step % print_step == 0:
                train_loss = epoch_loss / train_steps

                train_acc, train_report = classifiction_metric(
                    all_preds, all_labels, label_list)

                dev_loss, dev_acc, dev_report = evaluate(
                    model, dev_dataloader, criterion, device, label_list)
                
                c = global_step // print_step
                logger.info("loss/train,{0},{1}".format(train_loss,c))
                logger.info("loss/dev,{0},{1}".format(dev_loss,c))

                logger.info("acc/train,{0},{1}".format(train_acc,c))
                logger.info("acc/dev,{0},{1}".format(dev_acc,c))
                writer.add_scalar("loss/train", train_loss, c)
                writer.add_scalar("loss/dev", dev_loss, c)

                writer.add_scalar("acc/train", train_acc, c)
                writer.add_scalar("acc/dev", dev_acc, c)

                for label in label_list:
                    writer.add_scalar(label + ":" + "f1/train",
                                      train_report[label]['f1-score'], c)
                    writer.add_scalar(label + ":" + "f1/dev",
                                      dev_report[label]['f1-score'], c)
                    logger.info(label+":f1/train {0},{1}".format(train_report[label]['f1-score'],c))
                    logger.info(label+":f1/dev {0},{1}".format(dev_report[label]['f1-score'],c))

                print_list = ['macro avg', 'weighted avg']
                for label in print_list:
                    writer.add_scalar(label + ":" + "f1/train",
                                      train_report[label]['f1-score'], c)
                    writer.add_scalar(label + ":" + "f1/dev",
                                      dev_report[label]['f1-score'], c)
                    logger.info(label+":f1/train {0},{1}".format(train_report[label]['f1-score'],c))
                    logger.info(label+":f1/dev {0},{1}".format(dev_report[label]['f1-score'],c))

                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss

                    model_to_save = model.module if hasattr(
                        model, 'module') else model
                    torch.save(model_to_save.state_dict(), output_model_file)
                    with open(output_config_file, 'w') as f:
                        f.write(model_to_save.config.to_json_string())
                
    #writer.close()

def evaluate(model, dataloader, criterion, device, label_list):

    model.eval()
    epoch_loss = 0

    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)

    for batch in tqdm(dataloader, desc="Eval"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)
        loss = criterion(logits.view(-1, len(label_list)), label_ids.view(-1))

        preds = logits.detach().cpu().numpy()
        outputs = np.argmax(preds, axis=1)
        all_preds = np.append(all_preds, outputs)

        label_ids = label_ids.to('cpu').numpy()
        all_labels = np.append(all_labels, label_ids)

        epoch_loss += loss.mean().item()
    
    acc, report = classifiction_metric(all_preds, all_labels, label_list)
    return epoch_loss/len(dataloader), acc, report

def main(config):
    print(config.train_batch_size)
    #创建文件夹用于模型缓存，模型输出
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    if not os.path.exists(config.cache_dir):
        os.makedirs(config.cache_dir)

    output_model_file = os.path.join(config.output_dir, config.weights_name)  # 模型输出文件
    output_config_file = os.path.join(config.output_dir, config.config_name)
    
    # 准备设备
    gpu_ids = [int(device_id) for device_id in config.gpu_ids.split()]
    device, n_gpu = get_device(gpu_ids[0])
    if n_gpu > 1:
        n_gpu = len(gpu_ids)

    config.train_batch_size = config.train_batch_size // config.gradient_accumulation_steps

    #设定种子
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)

    #创建tokenizer和label_list
    tokenizer = BertTokenizer.from_pretrained(
        config.vocab_file, do_lower_case=config.do_lower_case)
    label_list = ["0", "1", "2", "3"]

    #准备数据
    if config.do_train:
        # 数据准备
        train_file = os.path.join(config.data_dir, "train.json")
        dev_file = os.path.join(config.data_dir, "dev.json")

        train_dataloader, train_len = load_data(train_file, tokenizer, config.max_seq_length, config.train_batch_size)

        dev_dataloader, dev_len = load_data(dev_file, tokenizer, config.max_seq_length, config.dev_batch_size)

        num_train_steps = int(
            train_len / config.train_batch_size / config.gradient_accumulation_steps * config.num_train_epochs)        

        # 模型准备
        model = BertOrigin.from_pretrained(
            config.bert_model_src,
            cache_dir=config.cache_dir, num_choices=4
        )

        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model,device_ids=gpu_ids)

        # 调优器
        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=config.learning_rate,
                            warmup=config.warmup_proportion,
                             t_total=num_train_steps)
        
        #损失函数
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        #训练
        train(
            epoch_num=config.num_train_epochs, 
            n_gpu=n_gpu, 
            train_dataloader=train_dataloader, 
            dev_dataloader=dev_dataloader, 
            model=model, 
            optimizer=optimizer, 
            criterion=criterion,
            gradient_accumulation_steps=config.gradient_accumulation_steps, 
            device=device, 
            label_list=label_list, 
            output_model_file=output_model_file, 
            output_config_file=output_config_file, 
            log_dir=config.log_dir, 
            print_step=config.print_step
        )
    
    if config.do_test:
        test_file = os.path.join(config.data_dir, "test.json")
        test_dataloader, _ = load_data(
            test_file, tokenizer, config.max_seq_length, config.test_batch_size)
        
        bert_config = BertConfig(output_config_file)
        model = BertOrigin(bert_config, num_choices=len(label_list))
        model.load_state_dict(torch.load(output_model_file))
        model.to(device)

        #损失函数
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        test_loss, test_acc, test_report = evaluate(
            model, 
            test_dataloader, 
            criterion, 
            device, 
            label_list
        )

        print("-------------- Test -------------")
        print(f'\t  Loss: {test_loss: .3f} | Acc: {test_acc*100: .3f} %')

        for label in label_list:
            print('\t {}: Precision: {} | recall: {} | f1 score: {}'.format(
                label, test_report[label]['precision'], test_report[label]['recall'], test_report[label]['f1-score']))
        print_list = ['macro avg', 'weighted avg']

        for label in print_list:
            print('\t {}: Precision: {} | recall: {} | f1 score: {}'.format(
                label, test_report[label]['precision'], test_report[label]['recall'], test_report[label]['f1-score']))



if __name__=="__main__":
    config=get_args()
    
    main(config)
