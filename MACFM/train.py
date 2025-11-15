import torch as th
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from utils import *
import dgl
import torch.utils.data as Data
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
import os
import shutil
import argparse
import sys
import csv
import logging
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datetime import datetime
from torch.optim import lr_scheduler
from model import BertGCN
from tqdm import tqdm
import datetime
import time

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=64, help='the input length for bert')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('-m', '--m', type=float, default=0.8, help='the factor balancing BERT and GCN prediction')
parser.add_argument('--nb_epochs', type=int, default=10)
parser.add_argument('--bert_init', type=str, default='/home/shenxiang/xysy/BERT-GCN/pretrained_models/Roberta-mid',
                    choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])
parser.add_argument('--pretrained_bert_ckpt', default=None)
parser.add_argument('--dataset', default='20ng', choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'iflytek', 'thucnews', 'inews', 'fudan', 'sogoucs'])
parser.add_argument('--checkpoint_dir', default=None, help='checkpoint directory, [bert_init]_[gcn_model]_[dataset] if not specified')
parser.add_argument('--gcn_model', type=str, default='gcn')
parser.add_argument('--gcn_layers', type=int, default=5)
parser.add_argument('--n_hidden', type=int, default=256, help='the dimension of gcn hidden layer')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--gcn_lr', type=float, default=1e-3)
parser.add_argument('--bert_lr', type=float, default=2e-5)

get_gpu = 'cuda:0'

args = parser.parse_args()
max_length = args.max_length
batch_size = args.batch_size
m = args.m
nb_epochs = args.nb_epochs
bert_init = args.bert_init
pretrained_bert_ckpt = args.pretrained_bert_ckpt
dataset = args.dataset
checkpoint_dir = args.checkpoint_dir
gcn_model = args.gcn_model
gcn_layers = args.gcn_layers
n_hidden = args.n_hidden
dropout = args.dropout
gcn_lr = args.gcn_lr
bert_lr = args.bert_lr

current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H-%M-%S")

if checkpoint_dir is None:
    ckpt_dir = './checkpoint/{}_{}_{}_{}'.format(bert_init, gcn_model, dataset, formatted_time)
else:
    ckpt_dir = checkpoint_dir
os.makedirs(ckpt_dir, exist_ok=True)
shutil.copy(os.path.basename(__file__), ckpt_dir)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(message)s'))
sh.setLevel(logging.INFO)
fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
fh.setFormatter(logging.Formatter('%(message)s'))
fh.setLevel(logging.INFO)
logger = logging.getLogger('training logger')
logger.addHandler(sh)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

th.cuda.device_count()
cpu = th.device('cpu')
gpu = th.device(get_gpu)

logger.info('arguments:')
logger.info(str(args))
logger.info('checkpoints will be saved in {}'.format(ckpt_dir))
# Model


# Data Preprocess
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)

# compute number of real train/val/test/word nodes and number of classes
nb_node = features.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]

# instantiate model according to class number
model = BertGCN(nb_class=nb_class, pretrained_model=bert_init, m=m, gcn_layers=gcn_layers,
                n_hidden=n_hidden, dropout=dropout)

if pretrained_bert_ckpt is not None:
    ckpt = th.load(pretrained_bert_ckpt, map_location=gpu)
    model.bert_model.load_state_dict(ckpt['bert_model'])
    model.classifier.load_state_dict(ckpt['classifier'])

corpse_file = './data/graph_data/clean_corpus/' + dataset +'_shuffle.txt'
with open(corpse_file, 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.split('\n')

def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
    return input.input_ids, input.attention_mask


input_ids, attention_mask = encode_input(text, model.tokenizer)
input_ids = th.cat([input_ids[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), input_ids[-nb_test:]])
attention_mask = th.cat([attention_mask[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), attention_mask[-nb_test:]])

# transform one-hot label to class ID for pytorch computation
y = y_train + y_test + y_val
y_train = y_train.argmax(axis=1)
y = y.argmax(axis=1)

# document mask used for update feature
doc_mask  = train_mask + val_mask + test_mask

# build DGL Graph
adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')
g.ndata['input_ids'], g.ndata['attention_mask'] = input_ids, attention_mask
g.ndata['label'], g.ndata['train'], g.ndata['val'], g.ndata['test'] = \
    th.LongTensor(y), th.FloatTensor(train_mask), th.FloatTensor(val_mask), th.FloatTensor(test_mask)
g.ndata['label_train'] = th.LongTensor(y_train)
g.ndata['cls_feats'] = th.zeros((nb_node, model.feat_dim))

logger.info('graph information:')
logger.info(str(g))

# create index loader
train_idx = Data.TensorDataset(th.arange(0, nb_train, dtype=th.long))
val_idx = Data.TensorDataset(th.arange(nb_train, nb_train + nb_val, dtype=th.long))
test_idx = Data.TensorDataset(th.arange(nb_node-nb_test, nb_node, dtype=th.long))
doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)
idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)

def save_confusion_matrix_to_csv(confusion_matrix, file_name=ckpt_dir + "confusion_matrix.csv"):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in confusion_matrix:
            writer.writerow(row)

# Training
def update_feature():
    global model, g, doc_mask
    # no gradient needed, uses a large batchsize to speed up the process
    dataloader = Data.DataLoader(
        Data.TensorDataset(g.ndata['input_ids'][doc_mask], g.ndata['attention_mask'][doc_mask]),
        batch_size=1024
    )
    with th.no_grad():
        model = model.to(gpu)
        model.eval()
        cls_list = []
        for i, batch in tqdm(enumerate(dataloader), desc="Update Feature", total=len(dataloader)):
            input_ids, attention_mask = [x.to(gpu) for x in batch]
            output = model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
            cls_list.append(output.cpu())
        cls_feat = th.cat(cls_list, axis=0)
    g = g.to(cpu)
    g.ndata['cls_feats'][doc_mask] = cls_feat
    return g

optimizer = th.optim.Adam([
        {'params': model.bert_model.parameters(), 'lr': bert_lr},
        {'params': model.classifier.parameters(), 'lr': bert_lr},
        {'params': model.gcn.parameters(), 'lr': gcn_lr},
    ], lr=1e-3
)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

def train_step(engine, batch):
    global model, g, optimizer
    model.train()
    model = model.to(gpu)
    g = g.to(gpu)
    optimizer.zero_grad()
    (idx, ) = [x.to(gpu) for x in batch]
    optimizer.zero_grad()
    train_mask = g.ndata['train'][idx].type(th.BoolTensor)
    y_pred = model(g, idx)[train_mask]
    y_true = g.ndata['label_train'][idx][train_mask]
    loss = F.nll_loss(y_pred, y_true)
    loss.backward()
    optimizer.step()
    g.ndata['cls_feats'].detach_()
    train_loss = loss.item()
    with th.no_grad():
        if train_mask.sum() > 0:
            y_true = y_true.detach().cpu()
            y_pred = y_pred.argmax(axis=1).detach().cpu()
            train_acc = accuracy_score(y_true, y_pred)
        else:
            train_acc = 1
    return train_loss, train_acc

trainer = Engine(train_step)

@trainer.on(Events.STARTED)
def start_timer(engine):
    engine.state.start_time = time.time()

@trainer.on(Events.EPOCH_COMPLETED)

def reset_graph(trainer):
    scheduler.step()
    update_feature()
    th.cuda.empty_cache()

def test_step(engine, batch):
    global model, g
    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        g = g.to(gpu)
        (idx, ) = [x.to(gpu) for x in batch]
        _y_pred = model(g, idx)
        _y_true = g.ndata['label'][idx]
        engine.state._y_pred = _y_pred
        engine.state._y_true = _y_true
        return _y_pred, _y_true

evaluator = Engine(test_step)
metrics={
    'acc': Accuracy(),
    'nll': Loss(th.nn.NLLLoss())
}
for n, f in metrics.items():
    f.attach(evaluator, n)

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    global model, g, optimizer
    
    elapsed_time = time.time() - trainer.state.start_time
    trainer.state.start_time = time.time()
    
    with tqdm(total=3, desc="Evaluating", leave=False) as pbar:
        evaluator.run(idx_loader_train)
        pbar.update(1)
        metrics = evaluator.state.metrics
        train_acc, train_nll = metrics["acc"], metrics["nll"]
        evaluator.run(idx_loader_val)
        pbar.update(1)
        metrics = evaluator.state.metrics
        val_acc, val_nll = metrics["acc"], metrics["nll"]
        evaluator.run(idx_loader_test)
        pbar.update(1)
        metrics = evaluator.state.metrics
        test_acc, test_nll = metrics["acc"], metrics["nll"]

    logger.info(
        "Epoch: {}  Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} loss: {:.4f}  Test acc: {:.4f} loss: {:.4f}"
        .format(trainer.state.epoch, train_acc, train_nll, val_acc, val_nll, test_acc, test_nll)
    )
    
    logger.info("Epoch: {}  Time per epoch: {:.4f}s".format(trainer.state.epoch, elapsed_time))
        
    if val_acc > log_training_results.best_val_acc:
        logger.info("New checkpoint")
        th.save(
            {
                'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'gcn': model.gcn.state_dict(),
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_val_acc = val_acc

y_true_list = []
y_pred_list = []
features_list = []

@evaluator.on(Events.ITERATION_COMPLETED)
def collect_y_true_and_y_pred(engine):
    features, _ = engine.state.output
    y_true = engine.state._y_true.cpu().numpy()
    features_list.extend(features.cpu().numpy())
    y_true_list.extend(y_true)
    y_pred_list.extend(engine.state._y_pred.argmax(axis=1).cpu().numpy())

@trainer.on(Events.COMPLETED)
def output_classification_report_and_confusion_matrix(trainer):
    evaluator.run(idx_loader_test)
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    logger.info(
        "\nAccuracy: {:.4f}".format(acc)
    )

    logger.info(
        "\nF1 Score: {:.4f}".format(f1)
    )

    logger.info(
        "\nConfusion Matrix:{}".format(confusion_matrix(y_true, y_pred))
    )
    
    save_confusion_matrix_to_csv(confusion_matrix(y_true, y_pred))

    logger.info(
        "\nClassification Report:{}".format(classification_report(y_true, y_pred, digits=4))
    )


log_training_results.best_val_acc = 0
g = update_feature()
start_total_time = time.time()
trainer.run(idx_loader, max_epochs=nb_epochs)
end_total_time = time.time()
total_elapsed_time = end_total_time - start_total_time
logger.info("Total training time: {:.4f}s".format(total_elapsed_time))

