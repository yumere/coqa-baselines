from torch.utils.data import DataLoader
from coqa_utils import CoqaDataset
import torch
from torch import nn
import argparse
import os
from transformers import BertForQuestionAnswering
from transformers import BertPreTrainedModel
from transformers import BertModel
from transformers import BertTokenizer
from transformers import AdamW, WarmupLinearSchedule


class CoqaBase(BertPreTrainedModel):
    def __init__(self, config):
        super(CoqaBase, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert: BertModel = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, segment_ids, input_mask):
        outputs = self.bert(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)


def train(model, args):
    tokenizer = BertTokenizer.from_pretrained(args.model_type)
    batch_size = args.batch_size * (args.n_gpu if args.n_gpu else 1)

    dataset_config = {
        "max_sequence_length": args.max_sequence_length
    }
    train_set = CoqaDataset(args.train_file, dataset_config, tokenizer)
    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=CoqaDataset.collate_fn, shuffle=False if args.debug else True)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rte, eps=args.adam_epsilon)


def eval(model, args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--model_type', type=str, default='bert-base-uncased')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--eval_file', type=str)

    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_sequence_length', type=int, default=384)
    parser.add_argument()
    args = parser.parse_args()

    assert args.train and args.eval, "You must set --train or --eval options"
    device = torch.device("cuda") if not args.no_cuda and torch.cuda.is_available() else torch.device("cpu")
    n_gpu = torch.cuda.device_count()
    args.device = device
    args.n_gpu = n_gpu

    model = BertForQuestionAnswering.from_pretrained(args.model_type)
    model.to(device)

    if args.train:
        train(model, args)

    if args.eval:
        eval()

