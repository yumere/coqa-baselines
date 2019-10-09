import argparse
import json
import re
import string
from collections import Counter
from collections import namedtuple

import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer

__all__ = ["CoqaDataset", "InputFeature"]


UNK = ' unknown'


class CoqaDataset(Dataset):
    def __init__(self, filename, args):
        super(CoqaDataset, self).__init__()
        self.args = args
        self.dataset = torch.load(open(filename, "rb"))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


def _check_is_max_context(doc_spans, cur_span_index, position):
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == '\xa0':
        return True
    return False


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def find_span(offsets, start, end):
    start_index = end_index = -1
    for i, offset in enumerate(offsets):
        if (start_index < 0) or (start >= offset[0]):
            start_index = i
        if (end_index < 0) and (end <= offset[1]):
            end_index = i
    return (start_index, end_index)


def normalize_answer(s):
    """Lower text and remove punctuation, storys and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def find_span_with_gt(context, offsets, ground_truth):
    best_f1 = 0.0
    best_span = (len(offsets) - 1, len(offsets) - 1)
    gt = normalize_answer(ground_truth).split()

    ls = [i for i in range(len(offsets))
          if context[offsets[i][0]:offsets[i][1]].lower() in gt]

    for i in range(len(ls)):
        for j in range(i, len(ls)):
            pred = normalize_answer(context[offsets[ls[i]][0]: offsets[ls[j]][1]]).split()
            common = Counter(pred) & Counter(gt)
            num_same = sum(common.values())
            if num_same > 0:
                precision = 1.0 * num_same / len(pred)
                recall = 1.0 * num_same / len(gt)
                f1 = (2 * precision * recall) / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_span = (ls[i], ls[j])
    return best_span


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return input_start, input_end


@dataclass
class InputFeature:
    unique_id: str
    paragraph_id: int
    turn_id: int
    doc_span_index: int

    context: str

    question: str
    answer: str
    annotated_question: list
    annotated_answer: list

    tokens: list

    start_position: int
    end_position: int

    input_ids: list
    segment_ids: list
    input_mask: list



def preprocess(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    def process(text, tokenize=False):
        nonlocal tokenizer
        tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        for i, c in enumerate(text):
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    tokens.append(c)
                else:
                    tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(tokens) - 1)

        if not tokenize:
            return tokens, char_to_word_offset

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_tokens = []
        for (i, token) in enumerate(tokens):
            orig_to_tok_index.append(len(all_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_tokens.append(sub_token)

        return tokens, all_tokens, char_to_word_offset, tok_to_orig_index, orig_to_tok_index

    with open(args.data_file, "rt") as f:
        dataset = json.load(f)

    data = []
    for paragraph_id, datum in enumerate(tqdm(dataset['data'], desc="Data", ncols=85)):
        context_str = datum["story"] + UNK
        _datum = {"context": context_str,
                  "source": datum["source"],
                  "id": datum["id"],
                  "filename": datum["filename"]}
        _datum['qas'] = []

        prev_is_whitespace = True

        doc_tokens, all_doc_tokens, char_to_word_offset, tok_to_orig_index, orig_to_tok_index = process(context_str, tokenize=True)

        _datum["annotated_context"] = all_doc_tokens
        _datum["char_to_word_offset"] = char_to_word_offset
        _datum["tok_to_orig_index"] = tok_to_orig_index
        _datum["orig_to_tok_index"] = orig_to_tok_index

        assert len(datum["questions"]) == len(datum["answers"])

        additional_answers = {}
        if "additional_answers" in datum:
            for k, answer in datum["additional_answers"].items():
                if len(answer) == len(datum['answers']):
                    for ex in answer:
                        idx = ex['turn_id']
                        if idx not in additional_answers:
                            additional_answers[idx] = []
                        additional_answers[idx].append(ex['input_text'])

        for question, answer in zip(datum['questions'], datum['answers']):
            assert question['turn_id'] == answer['turn_id']
            idx = question['turn_id']
            _qas = {'turn_id': idx,
                    'question': question['input_text'],
                    'answer': answer['input_text']}
            _qas['annotated_question'] = process(question['input_text'])[0]
            _qas['annotated_answer'] = process(answer['input_text'])[0]

            if idx in additional_answers:
                _qas["additional_answers"] = additional_answers[idx]

            start = answer['span_start']
            end = answer['span_end']

            # Sometimes there is unknown answer
            if start == -1 and end == -1:
                tok_start = len(all_doc_tokens) - 1
                tok_end = len(all_doc_tokens) - 1
                _qas['answer_span'] = (tok_start, tok_end)
                _datum['qas'].append(_qas)
                continue

            chosen_text = _datum['context'][start: end]
            if args.do_lower_case:
                chosen_text = chosen_text.lower()

            # strip the whitespaces in the span text of the answer
            while len(chosen_text) > 0 and chosen_text[0] in string.whitespace:
                chosen_text = chosen_text[1:]
                start += 1
            while len(chosen_text) > 0 and chosen_text[-1] in string.whitespace:
                chosen_text = chosen_text[:-1]
                end -= 1

            input_text = _qas['answer'].strip()
            if args.do_lower_case:
                input_text = input_text.lower()

            if input_text in chosen_text:
                input_text_tokens = whitespace_tokenize(input_text)
                input_text_length = len(input_text_tokens)

                start = -1
                for i, tok in enumerate(doc_tokens):
                    if input_text_tokens == doc_tokens[i: i + input_text_length]:
                        start = i
                        break

                end = start + input_text_length
                tok_start = orig_to_tok_index[start]
                if end < len(doc_tokens) - 1:
                    tok_end = orig_to_tok_index[end + 1] - 1
                else:
                    tok_end = len(all_doc_tokens) - 1

                tok_start, tok_end = _improve_answer_span(all_doc_tokens, tok_start, tok_end, tokenizer, input_text)
                _qas['answer_span'] = (tok_start, tok_end)

            else:
                answer_length = len(chosen_text)

                start_position = char_to_word_offset[start]
                end_position = char_to_word_offset[start + answer_length - 1]

                tok_start = orig_to_tok_index[start_position]
                if end < len(doc_tokens) - 1:
                    tok_end = orig_to_tok_index[end_position + 1] - 1
                else:
                    tok_end = len(all_doc_tokens) - 1
                tok_start, tok_end = _improve_answer_span(all_doc_tokens, tok_start, tok_end, tokenizer, chosen_text)
                _qas['answer_span'] = (tok_start, tok_end)

            _datum['qas'].append(_qas)
        data.append(_datum)

    paragraph_lens = []
    question_lens = []
    paragraphs = []
    examples = []

    for paragraph_id, paragraph in enumerate(data):
        history = []
        for qas in paragraph['qas']:
            tok_start, tok_end = qas['answer_span']
            qas['paragraph_id'] = paragraph_id

            temp = []
            n_history = len(history) if args.n_history < 0 else min(args.n_history, len(history))

            if n_history > 0:
                for i, (q, a) in enumerate(history[-n_history:]):
                    d = n_history - i
                    temp.extend(q)
                    temp.extend(a)

            temp.extend(qas['annotated_question'])
            history.append((qas['annotated_question'], qas['annotated_answer']))
            qas['annotated_question'] = temp

            question = qas['annotated_question']
            answers = [qas['answer']]
            if "additional_answers" in qas:
                answers = answers + qas['additional_answers']

            sample = {'id': (paragraph['id'], qas['turn_id']),
                      'question': question,
                      'answers': answers,
                      'evidence': paragraph['annotated_context']}

            max_tokens_for_doc = args.max_sequence_length - len(question) - 3

            _DocSpan = namedtuple("DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            all_doc_tokens = paragraph['annotated_context']
            tok_to_orig_index = paragraph['tok_to_orig_index']
            orig_to_tok_index = paragraph['orig_to_tok_index']
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc

                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, args.doc_stride)

            for doc_span_index, doc_span in enumerate(doc_spans):
                tokens = []
                segment_ids = []
                p_mask = []
                token_to_orig_map = {}
                token_is_max_context = {}

                tokens.append("[CLS]")
                segment_ids.append(0)
                p_mask.append(1)

                for token in question:
                    tokens.append(token)
                    segment_ids.append(0)
                    p_mask.append(1)

                tokens.append("[SEP]")
                segment_ids.append(0)
                p_mask.append(1)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index
                    is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                    p_mask.append(0)
                paragraph_len = doc_span.length

                tokens.append("[SEP]")
                segment_ids.append(1)
                p_mask.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                while len(input_ids) < args.max_sequence_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                    p_mask.append(1)

                assert len(input_ids) == len(input_mask) == len(segment_ids) == args.max_sequence_length

                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False

                if not (tok_start >= doc_start and tok_end <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    doc_offset = len(question) + 2
                    start_position = tok_start - doc_start + doc_offset
                    end_position = tok_end - doc_start + doc_offset

                examples.append(InputFeature(
                    unique_id=paragraph['id'],
                    paragraph_id=paragraph_id,
                    turn_id=qas['turn_id'],
                    doc_span_index=doc_span_index,
                    context=paragraph['context'],
                    question=qas['question'],
                    answer=qas['answer'],
                    annotated_question=qas['annotated_question'],
                    annotated_answer=qas['annotated_answer'],
                    tokens=tokens,
                    start_position=start_position,
                    end_position=end_position,
                    input_ids=input_ids,
                    segment_ids=segment_ids,
                    input_mask=input_mask
                ))
    with open(args.output_file, "wb") as f:
        torch.save(examples, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", "-d", type=str, required=True)
    parser.add_argument("--output_file", "-o", type=str, required=True)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument('--do_lower_case', action='store_true', default=False)
    parser.add_argument('--prepro', action='store_true', default=False)

    parser.add_argument('--n_history', type=int, default=2)
    parser.add_argument('--max_sequence_length', type=int, default=384)
    parser.add_argument('--doc_stride', type=int, default=128)
    args = parser.parse_args()

    if args.prepro:
        preprocess(args)

    else:
        from torch.utils.data import DataLoader

        dataset = CoqaDataset("./coqa-dataset/processed-train-50.pkl", args)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=lambda x: x)

        for i, d in enumerate(loader):
            print(d)

            batch = {
                'input_ids': torch.tensor([z.input_ids for z in d]),
                'segment_ids': torch.tensor([z.segment_ids for z in d]),
                'input_mask': torch.tensor([z.input_mask for z in d]),
                'start_positions': torch.tensor([z.start_position for z in d]),
                'end_positions': torch.tensor([z.end_position for z in d])
            }
            if i > 5:
                break


