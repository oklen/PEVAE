from __future__ import unicode_literals  # at top of module
from collections import Counter
import numpy as np
import json
from models.utils import get_tokenize, get_chat_tokenize, missingdict, Pack
import logging
import os
import itertools
from collections import defaultdict
import copy
import random
import torch

PAD = '[PAD]'
UNK = '[UNK]'
BOS = '[CLS]'
EOS = '[SEP]'
BOD = "<d>"
EOD = "</d>"
BOT = "<t>"
EOT = "</t>"
ME = "<me>"
OT = "<ot>"
SYS = "<sys>"
USR = "<usr>"
KB = "<kb>"
SEP = "|"
REQ = "<requestable>"
INF = "<informable>"
WILD = "%s"


class StanfordCorpus(object):
    logger = logging.getLogger(__name__)

    def __init__(self, config):
        self.config = config
        self._path = config.data_dir
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_chat_tokenize()
        self.train_corpus = self._read_file(os.path.join(self._path, 'kvret_train_public.json'))
        self.valid_corpus = self._read_file(os.path.join(self._path, 'kvret_dev_public.json'))
        self.test_corpus = self._read_file(os.path.join(self._path, 'kvret_test_public.json'))
        self._build_vocab(config.max_vocab_cnt)
        # self._output_hyps(os.path.join(self._path, 'kvret_test_public.hyp'))
        print("Done loading corpus")

    def _output_hyps(self, path):
        if not os.path.exists(path):
            f = open(path, "w", encoding="utf-8")
            for utts in self.test_corpus:
                for utt in utts:
                    if utt['speaker'] != 0:
                        f.write(' '.join(utt['utt_ori']) + "\n")
            f.close()

    def _read_file(self, path):
        with open(path, 'rb') as f:
            data = json.load(f)

        return self._process_dialog(data)

    def _process_dialog(self, data):
        new_dialog = []
        bod_utt = [BOS, BOD, EOS]
        eod_utt = [BOS, EOD, EOS]
        all_lens = []
        all_dialog_lens = []
        speaker_map = {'assistant': SYS, 'driver': USR}
        for raw_dialog in data:
            intent = raw_dialog['scenario']['task']['intent']
            dialog = [Pack(utt=bod_utt,
                           speaker=0,
                           meta={'intent': intent, "text": ' '.join(bod_utt[1:-1])})]
            for turn in raw_dialog['dialogue']:

                utt = turn['data']['utterance']
                utt_ori = self.tokenize(utt)
                utt = [BOS, speaker_map[turn['turn']]] + utt_ori + [EOS]
                all_lens.append(len(utt))
                # meta={"text": line.strip()}
                dialog.append(Pack(utt=utt, speaker=turn['turn'], utt_ori=utt_ori, meta={'intent': intent,
                                                                                         'text': ' '.join(utt[1:-1])}))

            if hasattr(self.config, 'include_eod') and self.config.include_eod:
                dialog.append(Pack(utt=eod_utt, speaker=0, meta={'intent': intent,
                                                                 'text': ' '.join(eod_utt[1:-1])}))

            all_dialog_lens.append(len(dialog))
            new_dialog.append(dialog)

        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        print("Max dialog len %d, mean dialog len %.2f" % (
            np.max(all_dialog_lens), float(np.mean(all_dialog_lens))))
        return new_dialog

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for dialog in self.train_corpus:
            for turn in dialog:
                all_words.extend(turn.utt)

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus),
                 len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1],
                 float(discard_wc) / len(all_words)))

        self.vocab = [PAD, UNK, SYS, USR] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab[UNK]
        print("<d> index %d" % self.rev_vocab[BOD])

    def _sent2id(self, sent):
        return [self.rev_vocab.get(t, self.unk_id) for t in sent]

    def _to_id_corpus(self, data):
        results = []
        for dialog in data:
            temp = []
            # convert utterance and feature into numeric numbers
            for turn in dialog:
                id_turn = Pack(utt=self._sent2id(turn.utt),
                               speaker=turn.speaker,
                               meta=turn.get('meta'))
                temp.append(id_turn)
            results.append(temp)
        return results

    def get_corpus(self):
        id_train = self._to_id_corpus(self.train_corpus)
        id_valid = self._to_id_corpus(self.valid_corpus)
        id_test = self._to_id_corpus(self.test_corpus)
        return Pack(train=id_train, valid=id_valid, test=id_test)

class PTBCorpus(object):
    logger = logging.getLogger()

    def __init__(self, config):
        self.config = config
        self._path = config.data_dir
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_chat_tokenize()
        self.train_corpus = self._read_file(os.path.join(self._path, 'ptb.train.txt'))
        self.valid_corpus = self._read_file(os.path.join(self._path, 'ptb.valid.txt'))
        self.test_corpus = self._read_file(os.path.join(self._path, 'ptb.test.txt'))
        self._build_vocab(config.max_vocab_cnt)
        self.unk = "<unk>"
        print("Done loading corpus")

    def _read_file(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()

        return self._process_data(lines)

    def _process_data(self, data):
        all_text = []
        all_lens = []
        for line in data:
            tokens = [BOS] + line.strip().split() + [EOS]
            all_lens.append(len(tokens))
            all_text.append(Pack(utt=tokens, speaker=0))
        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        return all_text

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for turn in self.train_corpus:
            all_words.extend(turn.utt)

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus),
                 len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1],
                 float(discard_wc) / len(all_words)))

        self.vocab = [PAD] + [t for t, cnt in vocab_count]
        if UNK not in self.vocab:
            self.vocab = [PAD, UNK] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab[UNK]

    def _sent2id(self, sent):
        return [self.rev_vocab.get(t, self.unk_id) for t in sent]

    def _to_id_corpus(self, data):
        results = []
        for line in data:
            id_turn = Pack(utt=self._sent2id(line.utt),
                           speaker=line.speaker,
                           meta=line.get('meta'))
            results.append(id_turn)
        return results

    def get_corpus(self):
        id_train = self._to_id_corpus(self.train_corpus)
        id_valid = self._to_id_corpus(self.valid_corpus)
        id_test = self._to_id_corpus(self.test_corpus)
        return Pack(train=id_train, valid=id_valid, test=id_test)

class DailyDialogCorpus(object):
    logger = logging.getLogger()

    def __init__(self, config):
        self.config = config
        self._path = config.data_dir
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_chat_tokenize()
        self.train_corpus = self._read_file(os.path.join(self._path, 'train'))
        self.valid_corpus = self._read_file(os.path.join(self._path, 'validation'))
        self.test_corpus = self._read_file(os.path.join(self._path, 'test'))
        self._build_vocab(config.max_vocab_cnt)
        print("Done loading corpus")

    def _read_file(self, path):
        with open(os.path.join(path, 'dialogues.txt'), 'r') as f:
            txt_lines = f.readlines()

        with open(os.path.join(path, 'dialogues_act.txt'), 'r') as f:
            da_lines = f.readlines()

        with open(os.path.join(path, 'dialogues_emotion.txt'), 'r') as f:
            emotion_lines = f.readlines()

        combined_data = [(t, d, e) for t, d, e in zip(txt_lines, da_lines, emotion_lines)]

        return self._process_dialog(combined_data)

    def _process_dialog(self, data):
        new_dialog = []
        bod_utt = [BOS, BOD, EOS]
        eod_utt = [BOS, EOD, EOS]
        all_lens = []
        all_dialog_lens = []
        for raw_dialog, raw_act, raw_emotion in data:
            dialog = [Pack(utt=bod_utt,
                           speaker=0,
                           meta=None)]

            # raw_dialog = raw_dialog.decode('ascii', 'ignore').encode()
            raw_dialog = raw_dialog.split('__eou__')[0:-1]
            raw_act = raw_act.split()
            raw_emotion = raw_emotion.split()

            for t_id, turn in enumerate(raw_dialog):
                utt = turn
                utt = [BOS] + self.tokenize(utt.lower()) + [EOS]
                all_lens.append(len(utt))
                dialog.append(Pack(utt=utt, speaker=t_id%2,
                                   meta={'emotion': raw_emotion[t_id], 'act': raw_act[t_id]}))

            if not hasattr(self.config, 'include_eod') or self.config.include_eod:
                dialog.append(Pack(utt=eod_utt, speaker=0))

            all_dialog_lens.append(len(dialog))
            new_dialog.append(dialog)

        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        print("Max dialog len %d, mean dialog len %.2f" % (
            np.max(all_dialog_lens), float(np.mean(all_dialog_lens))))
        return new_dialog

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for dialog in self.train_corpus:
            for turn in dialog:
                all_words.extend(turn.utt)

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus),
                 len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1],
                 float(discard_wc) / len(all_words)))

        self.vocab = [PAD, UNK, SYS, USR] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab[UNK]

    def _sent2id(self, sent):
        return [self.rev_vocab.get(t, self.unk_id) for t in sent]

    def _to_id_corpus(self, data):
        results = []
        for dialog in data:
            temp = []
            # convert utterance and feature into numeric numbers
            for turn in dialog:
                id_turn = Pack(utt=self._sent2id(turn.utt),
                               speaker=turn.speaker,
                               meta=turn.get('meta'))
                temp.append(id_turn)
            results.append(temp)
        return results

    def get_corpus(self):
        id_train = self._to_id_corpus(self.train_corpus)
        id_valid = self._to_id_corpus(self.valid_corpus)
        id_test = self._to_id_corpus(self.test_corpus)
        return Pack(train=id_train, valid=id_valid, test=id_test)
        
from transformers import AutoTokenizer

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 ans_ids=None,
                #  sen1_end,sen2_end,sen3_end
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.ans_ids = ans_ids

class aNLG_feature(object):
    def __init__(self,id,obs,seg_ids,input_masks,ans,ob1_end,obs_end,ans_end):
        self.obs_ids = obs
        self.obs_segment_ids = seg_ids
        self.input_masks = input_masks
        self.ans = ans
        self.ob1_end = ob1_end
        self.obs_end = obs_end
        self.ans_end = ans_end
        self.id = id

import pickle

class aNLG(object):
    logger = logging.getLogger()
    def __init__(self, config, tokenizer=None):
        self.config = config
        self.data_cnt = 0
        self.id2index = {}
        self.index2id = {}
        self._path = config.data_dir
        self.max_seq_len = 64
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config.encoder_model)
        self.pre_train_corpus = self.get_pretrain_data(os.path.join(self._path, 'train.json'))
        self.train_corpus = self._read_file(os.path.join(self._path, 'train.json'))
        self.valid_corpus = self._read_file(os.path.join(self._path, 'dev.json'))
        self.test_corpus = self._read_file(os.path.join(self._path, 'test.json')) # Test data annoate invaild answer

        self.vocab = self.tokenizer.vocab

        # self.test_corpus = self._read_file(os.path.join(self._path, 'test'))
        # self._build_vocab(config.max_vocab_cnt)
        
        print("Done loading corpus aNLG")

    def convert_and_fill(self, text, segment_bound):
        cur_data = self.tokenizer.convert_tokens_to_ids(['[CLS]']+ text +['[SEP]'])
        segment_ids = [0]*(segment_bound+2) + [1]*(len(cur_data)-segment_bound-2)
        input_masks = [1]*len(cur_data)
        padding = [0] * (self.max_seq_len - len(cur_data))
        cur_data = cur_data + padding
        segment_ids = segment_ids + padding
        input_masks = input_masks + padding
        return cur_data,segment_ids,input_masks

    def get_pretrain_data(self,path=None):
        if path is None:
            path = os.path.join(self._path, 'train.json')

        (dir,filename) = os.path.split(path)
        save_path = './feature/'+filename+'_pre.pkl'
        if os.path.exists(save_path):
            print("Load exists file: ",save_path)
            return pickle.load(open(save_path,'rb'))
        res = []
        label_list = ['obs1','obs2','hyp']

        with open(path, 'r') as f:
            for line in f.readlines():
                for label in label_list:
                    data = json.loads(line)
                    cur_data = data[label]
                    data_len = len(cur_data)
                    res.append([
                        self.convert_and_fill(self.tokenizer.tokenize(cur_data), len(self.tokenizer.tokenize(cur_data))),
                        data_len,
                    ])
        pickle.dump(res,open(save_path,'wb'))
        return res

    def get_pretrain_valid_data(self,path=None):
        if path is None:
            path = os.path.join(self._path, 'dev.json')

        (dir,filename) = os.path.split(path)
        save_path = './feature/'+filename+'_pre.pkl'
        if os.path.exists(save_path):
            print("Load exists file: ",save_path)
            return pickle.load(open(save_path,'rb'))
        res = []
        label_list = ['obs1','obs2','hyp']

        with open(path, 'r') as f:
            for line in f.readlines():
                for label in label_list:
                    data = json.loads(line)
                    cur_data = data[label]
                    data_len = len(cur_data)
                    res.append([
                        self.convert_and_fill(self.tokenizer.tokenize(cur_data), len(self.tokenizer.tokenize(cur_data))),
                        data_len,
                    ])
        pickle.dump(res,open(save_path,'wb'))
        return res

    def _read_file(self, path):
        (dir,filename) = os.path.split(path)
        print(path)
        save_path = './feature/'+filename+'.pkl'
        if os.path.exists(save_path):
            print('Load exist file ',save_path)
            return pickle.load(open(save_path,'rb'))
        res = []
        with open(path, 'r') as f:
            for line in f.readlines():
                data = json.loads(line)
                self.id2index[data['story_id']] = self.data_cnt
                self.index2id[self.data_cnt]=data['story_id']
                res.append([
                    self.data_cnt,
                    data['obs1'],
                    data['obs2'],
                    data['hyp'],
                ])
                self.data_cnt += 1
        res = self._process_data(res)
        pickle.dump(res,open(save_path,'wb'))
        return res

    def _process_data(self, data):
        res = []
        max_olens = 0
        for d in data:
            ob1 = self.tokenizer.tokenize(d[1])
            ob2 = self.tokenizer.tokenize(d[2])
            max_olens = max(max_olens,len(ob1)+len(ob2)+3)
        if max_olens > self.max_seq_len:
            print('actually max input lengths:{}',max_olens)
            self.max_seq_len *= 2
        for d in data:
            ob1 = self.tokenizer.tokenize(d[1])
            ob2 = self.tokenizer.tokenize(d[2])
            hyp = self.tokenizer.tokenize(d[3])
            #The Last Token index is point to last token
            res.append(aNLG_feature(d[0],*self.convert_and_fill(ob1+['[SEP]']+ob2,len(ob1)),
            self.tokenizer.convert_tokens_to_ids(hyp)+[0]*(self.max_seq_len-len(hyp)),len(ob1)+1,len(ob1)+3+len(ob2),
            len(hyp)))
        return res

    def get_corpus(self):
        return Pack(train=self.train_corpus, valid=self.valid_corpus, test=self.test_corpus)



class PW(object):
    logger = logging.getLogger()
    def __init__(self, config, tokenizer = None):
        self.config = config
        self.data_cnt = 0
        self.id2index = {}
        self.index2id = {}
        self._path = config.data_dir
        self.max_seq_len = 64
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config.encoder_model)
        self.pre_train_corpus = self.get_pretrain_data(os.path.join(self._path, 'train.wp_target.pkl'))
        self.pre_train_valid_corpus = self.get_pretrain_valid_data(os.path.join(self._path, 'valid.wp_target.pkl'))
        self.vocab = self.tokenizer.vocab

        
        print("Done loading corpus PW")


    def get_pretrain_data(self,path=None):
        if path is None:
            return self.pre_train_corpus
        (dir,filename) = os.path.split(path)
        save_path = './feature/'+filename
        return pickle.load(open(save_path,'rb'))

    def get_pretrain_valid_data(self,path=None):
        if path is None:
            return self.pre_train_valid_corpus
        (dir,filename) = os.path.split(path)
        save_path = './feature/'+filename
        return pickle.load(open(save_path,'rb'))

    def _read_file(self, path):
        (dir,filename) = os.path.split(path)
        save_path = './feature/'+filename+'.pkl'
        return pickle.load(open(save_path,'rb'))

    def get_corpus(self):
        return Pack(train=self.train_corpus, valid=self.valid_corpus, test=self.test_corpus)


PAD = '<pad>'
UNK = '<unk>'
BOS = '<s>'
EOS = '</s>'
BOD = "<d>"
EOD = "</d>"
BOT = "<t>"
EOT = "</t>"
ME = "<me>"
OT = "<ot>"
SYS = "<sys>"
USR = "<usr>"
KB = "<kb>"
SEP = "|"
REQ = "<requestable>"
INF = "<informable>"
WILD = "%s"

class yelp(object):
    logger = logging.getLogger()

    def __init__(self, config):
        self.config = config
        self._path = config.data_path
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_chat_tokenize()
        self.use_dict = self.load(self._path+'/user.pkl')
        self.item_dict = self.load(self._path+'/item.pkl')
        self.feature_dict = self.load(self._path+'/feature.pkl')

        self.train_corpus = self._read_file(os.path.join(self._path, 'gl_train.pkl'))
        self.valid_corpus = self._read_file(os.path.join(self._path, 'gl_dev.pkl'))
        self.test_corpus = self._read_file(os.path.join(self._path, 'gl_test.pkl'))
        self.pretrain_corpus = self._read_file(os.path.join(self._path, 'gl_pretrain.pkl'))

        self._build_vocab(config.max_vocab_cnt)
        print("Done loading corpus")
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def _read_file(self, path):
        cdir, filename = os.path.split(path)
        filename = os.path.join(cdir,'yelp_' + filename)
        if os.path.exists(filename):
            print("Load cache from {}".format(filename))
            with open(filename, 'rb') as f:
                return pickle.load(f)
        data = self.load(path)
        res = self._process_reivews(data)
        print("Load %d sentence from %s" % (len(res),path))
        with open(filename, 'wb') as f:
            pickle.dump(res, f)
        return res

    # def _read_pretrain_file(self, path):
    #     cdir, filename = os.path.split(path)
    #     filename = os.path.join(cdir,'yelp_' + filename)
    #     if os.path.exists(filename):
    #         with open(filename, 'rb') as f:
    #             return pickle.load(f)
    #     res = []
    #     with open(path, 'rb') as f:
    #         data = pickle.load(f)

    #     for d in data:
    #         sentence = self.tokenize(d)
    #         if sentence + 2 > self.max_seq_length:
    #             continue
    #         res.append([BOS] + sentence + [EOS])

    #     with open(filename, 'wb') as f:
    #         pickle.dump(res, f)
    #     return res

    def _process_reivews(self, data):
        res = []
        # ssen = set()
        for d in data:
            # sl = len(ssen)
            # ssen.add(d[3])
            # if sl == len(ssen): # Only assign a sentence to unique feature
            #     continue
            # sen = self.tokenize(d[3])
            # if len(sen) + 2 > self.max_seq_length: # discard too long sentence
            #     continue
            sentence = [BOS] + self.tokenize(d[3]) + [EOS]
            res.append((d[0],d[1],d[2], sentence, d[-1], torch.randn(1).item()))
        return res

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for example in self.train_corpus:
            all_words.extend(example[3])
        for example in self.valid_corpus:
            all_words.extend(example[3])
        for example in self.test_corpus:
            all_words.extend(example[3])

        for example in self.pretrain_corpus:
            all_words.extend(example[3])

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus),
                 len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1],
                 float(discard_wc) / len(all_words)))

        self.vocab = [PAD, UNK, SYS, USR] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab[UNK]


    def get_corpus(self):
        return Pack(train=self.train_corpus, valid=self.valid_corpus, test=self.test_corpus, pretrain=self.pretrain_corpus)
