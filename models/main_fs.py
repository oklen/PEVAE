
from __future__ import print_function
import numpy as np
import torch
from models.dataset.corpora import PAD, EOS, EOT
from models.enc2dec.decoders import TEACH_FORCE, GEN, DecoderRNN
from models.utils import get_dekenize, experiment_name, kl_anneal_function
import os
from collections import defaultdict
import logging
from models import utt_utils

logger = logging.getLogger()


class LossManager(object):
    def __init__(self):
        self.losses = defaultdict(list)
        self.backward_losses = []
        self.tensorborad_tmp_loss = None

    def add_loss(self, loss):
        self.tensorborad_tmp_loss = None
        for key, val in loss.items():
            if val is not None and type(val) is not bool:
                self.losses[key].append(val.item())
                if self.tensorborad_tmp_loss is None:
                    self.tensorborad_tmp_loss = val.item()
                else:
                    self.tensorborad_tmp_loss += val.item()
    
    def get_tensorbord_loss(self):
        return self.tensorborad_tmp_loss

    def add_backward_loss(self, loss):
        self.backward_losses.append(loss.item())

    def clear(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def pprint(self, name, window=None, prefix=None):
        str_losses = []
        for key, loss in self.losses.items():
            if loss is None:
                continue
            avg_loss = np.average(loss) if window is None else np.average(loss[-window:])
            str_losses.append("{} {:.3f}".format(key, avg_loss))
            if 'nll' in key and 'PPL' not in self.losses:
                str_losses.append("PPL {:.3f}".format(np.exp(avg_loss)))
        if prefix:
            return "{}: {} {}".format(prefix, name, " ".join(str_losses))
        else:
            return "{} {}".format(name, " ".join(str_losses))

    def return_dict(self, window=None):
        ret_losses = {}
        for key, loss in self.losses.items():
            if loss is None:
                continue
            avg_loss = np.average(loss) if window is None else np.average(loss[-window:])
            ret_losses[key] = avg_loss.item()
            if 'nll' in key and 'PPL' not in self.losses:
                ret_losses[key.split("nll")[0] + 'PPL'] = np.exp(avg_loss).item()
        return ret_losses

    def avg_loss(self):
        return np.mean(self.losses['vae_nll'])

def adjust_learning_rate(optimizer, last_lr, decay_rate=0.5):
    lr = last_lr * decay_rate
    print('New learning rate=', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate  # all decay half
    return lr

def get_sent(model, de_tknize, data, b_id, attn=None, attn_ctx=None, stop_eos=True, stop_pad=True):
    ws = []
    attn_ws = []
    has_attn = attn is not None and attn_ctx is not None
    for t_id in range(data.shape[1]):
        # w = model.vocab[data[b_id, t_id]] vocab shold convert id to token
        try:
            w = model.vocab[data[b_id, t_id]]
        except KeyError: # OOV
            w = '<unk>'

        if has_attn:
            a_val = np.max(attn[b_id, t_id])
            if a_val > 0.1:
                a = np.argmax(attn[b_id, t_id])
                # attn_w = model.vocab[attn_ctx[b_id, a]]
            try:
                attn_w = model.vocab[data[b_id, t_id]]
            except KeyError: # OOV
                attn_w = '<unk>'
                attn_ws.append("{}({})".format(attn_w, a_val))
        if (stop_eos and w in [EOS, EOT]) or (stop_pad and w == PAD):
            if w == EOT:
                ws.append(w)
            break
        if w != PAD:
            ws.append(w)

    att_ws = "Attention: {}".format(" ".join(attn_ws)) if attn_ws else ""
    if has_attn:
        return de_tknize(ws), att_ws
    else:
        try:
            return de_tknize(ws), ""
        except:
            return " ".join(ws), ""
# def get_sent(model, de_tknize, data, b_id, attn=None, attn_ctx=None, stop_eos=True, stop_pad=True):
#     ws = []
#     attn_ws = []
#     has_attn = attn is not None and attn_ctx is not None
#     for t_id in range(data.shape[1]):
#         w = model.vocab[data[b_id, t_id]]
#         if has_attn:
#             a_val = np.max(attn[b_id, t_id])
#             if a_val > 0.1:
#                 a = np.argmax(attn[b_id, t_id])
#                 attn_w = model.vocab[attn_ctx[b_id, a]]
#                 attn_ws.append("{}({})".format(attn_w, a_val))
#         if (stop_eos and w in [EOS, EOT]) or (stop_pad and w == PAD):
#             if w == EOT:
#                 ws.append(w)
#             break
#         if w != PAD:
#             ws.append(w)

#     att_ws = "Attention: {}".format(" ".join(attn_ws)) if attn_ws else ""
#     if has_attn:
#         return de_tknize(ws), att_ws
#     else:
#         try:
#             return de_tknize(ws), ""
#         except:
#             return " ".join(ws), ""

from torch.utils.tensorboard import SummaryWriter

def train(model, train_feed, valid_feed, evaluator, config):
    gen = generate
    writer = SummaryWriter("runs/"+config.exp_name)

    patience = 2  # wait for at least 10 epoch before stop
    valid_loss_threshold = np.inf
    best_valid_loss = np.inf
    valid_loss_record = []
    learning_rate = config.init_lr
    batch_cnt = 0
    optimizer = model.get_optimizer(config)
    # agg_learning_rate = model.init_agg_optimizer(config)
    
    done_epoch = 0
    train_loss = LossManager()
    model.train()

    logger.info("**** Embedding Generation Begins ****")
    norm_bound = 300
    norm_break_cnt = 0

    train_feed.epoch_init(config, verbose=done_epoch == 0, shuffle=True)
    with torch.no_grad():
        while True:
            model.backLoss= True
            batch = train_feed.next_batch()
            if batch is None:
                break
            model(batch, mode=TEACH_FORCE)

    model.embedding_recreate() 
    logger.info("\n**** Embedding Initilized Done ****")

    optimizer.zero_grad()
    model.zero_grad()

    logger.info("\n=== Evaluating Model ===")
    # logger.info(train_loss.pprint("Train"))
    done_epoch += 1
    config.few_shot = False
    model.few_shot_mode = False

    # validation
    valid_loss, valid_resdict = validate(model, valid_feed, config, batch_cnt)

    if writer is not None:
        writer.add_scalar(
                    "Valid loss",
                    valid_loss,
                    batch_cnt // config.ckpt_step
                )
    # if 'draw_points' in config and config.draw_points:
    #     utt_utils.draw_pics(model, valid_feed, config, batch_cnt)

    # generating
    model.backLoss = False
    gen_losses = gen(model, valid_feed, config, evaluator, num_batch=config.preview_batch_num,writer=writer,cnt = batch_cnt // config.ckpt_step)

    # adjust learning rate:
    valid_loss_record.append(valid_loss)

def validate(model, valid_feed, config, batch_cnt=None, outres2file=None):
    model.eval()
    with torch.no_grad():
        valid_feed.epoch_init(config, shuffle=False, verbose=True)
        losses = LossManager()
        valid_batch_cnt = 0
        while True:
            # if valid_batch_cnt > config.max_valid_batch_count:
            #     break
            batch = valid_feed.next_batch()
            if batch is None:
                break
            loss = model(batch, mode=GEN)
            losses.add_loss(loss)
            # losses.add_backward_loss(model.model_sel_loss(loss, batch_cnt))

        valid_loss = losses.avg_loss()
        # if outres2file:
        #     outres2file.write(losses.pprint(valid_feed.name))
        #     outres2file.write("\n")
        #     outres2file.write("Total valid loss {}".format(valid_loss))

        logger.info(losses.pprint(valid_feed.name))
        logger.info("Total valid loss {}".format(valid_loss))

        res_dict = losses.return_dict()
    return valid_loss, res_dict

from nltk.util import ngrams
from rouge.rouge import rouge_n_sentence_level


import collections
import math
import os
from multiprocessing import Pool

import nltk
from nltk.translate.bleu_score import SmoothingFunction
from abc import abstractmethod

class Metrics:
    def __init__(self):
        self.name = 'Metric'

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    @abstractmethod
    def get_score(self):
        pass

class SelfBleu(Metrics):
    def __init__(self, test_text='', gram=3):
        super().__init__()
        self.name = 'Self-Bleu'
        self.test_data = test_text
        self.gram = gram
        self.sample_size = 500
        self.reference = None
        self.is_first = True

    def get_name(self):
        return self.name

    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.get_bleu_fast()
        return self.get_bleu_parallel()

    def get_reference(self):
        return self.reference
        # if self.reference is None:
        #     reference = list()
        #     with open(self.test_data) as real_data:
        #         for text in real_data:
        #             text = nltk.word_tokenize(text)
        #             reference.append(text)
        #     self.reference = reference
        #     return reference
        # else:
        #     return self.reference

    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        with open(self.test_data) as test_data:
            for hypothesis in test_data:
                hypothesis = nltk.word_tokenize(hypothesis)
                bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                                    smoothing_function=SmoothingFunction().method1))
        return sum(bleu) / len(bleu)

    def calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def get_bleu_fast(self):
        reference = self.get_reference()
        # random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self.get_bleu_parallel(reference=reference)

    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        result = list()
        sentence_num = len(reference)
        for index in range(sentence_num):
            hypothesis = reference[index]
            other = reference[:index] + reference[index+1:]
            result.append(pool.apply_async(self.calc_bleu, args=(other, hypothesis, weight)))

        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt

def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.
  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.
  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts

def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
  """Computes BLEU score of translated segments against one or more references.
  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.
  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    reference_length += min(len(r) for r in references)
    translation_length += len(translation)

    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
        merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    translation_ngram_counts = _get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    for order in range(1, max_order+1):
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order-1] += possible_matches

  precisions = [0] * max_order
  for i in range(0, max_order):
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) /
                       (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) /
                         possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  if min(precisions) > 0:
    p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
    geo_mean = math.exp(p_log_sum)
  else:
    geo_mean = 0

  ratio = float(translation_length) / reference_length

  if ratio > 1.0:
    bp = 1.
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  return (bleu, precisions, bp, ratio, translation_length, reference_length)

def calculate_blue(r,h,n):
    rg = list(ngrams(r,n))
    rh = list(ngrams(h,n))
    sums = max(len(rh),1)
    up = 0
    dic = {}
    for tmp in rh:
        if dic.get(tmp) is None:
            dic[tmp] = 1
        else:
            dic[tmp] += 1
    for tmp in rg:
        if dic.get(tmp) is not None and dic[tmp]!=0:
            dic[tmp] -= 1
            up += 1
    return up / sums
    


def generate(model, data_feed, config, evaluator, num_batch=1, dest_f=None, writer=None, cnt=None):
    model.eval()
    with torch.no_grad():
        de_tknize = get_dekenize()

        # def write(msg):
        #     if msg is None or msg == '':
        #         return
        #     if dest_f is None:
        #         logger.info(msg)
        #     else:
        #         dest_f.write(msg + '\n')
        if config.embedding_record and writer is not None:
            writer.add_embedding(model.user_embedding.weight, metadata=model.user_list, tag='user', global_step=cnt)
            writer.add_embedding(model.item_embedding.weight, metadata=model.item_list, tag='item', global_step=cnt)
            writer.flush()

        data_feed.epoch_init(config, shuffle=True, verbose=False)
        evaluator.initialize()
        logger.info("Generation Begin.")
        blue_scores = []
        blue_scores4 = []

        rouge_p = []
        rouge_r = []
        rouge_f1 = []

        rouge2_p = []
        rouge2_r = []
        rouge2_f1 = []
        ratings = []

        # sb = []
        # for i in range(6):
        #     sb.append(SelfBleu(gram=i))
        while True:
            batch = data_feed.next_batch()
            if batch is None or (num_batch is not None
                                and data_feed.ptr > num_batch):
                break
            if config.do_pred:
                outputs, labels, rating = model(batch, mode=GEN, gen_type=config.gen_type)
                ratings.append(rating-batch['ratings'])
            else:
                outputs, labels = model(batch, mode=GEN, gen_type=config.gen_type)
            # move from GPU to CPU
            labels = labels.cpu()
            pred_labels = [t.cpu().data.numpy() for t in
                        outputs[DecoderRNN.KEY_SEQUENCE]]
            pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0,1)
            true_labels = labels.data.numpy()

            # get attention if possible
            # if config.use_attn or config.use_ptr:
            #     pred_attns = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_ATTN_SCORE]]
            #     pred_attns = np.array(pred_attns, dtype=float).squeeze(2).swapaxes(0,1)
            # else:
            #     pred_attns = None

            pred_attns = None
            # get last 1 context
            # ctx = None
            # ctx = batch.get('contexts')
            # ctx_len = batch.get('context_lens')
            # domains = batch.domains

            # logger.info the batch in String.
            preds = []
            trues = []

            for b_id in range(pred_labels.shape[0]):
                pred_str, attn = get_sent(model, de_tknize, pred_labels, b_id, attn=pred_attns)
                true_str, _ = get_sent(model, de_tknize, true_labels, b_id)


                pred = pred_str.split()
                true = true_str.split()
                preds.append(tuple(pred))
                trues.append([tuple(true)])

                # blue_scores.append(compute_bleu([true],pred,1))
                # blue_scores4.append(compute_bleu([true],pred,4))

                # blue_scores.append(calculate_blue(true,pred,1))
                # blue_scores4.append(calculate_blue(true,pred,4))

                p,r,f = rouge_n_sentence_level(pred,true,1)
                rouge_p.append(p)
                rouge_r.append(r)
                rouge_f1.append(f)

                p,r,f = rouge_n_sentence_level(pred,true,2)
                rouge2_p.append(p)
                rouge2_r.append(r)
                rouge2_f1.append(f)

                # prev_ctx = ""
                # if ctx is not None:
                #     ctx_str, _ = get_sent(model, de_tknize, ctx[:, ctx_len[b_id]-1, :], b_id)
                #     prev_ctx = "Source: {}".format(ctx_str)

                # domain = domains[b_id]
                # evaluator.add_example(true_str, pred_str, domain)
                # if num_batch is None or num_batch <= 2:
                #     write(prev_ctx)
                #     write("True: {} ||| Pred: {}".format(true_str, pred_str))
                #     if attn:
                #         write("[[{}]]".format(attn))

        # write(evaluator.get_report(include_error=dest_f is not None))
        # bleu_1 = np.mean(blue_scores)
        # bleu_4 = np.mean(blue_scores4)
        bleu_1, _, _, _, _, _ = compute_bleu(trues, preds, 1, False)
        bleu_4, _, _, _, _, _ = compute_bleu(trues, preds, 4, False)

        rouge_f1 = np.mean(rouge_f1)
        rouge2_f1 = np.mean(rouge2_f1)

        logger.info("BLUE-1:{} ROUGE1-P:{} ROUGE1-R:{} ROUGE1-F1:{}".format(bleu_1 * 100,
        np.mean(rouge_p)*100,np.mean(rouge_r)*100, rouge_f1*100))
        logger.info("BLUE-4:{} ROUGE2-P:{} ROUGE2-R:{} ROUGE2-F1:{}".format(bleu_4 * 100,
        np.mean(rouge2_p)*100,np.mean(rouge2_r)*100, rouge2_f1*100))

        if writer is not None:
            writer.add_scalar("BLEU-1", bleu_1 * 100, cnt)
            writer.add_scalar("BLEU-4", bleu_4 * 100, cnt)
            writer.add_scalar("ROUGE1-F1", rouge_f1 * 100, cnt)
            writer.add_scalar("ROUGE2-F1", rouge2_f1 * 100, cnt)
            # for i in range(2,6):
            #     scores = sb[i].get_bleu_parallel(preds)
            #     writer.add_scalar("S-BLEU-"+str(i), scores, cnt)
            #     logger.info('S-BLEU-{}:{}'.format(i,scores))
            if config.do_pred:
                ratings = torch.cat(ratings,-1)
                rmse = np.sqrt(torch.mean(ratings*ratings).cpu())
                mae = torch.mean(torch.abs(ratings)).cpu()
                logger.info("RMSE:{} MAE:{}".format(rmse, mae))
                writer.add_scalar("RMSE", rmse, cnt)
                writer.add_scalar("MAE", mae, cnt)

        logger.info("Generation Done")



