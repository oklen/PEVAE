# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
#from models.model_bases import summary
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
    agg_learning_rate = model.init_agg_optimizer(config)
    
    done_epoch = 0
    train_loss = LossManager()
    model.train()

    #logger.info(summary(model, show_weights=False))
    logger.info("**** Training Begins ****")
    logger.info("**** Epoch 0/{} ****".format(config.max_epoch))
    norm_bound = 300
    norm_break_cnt = 0

    while True:
        if done_epoch > 100: #Activate break model training
            break
        train_feed.epoch_init(config, verbose=done_epoch == 0, shuffle=True)
        while True:
            model.backLoss= True
            batch = train_feed.next_batch()
            if batch is None:
                break

            optimizer.zero_grad()
            model.zero_grad()
            loss = model(batch, mode=GEN)
            model.backward(batch_cnt, loss, step=batch_cnt)
            
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, norm_type=2)
            
            # if torch.sum(norm) > norm_bound or torch.isnan(torch.sum(norm)): #Skip wrong point
            if torch.sum(norm) > norm_bound: 
                norm_break_cnt += 1
                print("Skip Norm:{}".format(torch.sum(norm)))
                if norm_break_cnt >= 5:
                    norm_bound *= 1.2
                    print("adjust norm bound to {}".format(norm_bound))
                    norm_break_cnt = 0
                continue
            else:
                norm_break_cnt = max(0,norm_break_cnt - 1)

            writer.add_scalar(
                            "Norm",
                            torch.sum(norm),
                            batch_cnt
                        )

            optimizer.step()
            model.scheduler.step()
            train_loss.add_loss(loss)

            writer.add_scalar(
                            "Loss",
                            train_loss.get_tensorbord_loss(),
                            batch_cnt
                        )

            batch_cnt += 1

            if batch_cnt % config.print_step == 0:
                logger.info(train_loss.pprint("Train", window=config.print_step,
                                              prefix="{}/{}-({:.3f})".format(batch_cnt % (config.ckpt_step+1),
                                                                         config.ckpt_step,
                                                                         model.kl_w)))

            if batch_cnt % config.ckpt_step == 0:
                optimizer.zero_grad()
                model.zero_grad()

                logger.info("\n=== Evaluating Model ===")
                logger.info(train_loss.pprint("Train"))
                done_epoch += 1

                # validation
                valid_loss, valid_resdict = validate(model, valid_feed, config, batch_cnt)
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
                if config.lr_decay and learning_rate > 1e-6 and valid_loss > best_valid_loss and len(
                        valid_loss_record) - valid_loss_record.index(best_valid_loss) >= config.lr_hold and done_epoch >= config.teach_force_bound:
                    learning_rate = adjust_learning_rate(optimizer, learning_rate, config.lr_decay_rate)
                    agg_learning_rate = adjust_learning_rate(model.agg_optimizer, agg_learning_rate, config.lr_decay_rate)
                    logger.info("Adjust learning rete to {}".format(learning_rate))

                    # logger.info("Reloading the best model.")
                    # model.load_state_dict(torch.load(os.path.join(config.session_dir, "model")))

                # update early stopping stats
                if valid_loss < best_valid_loss:
                    if valid_loss <= valid_loss_threshold * config.improve_threshold:
                        patience = max(patience,
                                       done_epoch * config.patient_increase)
                        valid_loss_threshold = valid_loss
                        logger.info("Update patience to {}".format(patience))

                    # if config.save_model:
                    #     logger.info("Model Saved.")
                    #     torch.save(model.state_dict(),
                    #                os.path.join(config.session_dir, "model"))

                    best_valid_loss = valid_loss
                # exit eval model
                model.train()
                train_loss.clear()
                logger.info("\n**** Epoch {}/{} ****".format(done_epoch,
                                                       config.max_epoch))

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
# from rouge.rouge import rouge_n_sentence_level


import collections
import math


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
        data_feed.epoch_init(config, shuffle=True, verbose=False)
        evaluator.initialize()
        logger.info("Generation Begin.")
        ratings = []

        while True:
            batch = data_feed.next_batch()
            if batch is None or (num_batch is not None
                                and data_feed.ptr > num_batch):
                break
            rating = model(batch, mode=GEN, gen_type=config.gen_type)
            ratings.append(rating-batch['ratings'])
            # move from GPU to CPU

        if writer is not None:
            ratings = torch.cat(ratings,-1)
            rmse = np.sqrt(torch.mean(ratings*ratings).cpu())
            mae = torch.mean(torch.abs(ratings)).cpu()
            logger.info("RMSE:{} MAE:{}".format(rmse, mae))
            writer.add_scalar("RMSE", rmse, cnt)
            writer.add_scalar("MAE", mae, cnt)

        logger.info("Generation Done")

