
from __future__ import print_function
import numpy as np
# from models.models.model_bases import summary
import torch
from models.dataset.corpora import PAD, EOS, EOT
from models.enc2dec.decoders import TEACH_FORCE, GEN, DecoderRNN
from models.utils import get_dekenize, experiment_name, kl_anneal_function
import os
from collections import defaultdict
import logging
from models import utt_utils

logger = logging.getLogger()

def r0():
    return 0
def rs():
    return []

class LossManager(object):
    def __init__(self):
        self.losses = defaultdict(list)
        self.backward_losses = []
        self.tensorborad_tmp_loss = None

    def add_loss(self, loss):
        self.tensorborad_tmp_loss = None
        for key, val in loss.items():
            if val is not None and type(val) is not bool and type(val) is not int:
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
    gen = generate3
    writer = SummaryWriter("runs/"+config.exp_name)

    patience = 2  # wait for at least 10 epoch before stop
    valid_loss_threshold = np.inf
    best_valid_loss = np.inf
    valid_loss_record = []
    learning_rate = config.init_lr
    batch_cnt = 0
    optimizer = model.get_optimizer(config)

    best_model_name = os.path.join('./runs/','model_' + config.exp_name + '.pt')
    # best_model_name = os.path.join('./runs/', config.exp_name , 'model.pt') 

    # agg_learning_rate = model.init_agg_optimizer(config)
    
    done_epoch = 0
    train_loss = LossManager()
    model.train()

    # logger.info(summary(model, show_weights=False))
    logger.info("**** Training Begins ****")
    logger.info("**** Epoch 0/{} ****".format(config.max_epoch))
    norm_bound = 300
    norm_break_cnt = 0

    avger_norm = None
    rkl_list = []
    jump_cnt = 0
    if config.o2_follow:
        scheduler2, optimizer2 = model.get_o2_optimizer(config)
    model.E_steps = config.freeze_step
    final_epoch = 0
    
    while True:
        if done_epoch > 100: #Activate break model training
            break
        train_feed.epoch_init(config, shuffle=True)
        while True:
            model.backLoss= True
            batch = train_feed.next_batch()
            if batch is None:
                break
            optimizer.zero_grad()
            model.zero_grad()

                # Draw pics:
                # print("Draw pics!")
                # utt_utils.draw_pics(model, test_feed, config, -1, num_batch=5, shuffle=True, add_text=False)  # (num_batch * 50) points
            
            model.steps = batch_cnt
            if done_epoch < config.teach_force_bound:
                loss = model(batch, mode=TEACH_FORCE)
            else:
                loss = model(batch, mode=GEN)

            model.backward(batch_cnt, loss, step=batch_cnt)

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, norm_type=2)
            norm_sum = torch.sum(norm)

            if avger_norm is None:
                avger_norm = norm_sum

            if (torch.isinf(norm_sum) or torch.isnan(norm_sum)) and jump_cnt < 5:
                logger.info('inf or nan encouter!')
                jump_cnt += 1
                continue
            jump_cnt = 0

            if avger_norm * 3 < norm_sum: #Skip wrong point
                logger.info("Jump Norm:{}".format(norm_sum))
                avger_norm = (avger_norm) / 20 * 19 + norm_sum / 20
                continue

            # avger_norm = (avger_norm * batch_cnt + norm_sum) / (batch_cnt+1)
            avger_norm = avger_norm / (batch_cnt + 1) * batch_cnt + norm_sum / (batch_cnt + 1)

            if batch_cnt is not None and batch_cnt >= config.freeze_step and not model.restarted:
                logger.info('Switch to another train!')
                best_valid_loss = np.inf
                # config.ckpt_step = 1 # Stop Training and only do training
                try:
                    model.load_state_dict(torch.load(best_model_name), strict=False)
                    logger.info('Model {}: Load Done!'.format(best_model_name))
                except:
                    logger.info('Warnning: Model Load Failure ! {}:{}'.format(Error))
                model.restarted = True

                for param in model.x_embedding.parameters():
                    param.requires_grad = False
                for param in model.x_encoder.parameters():
                    param.requires_grad = False
                # for param in model.xtz.parameters():
                #     param.requires_grad = False
                for param in model.x_decoder.parameters():
                    param.requires_grad = False
                for param in model.x_init_connector.parameters():
                    param.requires_grad = False


                # for param in model.user_embedding.parameters():
                #     param.requires_grad = False
                # for param in model.item_embedding.parameters():
                #     param.requires_grad = False
                # for param in model.rating_embedding.parameters():
                #     param.requires_grad = False

                optimizer = model.get_freeze_optimizer(config)
                learning_rate = config.init_lr
                # for param_group in optimizer.param_groups:  # recover to the initial learning rate
                #     param_group['lr'] = config.init_lr
                batch_cnt = 0
                avger_norm = None
                # print("BEGIN INITAL Testing!")
                multi_sample_generate(model, valid_feed, config, evaluator, num_batch=config.preview_batch_num,writer=writer,cnt = batch_cnt // config.ckpt_step)
                gen(model, valid_feed, config, evaluator, num_batch=config.preview_batch_num,writer=writer,cnt = batch_cnt // config.ckpt_step)
                continue

            optimizer.step()
            model.scheduler.step()
            # if config.do_follow:
            #     for i in range(config.optim_times):
            #         agg_loss = model.agg_optim(batch, mode=GEN,optimizer=optimizer) # Do agg_optim here
            #     loss['vae_rkl'] = agg_loss
            train_loss.add_loss(loss)

            if writer is not None:
                writer.add_scalar(
                                "Norm",
                                torch.sum(norm),
                                batch_cnt
                            )
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
                if config.o2_follow and not model.restarted:
                    logger.info('rkl_loss:{}'.format(np.mean(rkl_list)))
                    rkl_list.clear()

            if batch_cnt % config.ckpt_step == 0:
                optimizer.zero_grad()
                model.zero_grad()

                logger.info("\n=== Evaluating Model ===")
                logger.info(train_loss.pprint("Train"))
                done_epoch += 1

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
                if not config.ost and not config.no_gen:
                    if model.restarted:
                        final_epoch += 1
                        logger.info('Average q(y|z) var: {}'.format(torch.mean(torch.tensor(model.sup_decoder.var_list).nan_to_num_())))
                        logger.info('Average q(x|z) var: {}'.format(torch.mean(torch.tensor(model.sup_decoder.x_var_list).nan_to_num_())))
                        logger.info('Average KL distance: {}'.format(torch.mean(torch.tensor(model.sup_decoder.dis_list).nan_to_num_())))
                        logger.info('Average center distance: {}'.format(torch.mean(torch.tensor(model.sup_decoder.x_y_dist_list).nan_to_num_())))
                        model.sup_decoder.var_list.clear()
                        model.sup_decoder.dis_list.clear()
                        model.sup_decoder.x_y_dist_list.clear()
                        model.sup_decoder.x_var_list.clear()

                        if final_epoch >= 5:
                            done_epoch += 100
                        if config.enable_multi_sample and final_epoch >= 4:
                            multi_sample_generate(model, valid_feed, config, evaluator, num_batch=config.preview_batch_num,writer=writer,cnt = batch_cnt // config.ckpt_step)
                        else:
                            gen_losses = gen(model, valid_feed, config, evaluator, num_batch=config.preview_batch_num,writer=writer,cnt = batch_cnt // config.ckpt_step)


                # adjust learning rate:
                valid_loss_record.append(valid_loss)
                if  learning_rate > 1e-6 and valid_loss > best_valid_loss * 0.99:
                    olr = learning_rate
                    learning_rate = adjust_learning_rate(optimizer, learning_rate, config.lr_decay_rate)
                    if config.o2_follow and not model.restarted:
                        adjust_learning_rate(optimizer2, olr, config.lr_decay_rate)
                    logger.info("Adjust learning rete to {}".format(learning_rate))

                    # logger.info("Reloading the best model.")
                    # model.load_state_dict(torch.load(os.path.join(config.session_dir, "model")))

                # update early stopping stats
                if valid_loss <= valid_loss_threshold * config.improve_threshold:
                    patience = max(patience,
                                    done_epoch * config.patient_increase)
                    valid_loss_threshold = valid_loss
                    logger.info("Update patience to {}".format(patience))
                    if not model.restarted:
                        torch.save(model.state_dict(),best_model_name)
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
        valid_feed.epoch_init(config, shuffle=True, verbose=True)
        losses = LossManager()
        valid_batch_cnt = 0
        while True:
            # if valid_batch_cnt > config.max_valid_batch_count:
            #     break
            batch = valid_feed.next_batch()
            if batch is None:
                break
            loss = model(batch, mode=TEACH_FORCE)
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
# from rouge.rouge import rouge_n_sentence_level,rouge_l_sentence_level,rouge_w_sentence_level

from nlgeval import NLGEval

import collections
import math
import os
from multiprocessing import Pool

import nltk
from nltk.translate.bleu_score import SmoothingFunction
from abc import abstractmethod
import pickle

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
    

import copy

def calculate_similiar(embds, name = None, config = None):
    if name is not None:
        with open(config.exp_name + '_' + name + '.pkl','wb') as f:
            pickle.dump(embds, f)
    weight = embds.weight.detach()
    mean_emb = torch.mean(weight, 0)
    mean_emb = mean_emb / torch.sqrt(torch.sum(mean_emb*mean_emb))
    tmp_embds = weight
    tmp_embds = tmp_embds / torch.sqrt(torch.sum(tmp_embds * tmp_embds, 1)).unsqueeze(1)
    return torch.mean(torch.sum(tmp_embds * mean_emb.unsqueeze(0),1))

def generate2(model, data_feed, config, evaluator, num_batch=1, dest_f=None, writer=None, cnt=None):
    model.eval()
    if model.restarted:
        print('User Embedding:',calculate_similiar(model.user_embedding, 'user_embedding', config))
        print('Item Embedding:',calculate_similiar(model.item_embedding, 'item_embedding', config))
    my_preds = []
    my_test = []

    with torch.no_grad():
        de_tknize = get_dekenize()

        data_feed.epoch_init(config, shuffle=True, verbose=True)
        evaluator.initialize()

        logger.info("Generation Begin.")
        ratings = []

        show_up = 0
        uir_reference = defaultdict(list)
        uir_predict = dict()

        nb_blue_1, nb_blue_4, nb_rouge_1, nb_rouge_2 = [],[],[],[]
        nb_rouge_1p,nb_rouge_1r,nb_rouge_2p,nb_rouge_2r = [],[],[],[]
        nb_rouge_lp,nb_rouge_lr,nb_rouge_lf = [],[],[]
        nb_meter,nb_cider,nb_tcs = [],[],[]
        done_batch = 0
        sample_size = 5

        while True:
            batch = data_feed.next_batch()
            if batch is None:
                break

            bleu_1 = []
            bleu_4 = []

            rouge_p = []
            rouge_r = []
            rouge_f1 = []

            rouge2_p = []
            rouge2_r = []
            rouge2_f1 = []

            rougel_p = []
            rougel_r = []
            rougel_f1 = []
            meteor = []
            cider = []
            tcs = []
            t_preds = []
            t_valid = []

            for i in range(sample_size):
                if config.do_pred:
                    outputs, labels, rating = model(batch, mode=GEN, gen_type=config.gen_type)
                    ratings.append(rating-batch['ratings'])
                else:
                    outputs, labels = model(batch, mode=GEN, gen_type=config.gen_type)

                batch['rd_ids'] = torch.randn_like(batch['rd_ids']) # replace rd_ids to multi sample

                # move from GPU to CPU
                labels = labels.cpu()
                pred_labels = [t.cpu().data.numpy() for t in
                            outputs[DecoderRNN.KEY_SEQUENCE]]

                pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0,1)
                true_labels = labels.data.numpy()

                pred_attns = None
                

                preds = []
                trues = []

                for b_id in range(pred_labels.shape[0]):
                    pred_str, attn = get_sent(model, de_tknize, pred_labels, b_id, attn=pred_attns)
                    true_str, _ = get_sent(model, de_tknize, true_labels, b_id)

                    if show_up < 10:
                        print('pred:',pred_str)
                        print('true:',true_str)
                        show_up += 1


                    pred = pred_str.split()
                    true = true_str.split()

                    preds = tuple(pred)
                    trues = [tuple(true)]

                    # id_now = str(batch['user_ids'][b_id]) + str(batch['item_ids'][b_id]) + str(batch['ratings'][b_id])
                    # uir_reference[id_now].append(tuple(true))
                    # uir_predict[id_now] = tuple(pred)
                    t_preds.append(pred_str)
                    t_valid.append(true_str)

                    p,r,f = rouge_n_sentence_level(pred,true,1)
                    rouge_p.append(p)
                    rouge_r.append(r)
                    rouge_f1.append(f)

                    p,r,f = rouge_n_sentence_level(pred,true,2)
                    rouge2_p.append(p)
                    rouge2_r.append(r)
                    rouge2_f1.append(f)

                    p,r,f = rouge_l_sentence_level(pred,true)
                    rougel_p.append(p)
                    rougel_r.append(r)
                    rougel_f1.append(f)


                    bleu_scores, _, _, _, _, _ = compute_bleu(trues, preds, 1, False)
                    bleu_1.append(bleu_scores)

                    bleu_scores, _, _, _, _, _ = compute_bleu(trues, preds, 4, False)
                    bleu_4.append(bleu_scores)

            batch_size = batch['rd_ids'].size(0)
            bleu_1 = tuple(bleu_1)
            bleu_4 = tuple(bleu_4)
            rouge_r = tuple(rouge_r)
            rouge_p = tuple(rouge_p)
            rouge_f1 = tuple(rouge_f1)
            rouge2_r = tuple(rouge2_r)
            rouge2_p = tuple(rouge_p)
            rouge2_f1 = tuple(rouge2_f1)
            rougel_p = tuple(rougel_p)
            rougel_r = tuple(rougel_r)
            rougel_f1 = tuple(rougel_f1)

            # if model.restarted:
            #     meteor = tuple(meteor)
            #     cider = tuple(cider)
            #     tcs = tuple(tcs)

            for i in range(batch_size):
                b1,b4,r1,p1,f1,r2,p2,f2,rl,pl,fl = None,None,None,None,None,None,None,None,None,None,None
                me,ci,tc = None,None,None
                best_pred = None
                best_valid = None
                
                for j in range(sample_size):
                    if b1 == None or (fl < rougel_f1[i+j*batch_size]):
                        b1 = bleu_1[i+j*batch_size]
                        b4 = bleu_4[i+j*batch_size]
                        r1  = rouge_r[i+j*batch_size]
                        p1 = rouge_p[i+j*batch_size]
                        f1 = rouge_f1[i+j*batch_size]
                        r2 = rouge2_r[i+j*batch_size]
                        p2 = rouge2_p[i+j*batch_size]
                        f2 = rouge2_f1[i+j*batch_size]
                        rl = rougel_r[i+j*batch_size]
                        pl = rougel_p[i+j*batch_size]
                        fl = rougel_f1[i+j*batch_size]
                        best_pred = t_preds[i+j*batch_size]
                        best_valid = t_valid[i+j*batch_size]
                        # if model.restarted:
                        #     me = meteor[i+j*batch_size]
                        #     ci = cider[i+j*batch_size]
                        #     tc = tcs[i+j*batch_size]

                nb_blue_1.append(b1)
                nb_blue_4.append(b4)
                nb_rouge_1.append(f1)
                nb_rouge_2.append(f2)
                nb_rouge_1r.append(r1)
                nb_rouge_1p.append(p1)
                nb_rouge_2r.append(r2)
                nb_rouge_2p.append(p2)
                nb_rouge_lr.append(rl)
                nb_rouge_lp.append(pl)
                nb_rouge_lf.append(fl)
                my_preds.append(best_pred)
                my_test.append(best_valid)
                
            done_batch+=1

        uir_ref_list = []
        uir_pred_list = []

        rouge_f1 = np.mean(nb_rouge_1)
        rouge2_f1 = np.mean(nb_rouge_2)
        bleu_1 = np.mean(nb_blue_1)
        bleu_4 = np.mean(nb_blue_4)
        rouge_p1 = np.mean(nb_rouge_1p)
        rouge_r1 = np.mean(nb_rouge_1r)
        rouge_p2 = np.mean(nb_rouge_2p)
        rouge_r2 = np.mean(nb_rouge_2r)
        rouge_lr = np.mean(nb_rouge_lr)
        rouge_lp = np.mean(nb_rouge_lp)
        rouge_lf = np.mean(nb_rouge_lf)

        if model.restarted:
            nlgeval = NLGEval()  # loads the models
            result = nlgeval.compute_metrics([my_test],my_preds)
            me = result['METEOR']
            cider = result['CIDEr']
            tcs = result['SkipThoughtCS']
            # me = cider = tcs = 0
            from bleurt import score
            cp = "./bleurt-base-128"
            ref = my_test
            cand = my_preds
            scorer = score.BleurtScorer(cp)
            print('bleurt',np.mean(scorer.score(references=ref, candidates=cand)))

        logger.info("BLUE-1:{} ROUGE1-P:{} ROUGE1-R:{} ROUGE1-F1:{}".format(bleu_1 * 100,
        rouge_p1*100, rouge_r1*100, rouge_f1*100))
        logger.info("BLUE-4:{} ROUGE2-P:{} ROUGE2-R:{} ROUGE2-F1:{}".format(bleu_4 * 100,
        rouge_p2*100, rouge_r2*100, rouge2_f1*100))
        logger.info("ROUGE-L-R:{} ROUGE-L-P:{} ROUGE-L-F:{}".format(rouge_lr * 100,
        rouge_lp*100, rouge_lf*100))

        if model.restarted:
            logger.info("METER:{} CIDEr:{} SkipThoughtCS:{}".format(me*100,
            cider*100, tcs*100))

        if writer is not None:
            writer.add_scalar("BLEU-1", bleu_1 * 100, cnt)
            writer.add_scalar("BLEU-4", bleu_4 * 100, cnt)
            writer.add_scalar("ROUGE1-F1", rouge_f1 * 100, cnt)
            writer.add_scalar("ROUGE2-F1", rouge2_f1 * 100, cnt)

        logger.info("Generation Done")


def reformat_reference(ref,mx):
    res = [[] for i in range(mx)]
    for index,item in enumerate(ref):
        for i in range(mx):
            try:
                res[i].append(item[index])
            except:
                res[i].append("BLEU")
    return res

def generate3(model, data_feed, config, evaluator, num_batch=1, dest_f=None, writer=None, cnt=None):
    model.eval()
    model.backLoss = False
    # if model.restarted:
    #     print('User Embedding:',calculate_similiar(model.user_embedding, 'user_embedding', config))
    #     print('Item Embedding:',calculate_similiar(model.item_embedding, 'item_embedding', config))

    with torch.no_grad():
        de_tknize = get_dekenize()

        data_feed.epoch_init(config, shuffle=True, verbose=True)
        evaluator.initialize()

        logger.info("Generation Begin.")
        ratings = []

        show_up = 0
        
        max_ref = 0
        my_reference = defaultdict(rs)
        my_predict = dict()

        done_batch = 0
        
        while True:
            batch = data_feed.next_batch()
            if batch is None:
                break


            if config.do_pred:
                outputs, labels, rating = model(batch, mode=GEN, gen_type=config.gen_type)
                ratings.append(rating-batch['ratings'])
            else:
                outputs, labels = model(batch, mode=GEN, gen_type=config.gen_type)

            batch['rd_ids'] = torch.randn_like(batch['rd_ids']) # replace rd_ids to multi sample
            ui_ids = batch['ui_ids'].tolist()

            # move from GPU to CPU
            labels = labels.cpu()
            pred_labels = [t.cpu().data.numpy() for t in
                        outputs[DecoderRNN.KEY_SEQUENCE]]

            pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0,1)
            true_labels = labels.data.numpy()

            pred_attns = None
            

            preds = []
            trues = []
            
            for b_id,ui_id_now in zip(range(pred_labels.shape[0]),ui_ids):
                pred_str, attn = get_sent(model, de_tknize, pred_labels, b_id, attn=pred_attns)
                true_str, _ = get_sent(model, de_tknize, true_labels, b_id)

                if show_up < 10:
                    print('pred:',pred_str)
                    print('true:',true_str)
                    show_up += 1
                    
                my_predict[ui_id_now] = pred_str
                my_reference[ui_id_now].append(true_str)
                
                pred = pred_str.split()
                true = true_str.split()

                preds = tuple(pred)
                trues = [tuple(true)]

                # p,r,f = rouge_n_sentence_level(pred,true,1)
                # rouge_p.append(p)
                # rouge_r.append(r)
                # rouge_f1.append(f)

                # p,r,f = rouge_n_sentence_level(pred,true,2)
                # rouge2_p.append(p)
                # rouge2_r.append(r)
                # rouge2_f1.append(f)

                # p,r,f = rouge_l_sentence_level(pred,true)
                # rougel_p.append(p)
                # rougel_r.append(r)
                # rougel_f1.append(f)



            # if model.restarted:
            #     meteor = tuple(meteor)
            #     cider = tuple(cider)
            #     tcs = tuple(tcs)
        fin_reference,fin_predict,fin_id = [],[],[]
        for k,v in my_predict.items():
            fin_predict.append(v)
            fin_id.append(k)
            fin_reference.append(my_reference[k])
            max_ref = max(max_ref, len(my_reference[k]))
            
        print("Get max_reference:", max_ref)
        # bleu_scores, _, _, _, _, _ = compute_bleu(fin_reference, fin_predict, 1, False)
        # bleu_1 = bleu_scores

        # bleu_scores, _, _, _, _, _ = compute_bleu(fin_reference, fin_predict, 4, False)
        # bleu_4 = bleu_scores
        
        rouge1_p,rouge1_r,rouge1_f,rouge2_p,rouge2_r,rouge2_f,rougel_r,rougel_p,rougel_f,rougew_r,rougew_p,rougew_f = [],[],[],[],[],[],[],[],[],[],[],[]
        
        def rouge_score(refs, pred, func, i = None):
            p = []
            r = []
            f = []
            for one_ref in refs:
                if i is not None:
                    tp,tr,tf = func(pred.split(), one_ref.split(), i)
                else:
                    tp,tr,tf = func(pred.split(), one_ref.split())
                p.append(tp)
                r.append(tr)
                f.append(tf)
            return np.mean(p),np.mean(r),np.mean(f)
        
        for ref,pred in zip(fin_reference, fin_predict):
        #     last_f1  = 0
        #     last_sentence = None
        #     for one_ref in ref:
        #         p,r,f = rouge_n_sentence_level(pred.split(),one_ref.split(),1)
        #         if f >= last_f1:
        #             last_f1 = f
        #             last_sentence = pred
            # p,r,f = rouge_n_sentence_level(pred.split(),last_sentence.split(),1)
            # p,r,f = rouge_n_sentence_level(pred.split(),last_sentence.split(),1)
            p,r,f = rouge_score(ref,pred,rouge_n_sentence_level,1)
            rouge1_p.append(p)
            rouge1_r.append(r)
            rouge1_f.append(f)
            
            # last_f1  = 0
            # for one_ref in ref:
            #     p,r,f = rouge_l_sentence_level(pred.split(),one_ref.split())
            #     if f >= last_f1:
            #         last_f1 = f
            #         last_sentence = pred
            # p,r,f = rouge_l_sentence_level(pred.split(),last_sentence.split())
            p,r,f = rouge_score(ref,pred,rouge_l_sentence_level)
            rougel_p.append(p)
            rougel_r.append(r)
            rougel_f.append(f)
        

            # last_f1  = 0
            # for one_ref in ref:
            #     p,r,f = rouge_n_sentence_level(pred.split(),one_ref.split(),2)
            #     if f >= last_f1:
            #         last_f1 = f
            #         last_sentence = pred
            # p,r,f = rouge_n_sentence_level(pred.split(),last_sentence.split(),2)
            p,r,f = rouge_score(ref,pred,rouge_n_sentence_level,2)
            rouge2_p.append(p)
            rouge2_r.append(r)
            rouge2_f.append(f)
            
            # last_f1  = 0
            # for one_ref in ref:
            #     p,r,f = rouge_w_sentence_level(pred.split(),one_ref.split())
            #     if f >= last_f1:
            #         last_f1 = f
            #         last_sentence = pred
            # p,r,f = rouge_w_sentence_level(pred.split(),last_sentence.split())
            
            p,r,f = rouge_score(ref,pred,rouge_w_sentence_level)
            rougew_p.append(p)
            rougew_r.append(r)
            rougew_f.append(f)
            
        def gm(scores):
            res = []
            for s in scores:
                res.append(np.mean(s))
            return res
        rouge1_p,rouge1_r,rouge1_f,rouge2_p,rouge2_r,rouge2_f,rougel_r,rougel_p,rougel_f,rougew_r,rougew_p,rougew_f = gm([rouge1_p,rouge1_r,rouge1_f,rouge2_p,rouge2_r,rouge2_f,rougel_r,rougel_p,rougel_f,rougew_r,rougew_p,rougew_f])

        # logger.info("BLUE-1:{} ROUGE1-P:{} ROUGE1-R:{} ROUGE1-F1:{}".format(bleu_1 * 100,
        # rouge1_p*100, rouge1_r*100, rouge1_f*100))
        # logger.info("BLUE-4:{} ROUGE2-P:{} ROUGE2-R:{} ROUGE2-F1:{}".format(bleu_4 * 100,
        # rouge2_p*100, rouge2_r*100, rouge2_f*100))
        logger.info("ROUGE-L-R:{} ROUGE-L-P:{} ROUGE-L-F:{}".format(rougel_r * 100,
        rougel_p*100, rougel_f*100))
        logger.info("ROUGE-W-R:{} ROUGE-W-P:{} ROUGE-W-F:{}".format(rougew_r * 100,
        rougew_p*100, rougew_f*100))

        if model.restarted:
            from nlgeval import NLGEval
            if model.nlgeval is None:
                model.nlgeval = NLGEval(no_glove=True, no_skipthoughts=True)  # loads the models
            # result_metric = nlgeval.compute_metrics(reformat_reference(text_test_reference, 5), text_predict)
            # names = ['Bleu_1','Bleu_2','Bleu_3','Bleu_3','Bleu_4','ROUGE_L','METEOR','CIDEr','SkipThoughtCS']
            names = ['Bleu_1','Bleu_2','Bleu_3','Bleu_3','Bleu_4','ROUGE_L','METEOR','CIDEr'] #Remove SKipThoughtCS
            my_sc = defaultdict(rs)

            # with open(config.exp_name+'_predict.txt','w') as f:
            #     f.writelines(fin_predict)
            # reference_file = open(config.exp_name+'_ref.txt','w')

            id_to_blue = {}

            for ref,pred,user_id_now in zip(fin_reference, fin_predict, fin_id):
                stores = defaultdict(r0)
                for n in names:
                    stores[n] = 0
                for r in ref:
                    one_result = model.nlgeval.compute_individual_metrics([r],pred)
                    for n in names:
                        stores[n] = max(stores[n],one_result[n])
                pred_words = pred.split(' ')
                dis_1 = len(set(ngrams(pred_words, 1)))/ len(pred_words)
                dis_2 = len(set(ngrams(pred_words, 2)))/ len(pred_words)
                id_to_blue[user_id_now] = stores
                for n in names:
                    my_sc[n].append(stores[n])
                my_sc['dis_1'].append(dis_1)
                my_sc['dis_2'].append(dis_2)

            names.append('dis_1')
            names.append('dis_2')
            for n in names:
                logger.info('best {}:{:7.10f}'.format(n,np.mean(my_sc[n])))
            
            with open(os.path.join('runs', config.exp_name, 'id_to_scores.pkl'),'wb') as f:
                pickle.dump(id_to_blue, f)
            # nlgeval = NLGEval()  # loads the models
            # result = nlgeval.compute_metrics(reformat_reference(fin_reference,max_ref), fin_predict)
            
            # me = result['METEOR']
            # cider = result['CIDEr']
            # tcs = result['SkipThoughtCS']
            # com_rougel = result['ROUGE_L']
            # com_blue_1 = result['Bleu_1']
            # com_blue_2 = result['Bleu_2']
            # com_blue_3 = result['Bleu_3']
            # com_blue_4 = result['Bleu_4']
            
            # # me = cider = tcs = 0
            # IGNORE bleurt scores
            # from bleurt import score
            # cp = "./bleurt-base-128"
            
            # refs = reformat_reference(fin_reference, max_ref)
            # cand = fin_predict
            # scorer = score.BleurtScorer(cp)
            # results = []
            # for ref in refs:
            #     results.append(scorer.score(references=ref, candidates=cand))
            # results = np.array(results)
            # results = np.max(results,-1)
            # print('bleurt:',np.mean(results))

        # if writer is not None:
        #     writer.add_scalar("BLEU-1", bleu_1 * 100, cnt)
        #     writer.add_scalar("BLEU-4", bleu_4 * 100, cnt)
        #     writer.add_scalar("ROUGE1-F1", rouge_f1 * 100, cnt)
        #     writer.add_scalar("ROUGE2-F1", rouge2_f1 * 100, cnt)

        logger.info("Generation Done")
        model.backLoss = True




def multi_sample_generate(model, data_feed, config, evaluator, num_batch=1, dest_f=None, writer=None, cnt=None):
    model.eval()
    model.backLoss = False
    model.sample_mode = True
    # if model.restarted:
    #     print('User Embedding:',calculate_similiar(model.user_embedding, 'user_embedding', config))
    #     print('Item Embedding:',calculate_similiar(model.item_embedding, 'item_embedding', config))

    with torch.no_grad():
        de_tknize = get_dekenize()

        data_feed.epoch_init(config, shuffle=True, verbose=True)
        evaluator.initialize()

        logger.info("Generation Begin.")
        ratings = []

        rep_rand_list = []

        show_up = 0
        
        max_ref = 0
        my_reference = defaultdict(rs)
        my_predict = dict()
        my_rand_id = dict()

        done_batch = 0
        
        while True:
            batch = data_feed.next_batch()
            if batch is None:
                break

            outputs, labels, rand_id = model(batch, mode=GEN, gen_type=config.gen_type)
            # rep_rand_list.extend(rand_id)

            batch['rd_ids'] = torch.randn_like(batch['rd_ids']) # replace rd_ids to multi sample
            ui_ids = batch['ui_ids'].tolist()

            # move from GPU to CPU
            labels = labels.cpu()
            pred_labels = [t.cpu().data.numpy() for t in
                        outputs[DecoderRNN.KEY_SEQUENCE]]

            pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0,1)
            true_labels = labels.data.numpy()

            pred_attns = None
            

            preds = []
            trues = []
            
            for b_id,ui_id_now,rep_id_now in zip(range(pred_labels.shape[0]),ui_ids, rand_id):
                pred_str, attn = get_sent(model, de_tknize, pred_labels, b_id, attn=pred_attns)
                true_str, _ = get_sent(model, de_tknize, true_labels, b_id)

                if show_up < 10:
                    print('pred:',pred_str)
                    print('true:',true_str)
                    show_up += 1
                    
                my_predict[ui_id_now] = pred_str
                my_rand_id[ui_id_now] = rep_id_now
                my_reference[ui_id_now].append(true_str)
                
                pred = pred_str.split()
                true = true_str.split()

                preds = tuple(pred)
                trues = [tuple(true)]

        fin_reference,fin_predict,fin_id,fin_rep_id = [],[],[],[]
        for k,v in my_predict.items():
            fin_predict.append(v)
            fin_id.append(k)
            fin_rep_id.append(my_rand_id[k])
            fin_reference.append(my_reference[k])
            max_ref = max(max_ref, len(my_reference[k]))
            
        print("Get max_reference:", max_ref)
        
        if model.restarted:
            from nlgeval import NLGEval
            if model.nlgeval is None:
                model.nlgeval = NLGEval(no_glove=True, no_skipthoughts=True)  # loads the models
            # result_metric = nlgeval.compute_metrics(reformat_reference(text_test_reference, 5), text_predict)
            # names = ['Bleu_1','Bleu_2','Bleu_3','Bleu_3','Bleu_4','ROUGE_L','METEOR','CIDEr','SkipThoughtCS']
            names = ['Bleu_1','Bleu_2','Bleu_3','Bleu_3','Bleu_4','ROUGE_L','METEOR','CIDEr'] #Remove SKipThoughtCS
            my_sc = defaultdict(rs)

            # with open(config.exp_name+'_predict.txt','w') as f:
            #     f.writelines(fin_predict)
            # reference_file = open(config.exp_name+'_ref.txt','w')

            id_to_blue = {}
            id_to_rep_rand = {}

            for ref, pred, user_id_now, rep_rand_id in zip(fin_reference, fin_predict, fin_id, fin_rep_id):
                stores = defaultdict(r0)
                for n in names:
                    stores[n] = 0
                for r in ref:
                    one_result = model.nlgeval.compute_individual_metrics([r],pred)
                    for n in names:
                        stores[n] = max(stores[n],one_result[n])
                pred_words = pred.split(' ')
                dis_1 = len(set(ngrams(pred_words, 1)))/ len(pred_words)
                dis_2 = len(set(ngrams(pred_words, 2)))/ len(pred_words)
                id_to_blue[user_id_now] = stores
                id_to_rep_rand[user_id_now] = rep_rand_id
                for n in names:
                    my_sc[n].append(stores[n])
                my_sc['dis_1'].append(dis_1)
                my_sc['dis_2'].append(dis_2)

            names.append('dis_1')
            names.append('dis_2')
            for n in names:
                logger.info('Multi-sample best {}:{:7.10f}'.format(n,np.mean(my_sc[n])))
            
            import pickle
            with open(os.path.join('runs', config.exp_name, 'id_to_scores.pkl'),'wb') as f:
                pickle.dump(id_to_blue, f)
            with open(os.path.join('runs', config.exp_name, 'id_to_rep_id.pkl'),'wb') as f:
                pickle.dump(id_to_rep_rand, f)

            # nlgeval = NLGEval()  # loads the models
            # result = nlgeval.compute_metrics(reformat_reference(fin_reference,max_ref), fin_predict)
            
            # me = result['METEOR']
            # cider = result['CIDEr']
            # tcs = result['SkipThoughtCS']
            # com_rougel = result['ROUGE_L']
            # com_blue_1 = result['Bleu_1']
            # com_blue_2 = result['Bleu_2']
            # com_blue_3 = result['Bleu_3']
            # com_blue_4 = result['Bleu_4']
            
            # # me = cider = tcs = 0
            #No BLEURT
            # from bleurt import score
            # cp = "./bleurt-base-128"
            
            # refs = reformat_reference(fin_reference, max_ref)
            # cand = fin_predict
            # scorer = score.BleurtScorer(cp)
            # results = []
            # for ref in refs:
            #     results.append(scorer.score(references=ref, candidates=cand))
            # results = np.array(results)
            # results = np.max(results,-1)
            # print('bleurt:',np.mean(results))

        model.sample_mode = False
        model.backLoss = True
        logger.info("Multi-sample Generation Done")
