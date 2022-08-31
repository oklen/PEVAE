from __future__ import print_function
import logging
import os
import json

from models import evaluators, utt_utils, dialog_utils
from models import main as main_train
from models import main_aggresive as main_train_agg
from models import main_pretrain,main_fs
from models.dataset import corpora
from models.dataset import data_loaders
from models.models.dialog_models import *
from models.utils import prepare_dirs_loggers, get_time
from models.multi_bleu import multi_bleu_perl
from models.options import get_parser_cond
from models.dataset.corpora import InputFeatures

logger = logging.getLogger()

def get_corpus_client(config):
    if config.data.lower() == "ptb":
        corpus_client = corpora.PTBCorpus(config)
    elif config.data.lower() == "daily_dialog":
        corpus_client = corpora.DailyDialogCorpus(config)
    elif config.data.lower() == "stanford":
        corpus_client = corpora.StanfordCorpus(config)
    elif config.data.lower() == "adr":
        corpus_client = corpora.aNLG(config, tokenizer=tokenizer)
    elif config.data.lower() == 'pw':
        corpus_client = corpora.PW(config, tokenizer=tokenizer)
    elif config.data.lower() == 'yelp':
        corpus_client = corpora.yelp(config)
    else:
        raise ValueError("Only support four corpus: adr,ptb, daily_dialog and stanford.")
    return corpus_client

def get_dataloader(config, corpus):
    if config.data.lower() == "ptb":
        dataloader = data_loaders.PTBDataLoader
    elif config.data.lower() == "daily_dialog":
        dataloader = data_loaders.DailyDialogSkipLoader
    elif config.data.lower() == "stanford":
        dataloader = data_loaders.SMDDataLoader
    elif config.data.lower() == "adr":
        dataloader = data_loaders.ADRDataLoader
    elif config.data.lower() == "pw":
        dataloader = data_loaders.PWDataLoader
    elif config.data.lower() == "yelp":
        dataloader = data_loaders.YELPDataLoader
    else:
        raise ValueError("Only support three corpus: ptb, daily_dialog and stanford.")

    train_dial, valid_dial, test_dial = corpus['train'], \
                                        corpus['valid'], \
                                        corpus['test']

    train_feed = dataloader("Train", train_dial, config)
    valid_feed = dataloader("Valid", valid_dial, config)
    test_feed = dataloader("Test", test_dial, config)

    try:
        pretrain_dial = corpus['pretrain']
        pretrain_feed = dataloader("PreTrain", pretrain_dial, config)
        return train_feed, valid_feed, test_feed, pretrain_feed
    except:
        return train_feed, valid_feed, test_feed, 

def get_model(corpus_client, config, data_loader=None):
    try:
        model = eval(config.model)(corpus_client, config, data_loader)
    except Exception as e:
        raise NotImplementedError("Fail to build model %s" % (config.model))
    if config.use_gpu:
        model.cuda()
    return model

def evaluation(model, test_feed, train_feed, evaluator):
    engine = main_pretrain
    if config.forward_only:
        test_file = os.path.join(config.log_dir, config.load_sess,
                                 "{}-test-{}.txt".format(get_time(), config.gen_type))
        dump_file = os.path.join(config.log_dir, config.load_sess,
                                 "{}-z.pkl".format(get_time()))
        model_file = os.path.join(config.log_dir, config.load_sess, "model")
        # sampling_file = os.path.join(config.log_dir, config.load_sess,
        #                          "{}-sampling.txt".format(get_time()))
    else:
        test_file = os.path.join(config.session_dir,
                                 "{}-test-{}.txt".format(get_time(), config.gen_type))
        dump_file = os.path.join(config.session_dir, "{}-z.pkl".format(get_time()))
        model_file = os.path.join(config.session_dir, "model")

    engine.generate(model,test_feed,config, evaluator)

    logger.info("Evluation Done!")

def checkData(train_feed, valid_feed):
    ui_map = {}
    train_cf_cnt = 0
    dev_cf_cnt = 0
    dev_missing_cnt = 0
    sentence_set = set()
    abs_missing = 0
    rtf,rvf = [],[]

    for d in train_feed:
        user = d[0]
        item = d[1]
        feature = d[2]
        sentence = d[3]

        if ui_map.get(str(user)+'#'+str(item)+'#'+str(feature)) is None:
            ui_map[str(user)+'#'+str(item)+'#'+str(feature)] = sentence
            sentence_set.add(" ".join(sentence))
            rtf.append(d)
        elif ui_map[str(user)+'#'+str(item)+'#'+str(feature)] != sentence:
            print("Conflict Dectect!")
            print(user,item,sentence)
            train_cf_cnt+=1

    for d in valid_feed:
        user = d[0]
        item = d[1]
        feature = d[2]
        sentence = d[3]
        if " ".join(sentence) not in sentence_set:
            abs_missing += 1
        # elif ui_map.get(str(user)+'#'+str(item)+'#'+str(feature)) is None:
        #     print("Missing Sentence",sentence)
            dev_missing_cnt+=1
        elif ui_map[str(user)+'#'+str(item)+'#'+str(feature)] != sentence:
            print("Conflict Dectect!")
            print(user,item,sentence)
            dev_cf_cnt += 1
        else:
            rvf.append(d)
        
    print(len(train_feed), len(valid_feed))
    print(train_cf_cnt, dev_missing_cnt, dev_cf_cnt, abs_missing)
    return rtf,rvf


def main(config):
    # Remove save logs
    # prepare_dirs_loggers(config, os.path.basename(__file__))
    torch.multiprocessing.set_start_method('forkserver')
    
    config.data = 'yelp'
    corpus_client = get_corpus_client(config)
    adr_corpus = corpus_client.get_corpus()
    evaluator = evaluators.BleuEvaluator("CornellMovie")

    train_feed, valid_feed, test_feed,pretrain_feed = get_dataloader(config, adr_corpus)

    if config.few_shot is True:
        pretrained_model = 'save_model35.pt'
    else:
        pretrained_model = None


    if config.forward_only is False:
        if pretrained_model is None:
            model = get_model(corpus_client, config, [train_feed, valid_feed, test_feed, pretrain_feed])
            if config.load_emb is None:
                engine = main_pretrain
                model.set_pretrained_mode()
                engine.train(model, train_feed, valid_feed, evaluator, config)
            else:
                engine = main_train
                model.set_normal_train_mode()
                engine.train(model, train_feed, valid_feed, evaluator, config)
        else:
            # print("==========Using Pretrained VAE============")
            print("==============Loading for evalation===============")
        #     # Strict remove for added attention
            config.few_shot = True
            model = get_model(corpus_client, config, [train_feed,valid_feed, test_feed, pretrain_feed])
            par_dict = torch.load(open(pretrained_model,'rb'))
            par_dict.pop('user_embedding.weight',None)
            par_dict.pop('item_embedding.weight',None)
            # par_dict.pop('feature_embedding.weight',None)
            par_dict.pop('rating_embedding.weight',None)
            model.load_state_dict(par_dict,strict=False)
            engine = main_fs
            model.set_pretrained_mode()
            engine.train(model, train_feed, valid_feed, evaluator, config)

            model.few_shot_model = False

            engine = main_pretrain
            engine.train(model, train_feed, valid_feed, evaluator, config)


    model.eval()
    evaluation(model, test_feed, train_feed, evaluator)

if __name__ == "__main__":

    config = get_parser_cond()
    with torch.cuda.device(config.gpu_idx):
        torch.backends.cudnn.benchmark = True
        main(config)
