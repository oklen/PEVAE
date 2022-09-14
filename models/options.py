from torch._C import default_generator
from models.utils import str2bool, process_config
import argparse
import logging
# import dgmvae.models.sent_models as sent_models
# import dgmvae.models.sup_models as sup_models
import models.dialog_models as dialog_models


def add_default_training_parser(parser):
    parser.add_argument('--op', type=str, default='adamw')
    parser.add_argument('--backward_size', type=int, default=5)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=0.5)
    parser.add_argument('--init_w', type=float, default=0.08)
    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--lr_hold', type=int, default=3)
    parser.add_argument('--lr_decay', type=str2bool, default=True)
    parser.add_argument('--lr_decay_rate', type=float, default=0.8)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--improve_threshold', type=float, default=0.996)
    parser.add_argument('--patient_increase', type=float, default=2.0)
    parser.add_argument('--early_stop', type=str2bool, default=True)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--save_model', type=str2bool, default=True)
    parser.add_argument('--use_gpu', type=str2bool, default=True)
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--print_step', type=int, default=500)
    parser.add_argument('--fix_batch', type=str2bool, default=False)
    parser.add_argument('--ckpt_step', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--preview_batch_num', type=int, default=1)
    parser.add_argument('--gen_type', type=str, default='greedy')
    parser.add_argument('--avg_type', type=str, default='seq')
    parser.add_argument('--beam_size', type=int, default=10)
    parser.add_argument('--forward_only', type=str2bool, default=False)
    parser.add_argument('--load_sess', type=str, default="", help="Load model directory.")
    parser.add_argument('--encoder_model',type=str, default='bert-base-uncased')
    return parser

def add_default_cond_training_parser(parser):
    parser.add_argument('--op', type=str, default='adam')
    parser.add_argument('--backward_size', type=int, default=30)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--init_w', type=float, default=0.1)
    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--lr_hold', type=int, default=3)
    parser.add_argument('--lr_decay', type=str2bool, default=True)
    parser.add_argument('--lr_decay_rate', type=float, default=0.8)
    parser.add_argument('--dropout', type=float, default=0.2) # from 0.3
    parser.add_argument('--improve_threshold', type=float, default=0.996)
    parser.add_argument('--patient_increase', type=float, default=4.0)
    parser.add_argument('--early_stop', type=str2bool, default=True)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--loss_type', type=str, default="e2e")

    parser.add_argument('--save_model', type=str2bool, default=True)
    parser.add_argument('--use_gpu', type=str2bool, default=True)
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--print_step', type=int, default=100)
    parser.add_argument('--fix_batch', type=str2bool, default=False)
    parser.add_argument('--ckpt_step', type=int, default=500)
    parser.add_argument('--freeze_step', type=int, default=99999)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--preview_batch_num', type=int, default=1)
    parser.add_argument('--gen_type', type=str, default='greedy')
    parser.add_argument('--avg_type', type=str, default='seq')
    parser.add_argument('--beam_size', type=int, default=10)
    parser.add_argument('--forward_only', type=str2bool, default=False)
    parser.add_argument('--load_sess', type=str, default="", help="Load model directory.")

    parser.add_argument('--embedding_path', type=str, default="data/word2vec/smd.txt", help="word embedding file.")
    parser.add_argument('--encoder_model',type=str, default='bert-base-uncased')
    parser.add_argument('--target_kl',type=float, default=0, help="Bound for KL")
    parser.add_argument('--max_valid_batch_count',type=int, default=1024)
    parser.add_argument('--warmup_steps', type=int, default=2000, help='Steps to Warmup model')
    parser.add_argument('--yelp_max_seq_len', type=int, default=16, help='Max sequence length for yelp.')
    parser.add_argument('--max_seq_len', type=int, default=16, help='Max sequence length for Any.')
    parser.add_argument('--tau', type=float, default=0.5, help='Set Tau for BN')
    parser.add_argument('--do_follow',  action='store_false', help='Optim Follow')
    parser.add_argument('--do_kld',  action='store_false', help='Whether cal KLD loss')
    parser.add_argument('--use_bnvae', action='store_false', help='Whether Use bn VAE to prevent KL-vanish')
    parser.add_argument('--data_path', type=str, default='./feature/yelp', help='Use which data')
    parser.add_argument('--teach_force_bound', type=int, default=30, help='Bound epoch to use TEACH FORCE')
    parser.add_argument('--exp_name', type=str, default='-1', help='Must choose a experiment name for record.')
    parser.add_argument('--optim_times', type=int, default=1, help='Run times for following')
    parser.add_argument('--use_vae',  action='store_false', help='Whether use vae')
    parser.add_argument('--direct_follow',  action='store_true', help='Whether use vae')
    parser.add_argument('--embedding_record',  action='store_true', help='Whether to save embedding for visilization')
    parser.add_argument('--do_pred',  action='store_true', help='Whether to do rating prediect')
    parser.add_argument('--rating_ab',  action='store_true', help='Whether to do ab compare exp')
    parser.add_argument('--load_emb', type=str, default=None, help='Where to Load initilized embeddings.')
    parser.add_argument('--few_shot',action='store_true', help='whether use few shot initilized')
    parser.add_argument('--emb_kl',action='store_true', help='whether use set kl for embeddings')
    parser.add_argument('--use_feature',action='store_true', help='whether use feature in generation.')
    parser.add_argument('--o2_follow',action='store_true', help='whether use another to do pre-follow.')
    parser.add_argument('--no_mutual_information',action='store_true', help='enable to skip mutual information regularization.')
    parser.add_argument('--enable_multi_sample',action='store_true', help='enable evaluation of multi-sample in generation.')
    parser.add_argument('--expand_weight',type=float, default=0.5, help='The weight for q(z|y) to cover distribution.')
    parser.add_argument('--converge_weight',type=float, default=0.1, help='The weight for q(z|x) to follow q(z|y).')
    
    return parser

def add_default_variational_training_parser(parser):
    # KL-annealing
    parser.add_argument('--anneal', type=str2bool, default=True)
    parser.add_argument('--anneal_function', type=str, default='logistic')
    parser.add_argument('--anneal_k', type=float, default=0.0025)
    parser.add_argument('--anneal_x0', type=int, default=2500)
    parser.add_argument('--anneal_warm_up_step', type=int, default=0)
    parser.add_argument('--anneal_warm_up_value', type=float, default=0.000)

    # Word dropout & posterior sampling number
    parser.add_argument('--word_dropout_rate', type=float, default=0.0)
    parser.add_argument('--post_sample_num', type=int, default=20)
    parser.add_argument('--sel_metric', type=str, default="elbo", help="select best checkpoint base on what metric.",
                        choices=['elbo', 'obj'],)
    parser.add_argument('--ost', action='store_true',help="Use to validate whether noise embedding prevent optimizing.")
    parser.add_argument('--no_gen', action='store_true',help="Disable generate.")

    #Discrete Tokens
    parser.add_argument('--user_token_cnt', type=int, default=8)
    parser.add_argument('--user_token_size', type=int, default=512)
    parser.add_argument('--item_token_cnt', type=int, default=8)
    parser.add_argument('--item_token_size', type=int, default=512)

    # Other:
    parser.add_argument('--aggressive', type=str2bool, default=False)
    return parser

def add_default_data_parser(parser):
    # Data & logging path
    parser.add_argument('--data', type=str, default='ptb')
    parser.add_argument('--data_dir', type=str, default='data/ptb')
    parser.add_argument('--log_dir', type=str, default='logs/ptb')
    # Draw points
    parser.add_argument('--fig_dir', type=str, default='figs')
    parser.add_argument('--draw_points', type=str2bool, default=False)
    return parser



def get_parser(model_class="sent_models"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="GMVAE")
    parser = add_default_data_parser(parser)
    parser = add_default_training_parser(parser)
    parser = add_default_variational_training_parser(parser)

    config, unparsed = parser.parse_known_args()

    try:
        model_name = model_class + "." + config.model
        model_class = eval(model_name)
        parser = model_class.add_args(parser)
    except Exception as e:
        raise ValueError("Wrong model" + config.model)

    config, _ = parser.parse_known_args()
    print(config)
    config = process_config(config)
    return config


def get_parser_cond(model_class="dialog_models"):
    # Conditional generation

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="ADVAE")
    parser = add_default_data_parser(parser)
    parser = add_default_cond_training_parser(parser)
    parser = add_default_variational_training_parser(parser)

    config, unparsed = parser.parse_known_args()

    try:
        model_name = model_class + "." + config.model
        print(model_name)
        model_class = eval(model_name)
        parser = model_class.add_args(parser)
    except Exception as e:
        raise ValueError("Wrong model" + config.model)

    config, _ = parser.parse_known_args()
    print(config)
    config = process_config(config)
    return config



