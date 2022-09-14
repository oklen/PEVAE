import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dataset.corpora import PAD, BOS, EOS, UNK
from torch.autograd import Variable
from models import criterions
from models.enc2dec.decoders import DecoderRNN
from models.enc2dec.encoders import EncoderRNN, RnnUttEncoder
from models.utils import INT, FLOAT, LONG, cast_type
from models import nn_lib
from models.enc2dec.decoders import GEN, TEACH_FORCE
from models.utils import Pack, kl_anneal_function
from torch.nn.utils import spectral_norm
import itertools
import numpy as np
import math
import transformers


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.use_gpu = config.use_gpu
        self.flush_valid = False
        self.config = config
        self.kl_w = 0.0
        self.scheduler = None
        self.optimizer = None

    @staticmethod
    def add_args(parser):
        return parser

    def np2var(self, inputs, dtype):
        if inputs is None:
            return None
        return cast_type(Variable(torch.from_numpy(inputs)), dtype,
                         self.use_gpu)

    def torch2var(self, inputs):
        if inputs is None:
            return None
        if self.use_gpu:
            return Variable(inputs).cuda()
        else:
            return Variable(inputs)

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, batch_cnt, loss, step=None):
        total_loss = self.valid_loss(loss, batch_cnt, step=step)
        total_loss.backward()

    def valid_loss(self, loss, batch_cnt=None, step = None):
        raise NotImplemented
        # total_loss = 0.0
        # for key, l in loss.items():
        #     if l is not None:
        #         total_loss += l
        # return total_loss

    def model_sel_loss(self, loss, batch_cnt):
        return self.valid_loss(loss, batch_cnt)

    def _gather_last_out(self, rnn_outs, lens):
        """
        :param rnn_outs: batch_size x T_len x dimension
        :param lens: [a list of lens]
        :return: batch_size x dimension
        """
        time_dimension = 1
        len_vars = self.np2var(np.array(lens), LONG)
        len_vars = len_vars.view(-1, 1).expand(len(lens), rnn_outs.size(2)).unsqueeze(1)
        slices = rnn_outs.gather(time_dimension, len_vars-1)
        return slices.squeeze(time_dimension)

    def _remove_padding(self, feats, words):
        """"
        :param feats: batch_size x num_words x feats
        :param words: batch_size x num_words
        :return: the same input without padding
        """
        if feats is None:
            return None, None

        batch_size = words.size(0)
        valid_mask = torch.sign(words).float()
        batch_lens = torch.sum(valid_mask, dim=1)
        max_word_num = torch.max(batch_lens)
        padded_lens = (max_word_num - batch_lens).cpu().data.numpy()
        valid_words = []
        valid_feats = []

        for b_id in range(batch_size):
            valid_idxs = valid_mask[b_id].nonzero().view(-1)
            valid_row_words = torch.index_select(words[b_id], 0, valid_idxs)
            valid_row_feat = torch.index_select(feats[b_id], 0, valid_idxs)

            padded_len = int(padded_lens[b_id])
            valid_row_words = F.pad(valid_row_words, (0, padded_len))
            valid_row_feat = F.pad(valid_row_feat, (0, 0, 0, padded_len))

            valid_words.append(valid_row_words.unsqueeze(0))
            valid_feats.append(valid_row_feat.unsqueeze(0))

        feats = torch.cat(valid_feats, dim=0)
        words = torch.cat(valid_words, dim=0)
        return feats, words

    def get_optimizer(self, config):
        if config.op == 'adamw':
            no_decay = ['bias', 'LayerNorm.weight']
            bert_compent_name = ['Bert','Transformer']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_compent_name)], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_compent_name)], 'weight_decay': 0.0},
                {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in bert_compent_name)], 'weight_decay': args.weight_decay, 'lr':config.init_lr/10},
                {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in bert_compent_name)], 'weight_decay': 0.0, 'lr':config.init_lr/10}
                ]
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.init_lr, eps=1e-6)
        if config.op == 'adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                           self.parameters()), lr=config.init_lr)
        elif config.op == 'sgd':
            self.optimizer =torch.optim.SGD(self.parameters(), lr=config.init_lr,
                                   momentum=config.momentum)
        elif config.op == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=config.init_lr,
                                       momentum=config.momentum)
        self.scheduler = transformers.get_constant_schedule_with_warmup(self.optimizer,config.warmup_steps)
        return self.optimizer

class GMMBase(BaseModel):
    def __init__(self, config):
        super(GMMBase, self).__init__(config)
        # self.init_gaussian()
        # self.last_z = None

    @staticmethod
    def add_args(parser):
        from models.utils import str2bool

        # Latent variable:
        parser.add_argument('--k', type=int, default=5, help="Latent size of discrete latent variable")
        parser.add_argument('--latent_size', type=int, default=16, help="The number of discrete latent variables.")
        parser.add_argument('--mult_k', type=int, default=3)

        # Network setting:
        parser.add_argument('--rnn_cell', type=str, default='gru')
        parser.add_argument('--embed_size', type=int, default=200)
        parser.add_argument('--utt_type', type=str, default='attn_rnn')
        parser.add_argument('--utt_cell_size', type=int, default=256)
        parser.add_argument('--ctx_cell_size', type=int, default=512)
        parser.add_argument('--dec_cell_size', type=int, default=512)
        parser.add_argument('--bi_ctx_cell', type=str2bool, default=False)
        # parser.add_argument('--num_layer_enc', type=int, default=1)
        # parser.add_argument('--num_layer_dec', type=int, default=1)
        parser.add_argument('--num_layer', type=int, default=1)
        parser.add_argument('--use_attn', type=str2bool, default=True)
        parser.add_argument('--attn_type', type=str, default='cat')
        parser.add_argument('--tie_output_embed', type=str2bool, default=False)
        parser.add_argument('--max_utt_len', type=int, default=40)
        parser.add_argument('--max_dec_len', type=int, default=40)
        parser.add_argument('--max_vocab_cnt', type=int, default=20000)
        parser.add_argument('--greedy_q', type=str2bool, default=True)

        # Other settings:
        parser.add_argument('--use_attribute', type=str2bool, default=True)
        parser.add_argument('--use_mutual', type=str2bool, default=True)
        parser.add_argument('--gmm', type=str2bool, default=True)
        parser.add_argument('--beta', type=float, default=1.0)
        # parser.add_argument('--kl_weight', type=float, default=1.0)

        return parser

    def model_sel_loss(self, loss, batch_cnt):
        if self.kl_w == 0.0:
            return self.valid_loss(loss)
        return loss.nll + loss.pi_nll

    def init_gaussian(self):
        mus = torch.randn(self.config.mult_k, self.config.k, self.config.latent_size)
        logvar = torch.randn(self.config.mult_k, self.config.k, self.config.latent_size)
        if torch.cuda.is_available():
            mus = mus.cuda()
            logvar = logvar.cuda()
        self.gaussian_mus = torch.nn.Parameter(mus, requires_grad=True)
        self.gaussian_logvar = torch.nn.Parameter(logvar, requires_grad=True)

    def reparameterization(self, mu, logvar, sample=True, z = None):
        if self.training or sample:
            # std = torch.exp(0.5 * logvar)
            std = torch.exp(logvar)
            # z = self.torch2var(torch.randn(mu.size()))
            if z is None:
                z = torch.randn_like(std)
                # z = torch.zeros_like(std)
                self.last_z = z
            z = z * std + mu
            return z
        else:
            return mu

    def zkl_loss(self, tgt_probs, mean, log_var, mean_prior=True):
        mean = mean.view(-1, self.config.mult_k, self.config.latent_size)
        log_var = log_var.view(-1, self.config.mult_k, self.config.latent_size)
        if mean_prior:
            tgt_probs_ = tgt_probs.unsqueeze(-1).expand(-1, -1, -1, self.config.latent_size)
            eta1 = self.gaussian_mus / torch.exp(self.gaussian_logvar)  # eta1 = \Sigma^-1 * mu
            eta2 = -0.5 * torch.pow(torch.exp(self.gaussian_logvar), -1)
            Eeta1 = torch.sum(tgt_probs_ * eta1, dim=-2)  # [batch_size, mult_k, latent_size]
            Eeta2 = torch.sum(tgt_probs_ * eta2, dim=-2)
            Emu = -0.5 * Eeta1 / Eeta2
            Evar = -0.5 * torch.pow(Eeta2, -1)
            # [batch_size, mult_k, latent_size]
            kl = 0.5 * (
                    torch.sum(log_var.exp().div(Evar), dim=-1)
                    + torch.sum((Emu - mean).pow(2) / Evar, dim=-1)
                    - mean.size(-1)
                    + torch.sum(Evar.log() - log_var, dim=-1)
            )
            # [batch_size, mult_k]
            return kl

        mu_repeat = mean.unsqueeze(-2).expand(-1, -1, self.config.k, -1)  # batch_size x k x z_dim
        logvar_repeat = log_var.unsqueeze(-2).expand(-1, -1, self.config.k, -1)
        gaussian_logvars = self.gaussian_logvar

        kl = 0.5 * (
                torch.sum(logvar_repeat.exp().div(gaussian_logvars.exp()), dim=-1)
                + torch.sum((self.gaussian_mus - mu_repeat).pow(2) / gaussian_logvars.exp(), dim=-1)
                - mean.size(-1)
                + torch.sum((gaussian_logvars - logvar_repeat), dim=-1)
        )  # batch_size x mult_k x k

        return torch.sum(kl * tgt_probs, dim=-1)  # batch_size*mult_k

    def dispersion(self, tgt_probs):
        # tgt_probs: batch_size x mult_k x k
        tgt_probs_ = tgt_probs.unsqueeze(-1).expand(-1, -1, -1, self.config.latent_size)
        eta1 = self.gaussian_mus / torch.exp(self.gaussian_logvar)  # eta1 = \Sigma^-1 * mu
        eta2 = -0.5 * torch.pow(torch.exp(self.gaussian_logvar), -1)
        Eeta1 = torch.sum(tgt_probs_ * eta1, dim=-2) # [batch_size, mult_k, latent_size]
        Eeta2 = torch.sum(tgt_probs_ * eta2, dim=-2)
        AE = -0.25 * Eeta1 * Eeta1 / Eeta2 - 0.5 * torch.log(-2 * Eeta2) # [batch_size, mult_k, latent_size]
        AE = torch.mean(torch.sum(AE, dim=(-1, -2)))

        EA = torch.sum(-0.25 * eta1 * eta1 / eta2 - 0.5 * torch.log(-2 * eta2), dim=-1) # [mult_k, k]
        EA = torch.mean(torch.sum(tgt_probs * EA, dim=(-1,-2)))
        return EA-AE

class Block(nn.Module):
    def __init__(self, input_dim,output_dim, mid_dim = None):
        super().__init__()
        if mid_dim is None:
            mid_dim = input_dim // 2
        self.c1 = spectral_norm(nn.Linear(input_dim,input_dim))
        self.c2 = spectral_norm(nn.Linear(input_dim, mid_dim))
        self.c3 = spectral_norm(nn.Linear(mid_dim, mid_dim))
        self.c4 = spectral_norm(nn.Linear(mid_dim, input_dim))
        self.layer_norm = nn.LayerNorm(input_dim)
        self.translate  = spectral_norm(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = self.layer_norm(x + xhat)
        return self.translate(out)
                                    

from torch.utils.data import DataLoader, RandomSampler
import pickle

class ElementwiseParams(nn.Module):
    def __init__(self, num_params, mode='interleaved'):
        super(ElementwiseParams, self).__init__()
        assert mode in {'interleaved', 'sequential'}
        self.num_params = num_params
        self.mode = mode

    def forward(self, x):
        assert x.dim() == 2, 'Expected input of shape (B,D)'
        if self.num_params != 1:
            assert x.shape[1] % self.num_params == 0
            dims = x.shape[1] // self.num_params
            # x.shape = (bs, num_params * dims)
            if self.mode == 'interleaved':
                x = x.reshape(x.shape[0:1] + (self.num_params, dims))
                # x.shape = (bs, num_params, dims)
                x = x.permute([0,2,1])
                # x.shape = (bs, dims, num_params)
            elif self.mode == 'sequential':
                x = x.reshape(x.shape[0:1] + (dims, self.num_params))
                # x.shape = (bs, dims, num_params)
        return x


class Fblock(nn.Module):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.c1 = nn.Linear(input_dim,input_dim)
        self.c2 = nn.Linear(input_dim,input_dim * 2)
        self.c3 = nn.Linear(input_dim * 2, input_dim * 2)
        self.c4 = nn.Linear(input_dim * 2, input_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.translate  = nn.Linear(input_dim, output_dim)
        self.element_wise = ElementwiseParams(2)

    def forward(self, x):
        xhat = self.c1(F.relu(x))
        xhat = self.c2(F.relu(xhat))
        xhat = self.c3(F.relu(xhat))
        xhat = self.c4(F.relu(xhat))
        # out = self.layer_norm(self.translate(x + xhat))
        out = self.layer_norm(self.translate(xhat))
        return self.element_wise(out)
        # out = self.layer_norm(x + xhat)
        # return self.element_wise(self.translate(out))
        
@torch.jit.script
def hash_rand(mu, logvar, d):
    return mu + logvar.exp() * d

# def hash_rand(mu, logvar, d):
#     return mu

class RecEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        while(input_dim>=output_dim):
            self.layers.append(nn.Sequential(Block(input_dim,input_dim), Block(input_dim, input_dim)))
            input_dim //= 2

    @staticmethod
    def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
        return torch.mean(torch.sum(-0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2),1),0)

    def forward(self, output, pos_embeddings):
        result = []
        kld_loss = 0
        for index, layer in enumerate(self.layers):
            output = layer(output)
            mean,logvar = output.chunk(2, -1)
            result.append((mean,logvar))
            kld_loss = kld_loss + self.gaussian_analytical_kl(mean, torch.zeros_like(mean), logvar, torch.ones_like(logvar))
            output = hash_rand(mean, logvar, pos_embeddings[index])
        return result,output,kld_loss,(mean,logvar)
    

class PriEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers = nn.ModuleList()
        while(input_dim>=output_dim):
            self.layers.append(nn.Sequential(Block(input_dim,input_dim,input_dim // 2 * 3), Block(input_dim, input_dim // 2 * 3, input_dim // 2 * 3)))
            input_dim //= 2

    @staticmethod
    def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
        return torch.mean(torch.sum(-0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2),1),0)

    def forward(self, output, result, pos_embeddings):
        rkl_loss = 0
        for index, layer in enumerate(self.layers):
            output = layer(output)
            mean,logvar,xpp = output.chunk(3, -1)
            tmean,tlogvar = result[index]
            rkl_loss = rkl_loss + self.gaussian_analytical_kl(tmean.detach(),mean,tlogvar.detach(),logvar) * (index+1) / len(self.layers)
            output = hash_rand(mean, logvar, pos_embeddings[index]) + xpp
            last_output = mean
        return output, rkl_loss

class SupRecEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        while(input_dim>=output_dim):
            self.layers.append(nn.Sequential(Block(input_dim,input_dim), Block(input_dim, input_dim // 2, input_dim // 2)))
            input_dim //= 2

    def forward(self, output):
        output_list = []
        for layer in self.layers:
            output_list.append(output)
            output = layer(output)
        return output,output_list

class SupPriEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        while(input_dim>=output_dim):
            self.layers.append(nn.Sequential(Block(input_dim,input_dim), Block(input_dim, input_dim // 2, input_dim // 2)))
            input_dim //= 2

    def forward(self, output):
        output_list = []
        for layer in self.layers:
            output_list.append(output)
            output = layer(output)
        return output,output_list

class SupDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dis_layers = nn.ModuleList()
        self.config = config
        while(input_dim <= output_dim):
            self.layers.append(nn.Sequential(Block(input_dim,input_dim, input_dim * 2), Block(input_dim, input_dim * 2, input_dim * 2)))
            self.dis_layers.append(nn.Sequential(Block(input_dim * 2, input_dim * 2, input_dim * 4), Block(input_dim * 2, input_dim * 4, input_dim * 4)))
            input_dim *= 2

    @staticmethod
    @torch.jit.script
    def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
        return torch.mean(torch.sum(-0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2),1),0)
    @staticmethod
    def gaussian_analytical_kl_standard(mu1, logsigma1):
        return torch.mean(torch.sum(0.5 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1) ** 2),1),0)
    
    def forward(self, output, act1, act2, pos_embeddings, restarted):
        rkl = 0
        if not restarted and not self.config.ost:
            for index, (layer, dis_layer) in enumerate(zip(self.layers, self.dis_layers)):
                mean,logvar = dis_layer(act1[-index]).chunk(2, -1)
                t_mean,t_logvar = dis_layer(act2[-index]).detach().chunk(2, -1) # Already detach here
                # t_mean,t_logvar = dis_layer(act2[-index-1]).detach().chunk(2, -1) # Already detach here
                rkl = rkl + self.gaussian_analytical_kl(mean, t_mean, logvar, t_logvar) / (index + 1)
                rkl = rkl + self.gaussian_analytical_kl_standard(mean, logvar) * self.config.beta
                output = layer(output) + hash_rand(mean, logvar, pos_embeddings[index])
        else:
            for index, (layer, dis_layer) in enumerate(zip(self.layers, self.dis_layers)):
                t_mean,t_logvar = dis_layer(act2[-index-1]).chunk(2, -1)
                if self.training:
                    output =  layer(output) + hash_rand(t_mean, t_logvar, pos_embeddings[index])
                else:
                    output = layer(output) + t_mean
        return output, rkl

class FS_VAE(GMMBase):
    def get_pos_layer(self, latent_size):
        return nn.Sequential(nn.Linear(1, latent_size), nn.GELU(), nn.Linear(latent_size, latent_size),nn.Tanh())

    def __init__(self, corpus, config, data_loader=None):
        super(FS_VAE, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.unk_id = self.rev_vocab[UNK]
        self.user_dict = corpus.use_dict
        self.user_list = list(self.user_dict.keys())
        self.item_dict = corpus.item_dict
        self.item_list = list(self.item_dict.keys())
        self.feature_dict = corpus.feature_dict
        self.agg_optimizer = None
        self.agg_learning_rate = None
        self.few_shot_mode = config.few_shot
        

        self.x_embedding = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.rev_vocab[PAD]) #Word Embedding

        self.emb_size = config.dec_cell_size // 2
        emb_size = config.dec_cell_size // 2

        self.user_embedding = nn.Embedding(len(self.user_dict), self.emb_size) # Give 128

        self.item_embedding = nn.Embedding(len(self.item_dict), self.emb_size) # Give 128

        self.restarted = False

        # if config.load_emb is not None and not config.rating_ab:
        #     with open(config.load_emb,'rb') as f:
        #         embs = pickle.load(f)
        #     self.user_embedding.weight = torch.nn.Parameter(embs['user'])
        #     self.item_embedding.weight = torch.nn.Parameter(embs['item'])
        
        # self.item_embedding.weight.requires_grad = False
        # nn.init.zeros_(self.item_embedding.weight)

        self.feature_embedding = nn.Embedding(len(self.feature_dict), self.emb_size)

        if config.few_shot:
            self.user_store = torch.zeros(len(self.user_dict), config.dec_cell_size // 4).cuda()
            self.item_store = torch.zeros(len(self.item_dict), config.dec_cell_size // 4).cuda()
            self.rating_store = torch.zeros(5, config.dec_cell_size // 4).cuda()
            self.user_cnt = torch.zeros(len(self.user_dict)).cuda()
            self.item_cnt = torch.zeros(len(self.item_dict)).cuda()
            self.rating_cnt = torch.zeros(5).cuda()

        # self.feature_embedding.weight.requires_grad = False
    
        # nn.init.zeros_(self.feature_embedding.weight)
        
        self.rating_embedding = nn.Embedding(5, self.emb_size)

        # 512 word 1024 hidden  256 latent_size

        # self.rating_embedding.weight.requires_grad = False

        self.ls_size = config.dec_cell_size // 2
        self.latent_size =  config.dec_cell_size // 4
        self.latent_size2 =  self.latent_size // 2
        self.latent_size3 =  self.latent_size2 // 2
        self.latent_size4 =  self.latent_size3 // 2
        self.latent_size5 =  self.latent_size4 // 2
        self.latent_size6 =  self.latent_size5 // 2

        self.pos_embedding_generate = nn.Sequential(nn.Linear(1, self.latent_size6), nn.GELU(), nn.Linear(self.latent_size6, self.latent_size6),nn.Tanh())

        self.pos_embedding = nn.ModuleList()
        self.pos_embedding.append(self.get_pos_layer(self.latent_size4))
        self.pos_embedding.append(self.get_pos_layer(self.latent_size5))
        self.pos_embedding.append(self.get_pos_layer(self.latent_size6))


        for dl in data_loader:
            # dl.embedding = self.x_embedding
            dl.rev_vocab = corpus.rev_vocab
            dl.sampler = RandomSampler(dl.data)
            dl.dataloader = DataLoader(dl.data, config.batch_size, sampler=dl.sampler, collate_fn=dl.batcher(dl.rev_vocab,config.max_seq_len, torch.cuda.current_device()), num_workers=1)

        self.x_encoder = EncoderRNN(config.embed_size, config.dec_cell_size,
                                    dropout_p=config.dropout,
                                    rnn_cell=config.rnn_cell,
                                    # n_layers=config.num_layer,
                                    variable_lengths=False)
        
        self.bn = nn.BatchNorm1d(self.latent_size6,affine=False,eps=1e-8,track_running_stats=False)

        self.scale = torch.nn.Parameter(torch.tensor(0.0))
        self.scale2 = torch.nn.Parameter(torch.tensor(0.0))

        self.backLoss = True
        self.GEN_MODE = False

        self.dropout = nn.Dropout(0.2)
        self.mse_loss = torch.nn.MSELoss(reduction='sum')
        self.mse_scale = 20
        self.zkl_scale = 1
        
        # self.white = nn.Linear(bert_config.hidden_size, self.latent_size)
        # 101 == [CLS]

        self.function = F.log_softmax
        self.decoder_size = config.embed_size

        if self.config.use_feature:
            dec_emb_size = config.dec_cell_size // 4 * 3
        else:
            dec_emb_size = config.dec_cell_size // 2

        self.x_decoder = DecoderRNN(self.vocab_size, config.max_seq_len,
                                    dec_emb_size, config.dec_cell_size,
                                    self.go_id, self.eos_id, self.unk_id,
                                    n_layers=config.num_layer, 
                                    rnn_cell=config.rnn_cell,
                                    input_dropout_p=config.dropout,
                                    dropout_p=config.dropout,
                                    use_attention=True,
                                    attn_size=config.dec_cell_size,
                                    use_gpu=config.use_gpu,
                                    embedding=self.x_embedding)

        # self.xtz = nn.Sequential(Block(config.dec_cell_size, config.dec_cell_size), 
        # Block(config.dec_cell_size, self.latent_size, config.dec_cell_size),
        # Block(self.latent_size , self.latent_size),
        # Block(self.latent_size , self.latent_size2, self.latent_size),
        # Block(self.latent_size2, self.latent_size2),
        # Block(self.latent_size2, self.latent_size3),
        # Block(self.latent_size3, self.latent_size3),
        # Block(self.latent_size3, self.latent_size4),
        # Block(self.latent_size4, self.latent_size4),
        # Block(self.latent_size4, self.latent_size5))

        self.rec_encoder = RecEncoder(self.latent_size3, self.latent_size5)
        self.pri_encoder = PriEncoder(self.latent_size3, self.latent_size5)

        self.xtz = nn.Sequential(Block(config.dec_cell_size, config.dec_cell_size), 
        Block(config.dec_cell_size, self.latent_size, config.dec_cell_size),
        Block(self.latent_size , self.latent_size),
        Block(self.latent_size , self.latent_size2, self.latent_size),
        Block(self.latent_size2, self.latent_size2),
        Block(self.latent_size2, self.latent_size3))

        # tH = self.latent_size5 // 2 * 3
        # self.cxtz = nn.Sequential(Block(self.latent_size, self.latent_size), 
        # Block(self.latent_size, self.latent_size2),
        # Block(self.latent_size2, self.latent_size2),
        # Block(self.latent_size2, self.latent_size3),
        # Block(self.latent_size3, self.latent_size3),
        # Block(self.latent_size3, self.latent_size4),
        # Block(self.latent_size4, self.latent_size4, tH),
        # Block(self.latent_size4, tH, tH))

        self.cxtz = nn.Sequential(Block(self.latent_size, self.latent_size), 
        Block(self.latent_size, self.latent_size2),
        Block(self.latent_size2, self.latent_size2),
        Block(self.latent_size2, self.latent_size3))

        emb_size = config.dec_cell_size // 2
        emb3 = emb_size * 3

        self.base_builder = nn.Sequential(Block(emb3, emb3, self.latent_size), Block(emb3, self.latent_size))

        self.tau = config.tau

        self.x_init_connector = nn.Sequential(
            Block(self.latent_size6, self.latent_size6, self.latent_size5),
            Block(self.latent_size6, self.latent_size5, self.latent_size5),
            Block(self.latent_size5, self.latent_size5, self.latent_size4),
            Block(self.latent_size5, self.latent_size4, self.latent_size4),
            Block(self.latent_size4, self.latent_size4, self.latent_size3),
            Block(self.latent_size4, self.latent_size3, self.latent_size3),
            Block(self.latent_size3, self.latent_size3, self.latent_size2),
            Block(self.latent_size3, self.latent_size2, self.latent_size3),
            Block(self.latent_size2, self.latent_size2, self.latent_size),
            Block(self.latent_size2, self.latent_size, self.latent_size),
            nn_lib.LinearConnector(self.latent_size,config.dec_cell_size, config.rnn_cell == 'lstm'))
        
        # self.x_init_encoder = nn.Sequential(
        #     Block(self.latent_size * 2, self.latent_size),
        #     nn_lib.LinearConnector(self.latent_size,self.latent_size,
        #                                                config.rnn_cell == 'lstm'))
        
        self.greedy_cat_connector = nn_lib.GreedyConnector()
        self.nll_loss = criterions.NLLEntropy(0, self.config)
        self.rating_loss = nn.MSELoss()
        self.nll_loss_word = criterions.NLLEntropy(0, self.config, avg_type="word")
        self.ppl = criterions.Perplexity(0, self.config)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.log_uniform_y = Variable(torch.log(torch.ones(1) / config.k))
        self.entropy_loss = criterions.Entropy()
        self.uni_layer_norm = nn.LayerNorm(self.emb_size, elementwise_affine=False)
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
        self.kl_w = 0.0
        self.mode = None
        self.sqrt = torch.sqrt

    def call(self, inputs, mode='positive',scale = None):
        if scale is None:
            scale = self.scale
        if mode == 'positive':
            scale = self.tau + (1 - self.tau) * torch.sigmoid(self.scale)
        else:
            scale = (1 - self.tau) * torch.sigmoid(-self.scale)
        return inputs * self.sqrt(scale)

    def decode(self, input):
        for layer in self.decoder_layer:
            input = layer(input) + input
        return input

    def hash_rand(self, mu, logvar, d):
        return mu + torch.exp(logvar) * d

    def valid_loss(self, loss, batch_cnt=None, step=None):
        if batch_cnt is not None:
            step = batch_cnt

        if step is not None:
            vae_kl_weight = kl_anneal_function(self.config.anneal_function, step,
                                               self.config.anneal_k, self.config.anneal_x0) * self.config.beta # Use beta here !
        else:
            vae_kl_weight = 1.0
        # return Pack(zkl=kld_loss, vae_nll=vae_nll, vae_PPL = vae_ppl, vae_emb=vae_embl*self.mse_scale, vae_rnll=vae_rnll)
        # mi_weight = 0.0 if self.config.use_mutual else 1.0

        if loss.vae_rkl is None or torch.isinf(loss.vae_rkl) or torch.isnan(loss.vae_rkl):
            loss.vae_rkl = 0
        if loss.vae_rnll is None or torch.isinf(loss.vae_rnll) or torch.isnan(loss.vae_rnll):
            loss.vae_rnll = 0
        if loss.vae_zkl is None or torch.isinf(loss.vae_zkl) or torch.isnan(loss.vae_zkl):
            loss.vae_zkl = 0
        if loss.vae_nll is None:
            loss.vae_nll = 0

        if self.training:
            # vae_loss = loss.vae_nll + vae_kl_weight * self.config.beta * (loss.zkl) + loss.vae_rkl * 5 * vae_kl_weight + loss.vae_rnll
            vae_loss = loss.vae_nll + loss.vae_zkl * vae_kl_weight + loss.vae_rnll  + loss.vae_rkl
        else:
            vae_loss = loss.vae_nll +  loss.vae_rnll
            # vae_loss = loss.vae_nll


        return vae_loss

    def set_pretrained_mode(self):
        # self.x_encoder.config.max_seq_len = 40
        self.mode = 0
        
    def set_normal_train_mode(self):
        # self.x_encoder.config.max_seq_len = 40
        self.mode = 1
        
    def pxz_forward(self, batch_size, results, out_utts, mode, gen_type, encoder_output=None,feature_emb=None):

        # map sample to initial state of decoder
        # if self.training:
        #     dec_init_state = self.x_init_connector(results.sample_z)
        # else:
        #     dec_init_state = self.x_init_connector(results.qz_mean)
        dec_init_state = self.x_init_connector(results.sample_z)
        dec_outs, dec_last, dec_ctx = self.x_decoder(batch_size, out_utts, dec_init_state,
                                                     mode=mode, gen_type=gen_type,
                                                     beam_size=self.config.beam_size,
                                                     attn_context=None,latent_variable=feature_emb)
        results['dec_outs'] = dec_outs
        results['dec_ctx'] = dec_ctx
        return results
    
    @staticmethod
    def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
        # y = torch.mean(torch.nan_to_num(torch.sum(-0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2),1),0))
        y = torch.mean(torch.sum(-0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2),1),0)
        # Here we define a special version to pervernt number overflow
        # y = torch.sum(torch.mean(-0.5 + (logsigma2 - logsigma1)/2 + 0.5 * (logsigma1.exp() + (mu1 - mu2) ** 2) / (logsigma2.exp() ),0),0)
        return y
        # return torch.nan_to_num(y)
    
    def gaussian_analytical_standard(self, z_mean, z_logvar):
        z_mean = torch.nan_to_num(z_mean)
        z_logvar = torch.nan_to_num(z_logvar)
        z_mean = self.uni_layer_norm(z_mean)
        z_logvar = self.uni_layer_norm(z_logvar)

        # if self.config.use_bnvae:
        #     z_mean = self.call(self.bn(z_mean),self.scale2)
        #     z_logvar = self.call(self.bn(z_logvar),'n',self.scale2)
        return torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mean ** 2 - z_logvar.exp(), dim = 1), dim = 0)

    
    @staticmethod
    def analysis_mu(mu1, mu2):
        return torch.mean(torch.sum((mu1 - mu2) ** 2,1),0)

    @staticmethod
    def check_loss(loss):
        if torch.isnan(loss) or torch.isnan(loss):
            return 0
        return loss

    def bounded_mse(self, input, target, logvar):
        diff = torch.abs(target - input)
        return torch.mean(torch.sum(torch.clamp_min(diff - 2*logvar.exp(),0)**2,1))

    def forward(self, data_feed, mode, sample_n=1, gen_type='greedy', return_latent=False):
        
        real_feature_embs = None
        real_user_embs = self.user_embedding(data_feed['user_ids'])
        real_item_embs = self.item_embedding(data_feed['item_ids'])
        real_rating_embs = self.rating_embedding(data_feed['ratings'].ge(3).int())

        if self.config.use_feature:
            real_feature_embs = self.feature_embedding(data_feed['feature_ids'])

        enc_outs, enc_last = self.x_encoder(self.x_embedding(data_feed['sentence_embeddings']))
        h = enc_last.transpose(0,1).contiguous().squeeze(1)
        pos_embds = []

        for i in range(3):
            pos_embds.append(self.pos_embedding[i](data_feed['rd_ids'].unsqueeze(-1)))

        result,z_sample,kld_loss,mv = self.rec_encoder(self.xtz(h), pos_embds)
        z_mean,z_logvar = mv

        u_tmp = self.base_builder(torch.cat([real_user_embs, real_item_embs, real_rating_embs],-1))
        cz_sample,rkl_loss = self.pri_encoder(self.cxtz(u_tmp), result, pos_embds)
        # if self.training:
        #     cz_sample,rkl_loss = self.pri_encoder(self.cxtz(u_tmp), result, pos_embds)
        # else:
        #     cz_sample,rkl_loss = self.pri_encoder(self.cxtz(u_tmp), result, [torch.zeros_like(i) for i in pos_embds])

        if self.restarted:
            # cz_sample = self.hash_rand(cz_mean, cz_logvar, self.pos_embedding_generate(data_feed['rd_ids'].unsqueeze(-1))) + xpp
            result = Pack(sample_z=cz_sample)
        else:
            # z_sample = self.hash_rand(z_mean, z_logvar, self.pos_embedding_generate(data_feed['rd_ids'].unsqueeze(-1))) + xpp
            result = Pack(sample_z=z_sample)

        vae_resp = self.pxz_forward(data_feed['sentence_embeddings'].size(0), result, data_feed['input_ids'], mode, gen_type,feature_emb=real_feature_embs)
        vae_nll = self.nll_loss(vae_resp['dec_outs'], data_feed['ans_ids'])
        vae_ppl = self.ppl(vae_resp['dec_outs'], data_feed['ans_ids'])
        vae_rkl = None

        if self.backLoss:
            if self.training:
                # vae_rkl = self.gaussian_analytical_kl(z_mean.detach(), cz_mean, z_logvar.detach(), cz_logvar)
                vae_rkl = rkl_loss
                
            if not self.restarted and self.training:
                z_mean = self.call(self.bn(z_mean))
                z_logvar = self.call(self.bn(z_logvar),'n')
                # kld_loss = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mean ** 2 - z_logvar.exp(), dim = 1), dim = 0)
                kld_loss = self.gaussian_analytical_kl(z_mean,torch.zeros_like(z_mean),z_logvar,torch.ones_like(z_logvar))

                if self.config.emb_kl:
                    tkl = self.gaussian_analytical_standard(uz_mean,uz_logvar) 
                    tkl += self.gaussian_analytical_standard(iz_mean,iz_logvar) 
                    tkl += self.gaussian_analytical_standard(rz_mean,rz_logvar) 
                    kld_loss += tkl
            else:
                kld_loss = None
            return Pack(vae_zkl=kld_loss, vae_nll=vae_nll, vae_PPL = vae_ppl, vae_rnll=None, vae_rkl=vae_rkl)
        else:
            if self.config.do_pred:
                return vae_resp['dec_ctx'], data_feed['ans_ids'], self.dnn(torch.cat([real_user_embs, real_item_embs],-1)).squeeze(-1)
            else:
                return vae_resp['dec_ctx'], data_feed['ans_ids']
        return result


    def get_freeze_optimizer(self, config):
        if config.op == 'adamw':
            no_decay = ['bias', 'LayerNorm.weight']
            bert_compent_name = ['x_encoder','xtz','x_embedding','x_decoder','x_init_connector']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_compent_name)], 'weight_decay': config.weight_decay},
                {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_compent_name)], 'weight_decay': 0.0},
                ]
            print([n for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_compent_name)])
            print("Not use this optimizer!")
            exit(0)
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.init_lr, eps=1e-6)
        if config.op == 'adam':
            self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,
                                           self.parameters()), lr=config.init_lr)
        elif config.op == 'sgd':
            self.optimizer =torch.optim.SGD(self.parameters(), lr=config.init_lr,
                                   momentum=config.momentum)
        elif config.op == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=config.init_lr,
                                       momentum=config.momentum)
        self.scheduler = transformers.get_constant_schedule_with_warmup(self.optimizer,config.warmup_steps)
        return self.optimizer





# from ..gpt import *
from transformers import BertModel,BertConfig

class TFS_VAE(GMMBase):
    def get_pos_layer(self, latent_size):
        return nn.Sequential(nn.Linear(1, latent_size), nn.GELU(), nn.Linear(latent_size, latent_size),nn.Tanh())
    def __init__(self, corpus, config, data_loader=None):
        super(TFS_VAE, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.unk_id = self.rev_vocab[UNK]
        
        self.user_dict = corpus.use_dict
        self.user_list = list(self.user_dict.keys())
        self.item_dict = corpus.item_dict
        self.item_list = list(self.item_dict.keys())
        self.feature_dict = corpus.feature_dict

        #Setting up GPT decoder model
        gpt_config = GPT2Config.from_pretrained('gpt2')
        gpt_config.bos_token_id = self.go_id
        gpt_config.eos_id = self.eos_id
        gpt_config.n_ctx = gpt_config.n_embd = config.embed_size
        gpt_config.vocab_size = self.vocab_size
        gpt_config.n_head = 16 
        gpt_config.n_layer = 2
        gpt_config.n_positions = config.max_seq_len+1 #The first two token is used as a guide
        gpt_config.vocab_size = self.vocab_size
        gpt_config.scale_attn_weights = False #!!!
        gpt_config.padding_id = self.rev_vocab[PAD]

        #Setting up BERT encoder model
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        bert_config.hidden_size = config.embed_size
        bert_config.vocab_size = self.vocab_size
        bert_config.num_attention_heads = 16
        bert_config.num_hidden_layers = 2
        bert_config.max_position_embeddings = config.max_seq_len + 1
        bert_config.intermediate_size = config.embed_size * 4

        
        self.x_decoder = GPT2Model(gpt_config)
        self.x_encoder = BertModel(bert_config)

        self.x_embedding = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.rev_vocab[PAD]) #Word Embedding
        self.x_decoder.wte = self.x_embedding
        self.x_encoder.embeddings.word_embeddings = self.x_embedding

        self.emb_size = config.dec_cell_size // 2
        emb_size = config.dec_cell_size // 2

        self.user_embedding = nn.Embedding(len(self.user_dict), self.emb_size) # Give 128
        self.item_embedding = nn.Embedding(len(self.item_dict), self.emb_size) # Give 128
        self.feature_embedding = nn.Embedding(len(self.feature_dict), self.emb_size)
        self.restarted = False
        
        if config.few_shot:
            self.user_store = torch.zeros(len(self.user_dict), config.dec_cell_size // 4).cuda()
            self.item_store = torch.zeros(len(self.item_dict), config.dec_cell_size // 4).cuda()
            self.rating_store = torch.zeros(5, config.dec_cell_size // 4).cuda()
            self.user_cnt = torch.zeros(len(self.user_dict)).cuda()
            self.item_cnt = torch.zeros(len(self.item_dict)).cuda()
            self.rating_cnt = torch.zeros(5).cuda()

        # self.feature_embedding.weight.requires_grad = False
    
        # nn.init.zeros_(self.feature_embedding.weight)
        
        self.rating_embedding = nn.Embedding(5, self.emb_size)

        # 512 word 1024 hidden  256 latent_size

        # self.rating_embedding.weight.requires_grad = False

        self.ls_size = config.dec_cell_size // 2
        self.latent_size =  config.dec_cell_size // 4
        self.latent_size2 =  self.latent_size // 2
        self.latent_size3 =  self.latent_size2 // 2
        self.latent_size4 =  self.latent_size3 // 2
        self.latent_size5 =  self.latent_size4 // 2
        self.latent_size6 =  self.latent_size5 // 2

        self.pos_embedding_generate = nn.Sequential(nn.Linear(1, self.latent_size6), nn.GELU(), nn.Linear(self.latent_size6, self.latent_size6),nn.Tanh())

        self.pos_embedding = nn.ModuleList()
        self.pos_embedding.append(self.get_pos_layer(self.latent_size4))
        self.pos_embedding.append(self.get_pos_layer(self.latent_size5))
        self.pos_embedding.append(self.get_pos_layer(self.latent_size6))


        for dl in data_loader:
            # dl.embedding = self.x_embedding
            dl.rev_vocab = corpus.rev_vocab
            dl.sampler = RandomSampler(dl.data)
            dl.dataloader = DataLoader(dl.data, config.batch_size, sampler=dl.sampler, collate_fn=dl.batcher(dl.rev_vocab,config.max_seq_len, torch.cuda.current_device()), num_workers=1)

        
        self.bn = nn.BatchNorm1d(self.latent_size6,affine=False,eps=1e-8,track_running_stats=False)

        self.scale = torch.nn.Parameter(torch.tensor(0.0))
        self.scale2 = torch.nn.Parameter(torch.tensor(0.0))

        self.backLoss = True
        self.GEN_MODE = False

        self.dropout = nn.Dropout(0.2)
        self.mse_loss = torch.nn.MSELoss(reduction='sum')
        self.mse_scale = 20
        self.zkl_scale = 1
        
        # self.white = nn.Linear(bert_config.hidden_size, self.latent_size)
        # 101 == [CLS]

        self.function = F.log_softmax
        self.decoder_size = config.embed_size

        if self.config.use_feature:
            dec_emb_size = config.dec_cell_size // 4 * 3
        else:
            dec_emb_size = config.dec_cell_size // 2


        self.rec_encoder = RecEncoder(self.latent_size3, self.latent_size5)
        self.pri_encoder = PriEncoder(self.latent_size3, self.latent_size5)

        self.xtz = nn.Sequential(Block(config.dec_cell_size, config.dec_cell_size), 
        Block(config.dec_cell_size, self.latent_size, config.dec_cell_size),
        Block(self.latent_size , self.latent_size),
        Block(self.latent_size , self.latent_size2, self.latent_size),
        Block(self.latent_size2, self.latent_size2),
        Block(self.latent_size2, self.latent_size3))


        self.cxtz = nn.Sequential(Block(self.latent_size, self.latent_size), 
        Block(self.latent_size, self.latent_size2),
        Block(self.latent_size2, self.latent_size2),
        Block(self.latent_size2, self.latent_size3))

        emb_size = config.dec_cell_size // 2
        emb3 = emb_size * 3

        self.base_builder = nn.Sequential(Block(emb3, emb3, self.latent_size), Block(emb3, self.latent_size))

        self.tau = config.tau

        self.x_init_connector = nn.Sequential(
            Block(self.latent_size6, self.latent_size6, self.latent_size5),
            Block(self.latent_size6, self.latent_size5, self.latent_size5),
            Block(self.latent_size5, self.latent_size5, self.latent_size4),
            Block(self.latent_size5, self.latent_size4, self.latent_size4),
            Block(self.latent_size4, self.latent_size4, self.latent_size3),
            Block(self.latent_size4, self.latent_size3, self.latent_size3),
            Block(self.latent_size3, self.latent_size3, self.latent_size2),
            Block(self.latent_size3, self.latent_size2, self.latent_size3),
            Block(self.latent_size2, self.latent_size2, self.latent_size),
            Block(self.latent_size2, self.latent_size, self.latent_size),
            nn_lib.LinearConnector(self.latent_size,config.dec_cell_size, config.rnn_cell == 'lstm'))
        
        self.greedy_cat_connector = nn_lib.GreedyConnector()
        self.nll_loss = criterions.NLLEntropy(0, self.config)
        self.rating_loss = nn.MSELoss()
        self.nll_loss_word = criterions.NLLEntropy(0, self.config, avg_type="word")
        self.ppl = criterions.Perplexity(0, self.config)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.log_uniform_y = Variable(torch.log(torch.ones(1) / config.k))
        self.entropy_loss = criterions.Entropy()
        self.uni_layer_norm = nn.LayerNorm(self.emb_size, elementwise_affine=False)
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
        self.kl_w = 0.0
        self.mode = None
        self.sqrt = torch.sqrt

    def call(self, inputs, mode='positive',scale = None):
        if scale is None:
            scale = self.scale
        if mode == 'positive':
            scale = self.tau + (1 - self.tau) * torch.sigmoid(self.scale)
        else:
            scale = (1 - self.tau) * torch.sigmoid(-self.scale)
        return inputs * self.sqrt(scale)

    def decode(self, input):
        for layer in self.decoder_layer:
            input = layer(input) + input
        return input

    def hash_rand(self, mu, logvar, d):
        return mu + torch.exp(logvar) * d

    def valid_loss(self, loss, batch_cnt=None, step=None):
        if batch_cnt is not None:
            step = batch_cnt

        if step is not None:
            vae_kl_weight = kl_anneal_function(self.config.anneal_function, step,
                                               self.config.anneal_k, self.config.anneal_x0) * self.config.beta # Use beta here !
        else:
            vae_kl_weight = 1.0
        # return Pack(zkl=kld_loss, vae_nll=vae_nll, vae_PPL = vae_ppl, vae_emb=vae_embl*self.mse_scale, vae_rnll=vae_rnll)
        mi_weight = 0.0 if self.config.use_mutual else 1.0

        if loss.vae_rkl is None or torch.isinf(loss.vae_rkl) or torch.isnan(loss.vae_rkl):
            loss.vae_rkl = 0
        if loss.vae_rnll is None or torch.isinf(loss.vae_rnll) or torch.isnan(loss.vae_rnll):
            loss.vae_rnll = 0
        if loss.vae_zkl is None or torch.isinf(loss.vae_zkl) or torch.isnan(loss.vae_zkl):
            loss.vae_zkl = 0
        if loss.vae_nll is None:
            loss.vae_nll = 0

        if self.training:
            # vae_loss = loss.vae_nll + vae_kl_weight * self.config.beta * (loss.zkl) + loss.vae_rkl * 5 * vae_kl_weight + loss.vae_rnll
            vae_loss = loss.vae_nll + loss.vae_zkl * vae_kl_weight + loss.vae_rnll  + loss.vae_rkl
        else:
            vae_loss = loss.vae_nll +  loss.vae_rnll
            # vae_loss = loss.vae_nll


        return vae_loss

    def set_pretrained_mode(self):
        self.mode = 0
        
    def set_normal_train_mode(self):
        # self.x_encoder.config.max_seq_len = 40
        self.mode = 1
        
    def pxz_forward(self, batch_size, results, out_utts, mode, gen_type, encoder_output=None,feature_emb=None):

        # map sample to initial state of decoder
        # if self.training:
        #     dec_init_state = self.x_init_connector(results.sample_z)
        # else:
        #     dec_init_state = self.x_init_connector(results.qz_mean)
        dec_init_state = self.x_init_connector(results.sample_z).squeeze(0) #ORI: 1,batch_size,dec_size
        dec_outs, dec_last, dec_ctx = self.x_decoder(dec_init_state, self.config, out_utts, mode)
        # dec_outs, dec_last, dec_ctx = self.x_decoder(batch_size, out_utts, dec_init_state,
        #                                              mode=mode, gen_type=gen_type,
        #                                              beam_size=self.config.beam_size,
        #                                              attn_context=None,latent_variable=feature_emb)
        results['dec_outs'] = dec_outs
        results['dec_ctx'] = dec_ctx
        return results
    
    @staticmethod
    def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
        # y = torch.mean(torch.nan_to_num(torch.sum(-0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2),1),0))
        y = torch.mean(torch.sum(-0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2),1),0)
        # Here we define a special version to pervernt number overflow
        # y = torch.sum(torch.mean(-0.5 + (logsigma2 - logsigma1)/2 + 0.5 * (logsigma1.exp() + (mu1 - mu2) ** 2) / (logsigma2.exp() ),0),0)
        return y
        # return torch.nan_to_num(y)
    
    def gaussian_analytical_standard(self, z_mean, z_logvar):
        z_mean = torch.nan_to_num(z_mean)
        z_logvar = torch.nan_to_num(z_logvar)
        z_mean = self.uni_layer_norm(z_mean)
        z_logvar = self.uni_layer_norm(z_logvar)

        # if self.config.use_bnvae:
        #     z_mean = self.call(self.bn(z_mean),self.scale2)
        #     z_logvar = self.call(self.bn(z_logvar),'n',self.scale2)
        return torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mean ** 2 - z_logvar.exp(), dim = 1), dim = 0)

    
    @staticmethod
    def analysis_mu(mu1, mu2):
        return torch.mean(torch.sum((mu1 - mu2) ** 2,1),0)

    @staticmethod
    def check_loss(loss):
        if torch.isnan(loss) or torch.isnan(loss):
            return 0
        return loss

    def bounded_mse(self, input, target, logvar):
        diff = torch.abs(target - input)
        return torch.mean(torch.sum(torch.clamp_min(diff - 2*logvar.exp(),0)**2,1))
    @staticmethod
    def radd(input):
        return torch.cat([input[:,0].unsqueeze(-1),input],-1)
    
    def forward(self, data_feed, mode, sample_n=1, gen_type='greedy', return_latent=False):
        
        real_feature_embs = None
        real_user_embs = self.user_embedding(data_feed['user_ids'])
        real_item_embs = self.item_embedding(data_feed['item_ids'])
        real_rating_embs = self.rating_embedding(data_feed['ratings'].ge(3).int())

        if self.config.use_feature:
            real_feature_embs = self.feature_embedding(data_feed['feature_ids'])
            
        #Don't not use input_ids
        enc_outs = self.x_encoder(self.radd(data_feed['sentence_embeddings']), self.radd(data_feed['attention_masks']), self.radd(torch.zeros_like(data_feed['attention_masks'],dtype=torch.long)))
        enc_last = enc_outs.last_hidden_state[:,:2,:]
        enc_last = enc_last.view(enc_last.size(0),-1)
        # h = enc_last.transpose(0,1).contiguous().squeeze(1)
        
        h = enc_last
        pos_embds = []

        for i in range(3):
            pos_embds.append(self.pos_embedding[i](data_feed['rd_ids'].unsqueeze(-1)))

        result,z_sample,kld_loss,mv = self.rec_encoder(self.xtz(h), pos_embds)
        z_mean,z_logvar = mv

        u_tmp = self.base_builder(torch.cat([real_user_embs, real_item_embs, real_rating_embs],-1))
        cz_sample,rkl_loss = self.pri_encoder(self.cxtz(u_tmp), result, pos_embds)
        # if self.training:
        #     cz_sample,rkl_loss = self.pri_encoder(self.cxtz(u_tmp), result, pos_embds)
        # else:
        #     cz_sample,rkl_loss = self.pri_encoder(self.cxtz(u_tmp), result, [torch.zeros_like(i) for i in pos_embds])

        if self.restarted:
            # cz_sample = self.hash_rand(cz_mean, cz_logvar, self.pos_embedding_generate(data_feed['rd_ids'].unsqueeze(-1))) + xpp
            result = Pack(sample_z=cz_sample)
        else:
            # z_sample = self.hash_rand(z_mean, z_logvar, self.pos_embedding_generate(data_feed['rd_ids'].unsqueeze(-1))) + xpp
            result = Pack(sample_z=z_sample)

        vae_resp = self.pxz_forward(data_feed['sentence_embeddings'].size(0), result, data_feed['ans_ids'], mode, gen_type,feature_emb=real_feature_embs)
        vae_nll = self.nll_loss(vae_resp['dec_outs'], data_feed['ans_ids'])
        vae_ppl = self.ppl(vae_resp['dec_outs'], data_feed['ans_ids'])
        vae_rkl = None

        if self.backLoss:
            # if self.training:
            vae_rkl = rkl_loss
                
            # if not self.restarted and self.training:
            #     z_mean = self.call(self.bn(z_mean))
            #     z_logvar = self.call(self.bn(z_logvar),'n')
            #     # kld_loss = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mean ** 2 - z_logvar.exp(), dim = 1), dim = 0)
            #     kld_loss = self.gaussian_analytical_kl(z_mean,torch.zeros_like(z_mean),z_logvar,torch.ones_like(z_logvar))

            #     if self.config.emb_kl:
            #         tkl = self.gaussian_analytical_standard(uz_mean,uz_logvar) 
            #         tkl += self.gaussian_analytical_standard(iz_mean,iz_logvar) 
            #         tkl += self.gaussian_analytical_standard(rz_mean,rz_logvar) 
            #         kld_loss += tkl
            # else:
            #     kld_loss = None
        #  kld_loss = None
        
            return Pack(vae_zkl=kld_loss, vae_nll=vae_nll, vae_PPL = vae_ppl, vae_rnll=None, vae_rkl=vae_rkl)
        else:
            if self.config.do_pred:
                return vae_resp['dec_ctx'], data_feed['ans_ids'], self.dnn(torch.cat([real_user_embs, real_item_embs],-1)).squeeze(-1)
            else:
                return vae_resp['dec_ctx'], data_feed['ans_ids']
        return result


    def get_freeze_optimizer(self, config):
        if config.op == 'adamw':
            no_decay = ['bias', 'LayerNorm.weight']
            bert_compent_name = ['x_encoder','xtz','x_embedding','x_decoder','x_init_connector']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_compent_name)], 'weight_decay': config.weight_decay},
                {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_compent_name)], 'weight_decay': 0.0},
                ]
            print([n for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_compent_name)])
            print("Not use this optimizer!")
            exit(0)
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.init_lr, eps=1e-6)
        if config.op == 'adam':
            self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,
                                           self.parameters()), lr=config.init_lr)
        elif config.op == 'sgd':
            self.optimizer =torch.optim.SGD(self.parameters(), lr=config.init_lr,
                                   momentum=config.momentum)
        elif config.op == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=config.init_lr,
                                       momentum=config.momentum)
        self.scheduler = transformers.get_constant_schedule_with_warmup(self.optimizer,config.warmup_steps)
        return self.optimizer



class VTFS_VAE(GMMBase):
    def get_pos_layer(self, latent_size):
        return nn.Sequential(nn.Linear(1, latent_size), nn.GELU(), nn.Linear(latent_size, latent_size),nn.Tanh())
    def __init__(self, corpus, config, data_loader=None):
        super(VTFS_VAE, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.unk_id = self.rev_vocab[UNK]
        
        self.user_dict = corpus.use_dict
        self.user_list = list(self.user_dict.keys())
        self.item_dict = corpus.item_dict
        self.item_list = list(self.item_dict.keys())
        self.feature_dict = corpus.feature_dict

        #Setting up GPT decoder model
        gpt_config = GPT2Config.from_pretrained('gpt2')
        gpt_config.bos_token_id = self.go_id
        gpt_config.eos_id = self.eos_id
        gpt_config.n_ctx = gpt_config.n_embd = config.embed_size
        gpt_config.vocab_size = self.vocab_size
        gpt_config.n_head = 16 
        gpt_config.n_layer = 2
        gpt_config.n_positions = config.max_seq_len + 1 #The first two token is used as a guide
        gpt_config.vocab_size = self.vocab_size
        gpt_config.scale_attn_weights = False #!!!
        gpt_config.padding_id = self.rev_vocab[PAD]

        #Setting up BERT encoder model
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        bert_config.hidden_size = config.embed_size
        bert_config.vocab_size = self.vocab_size
        bert_config.num_attention_heads = 16
        bert_config.num_hidden_layers = 2
        bert_config.max_position_embeddings = config.max_seq_len + 1
        bert_config.intermediate_size = config.embed_size * 4

        
        self.x_decoder = GPT2Model(gpt_config)
        self.x_encoder = BertModel(bert_config)

        self.x_embedding = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.rev_vocab[PAD]) #Word Embedding
        self.x_decoder.wte = self.x_embedding
        self.x_encoder.embeddings.word_embeddings = self.x_embedding

        self.emb_size = config.dec_cell_size // 2
        emb_size = config.dec_cell_size // 2

        self.user_embedding = nn.Embedding(len(self.user_dict), self.emb_size) # Give 128
        self.item_embedding = nn.Embedding(len(self.item_dict), self.emb_size) # Give 128
        self.feature_embedding = nn.Embedding(len(self.feature_dict), self.emb_size)
        self.restarted = False
        
        if config.few_shot:
            self.user_store = torch.zeros(len(self.user_dict), config.dec_cell_size // 4).cuda()
            self.item_store = torch.zeros(len(self.item_dict), config.dec_cell_size // 4).cuda()
            self.rating_store = torch.zeros(5, config.dec_cell_size // 4).cuda()
            self.user_cnt = torch.zeros(len(self.user_dict)).cuda()
            self.item_cnt = torch.zeros(len(self.item_dict)).cuda()
            self.rating_cnt = torch.zeros(5).cuda()

        # self.feature_embedding.weight.requires_grad = False
    
        # nn.init.zeros_(self.feature_embedding.weight)
        
        self.rating_embedding = nn.Embedding(5, self.emb_size)

        # 512 word 1024 hidden  256 latent_size

        # self.rating_embedding.weight.requires_grad = False

        self.ls_size = config.dec_cell_size // 2
        self.latent_size =  config.dec_cell_size // 4
        self.latent_size2 =  self.latent_size // 2
        self.latent_size3 =  self.latent_size2 // 2
        self.latent_size4 =  self.latent_size3 // 2
        self.latent_size5 =  self.latent_size4 // 2
        self.latent_size6 =  self.latent_size5 // 2

        self.pos_embedding_generate = nn.Sequential(nn.Linear(1, self.latent_size6), nn.GELU(), nn.Linear(self.latent_size6, self.latent_size6),nn.Tanh())

        self.pos_embedding = nn.ModuleList()
        self.pos_embedding.append(self.get_pos_layer(self.latent_size3))
        self.pos_embedding.append(self.get_pos_layer(self.latent_size4))
        self.pos_embedding.append(self.get_pos_layer(self.latent_size5))


        for dl in data_loader:
            # dl.embedding = self.x_embedding
            dl.rev_vocab = corpus.rev_vocab
            dl.sampler = RandomSampler(dl.data)
            dl.dataloader = DataLoader(dl.data, batch_size=config.batch_size, sampler=dl.sampler, collate_fn=dl.batcher(dl.rev_vocab,config.max_seq_len, torch.cuda.current_device()), num_workers=1)

        
        self.bn = nn.BatchNorm1d(self.latent_size6,affine=False,eps=1e-8,track_running_stats=False)

        self.scale = torch.nn.Parameter(torch.tensor(0.0))
        self.scale2 = torch.nn.Parameter(torch.tensor(0.0))

        self.backLoss = True
        self.GEN_MODE = False

        self.dropout = nn.Dropout(0.2)
        self.mse_loss = torch.nn.MSELoss(reduction='sum')
        self.mse_scale = 20
        self.zkl_scale = 1
        
        # self.white = nn.Linear(bert_config.hidden_size, self.latent_size)
        # 101 == [CLS]

        self.function = F.log_softmax
        self.decoder_size = config.embed_size

        if self.config.use_feature:
            dec_emb_size = config.dec_cell_size // 4 * 3
        else:
            dec_emb_size = config.dec_cell_size // 2


        self.rec_encoder = SupRecEncoder(self.latent_size3, self.latent_size5)
        self.pri_encoder = SupPriEncoder(self.latent_size3, self.latent_size5)
        self.sup_decoder = SupDecoder(self.latent_size6, self.latent_size4, config)

        self.xtz = nn.Sequential(Block(config.dec_cell_size, config.dec_cell_size), 
        Block(config.dec_cell_size, self.latent_size, config.dec_cell_size),
        Block(self.latent_size , self.latent_size),
        Block(self.latent_size , self.latent_size2, self.latent_size),
        Block(self.latent_size2, self.latent_size2),
        Block(self.latent_size2, self.latent_size3))


        self.cxtz = nn.Sequential(Block(self.latent_size, self.latent_size), 
        Block(self.latent_size, self.latent_size2),
        Block(self.latent_size2, self.latent_size2),
        Block(self.latent_size2, self.latent_size3))

        emb_size = config.dec_cell_size // 2
        emb3 = emb_size * 3

        self.base_builder = nn.Sequential(Block(emb3, emb3, self.latent_size), Block(emb3, self.latent_size))

        self.tau = config.tau

        self.x_init_connector = nn.Sequential(
            # Block(self.latent_size6, self.latent_size6, self.latent_size5),
            # Block(self.latent_size6, self.latent_size5, self.latent_size5),
            # Block(self.latent_size5, self.latent_size5, self.latent_size4),
            # Block(self.latent_size5, self.latent_size4, self.latent_size4),
            # Block(self.latent_size4, self.latent_size4, self.latent_size3),
            # Block(self.latent_size4, self.latent_size3, self.latent_size3),
            Block(self.latent_size3, self.latent_size3, self.latent_size2),
            Block(self.latent_size3, self.latent_size2, self.latent_size3),
            Block(self.latent_size2, self.latent_size2, self.latent_size),
            Block(self.latent_size2, self.latent_size, self.latent_size),
            nn_lib.LinearConnector(self.latent_size,config.dec_cell_size, config.rnn_cell == 'lstm'))
        
        self.greedy_cat_connector = nn_lib.GreedyConnector()
        self.nll_loss = criterions.NLLEntropy(0, self.config)
        self.rating_loss = nn.MSELoss()
        self.nll_loss_word = criterions.NLLEntropy(0, self.config, avg_type="word")
        self.ppl = criterions.Perplexity(0, self.config)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.log_uniform_y = Variable(torch.log(torch.ones(1) / config.k))
        self.entropy_loss = criterions.Entropy()
        self.uni_layer_norm = nn.LayerNorm(self.emb_size, elementwise_affine=False)
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
        self.kl_w = 0.0
        self.mode = None
        self.sqrt = torch.sqrt

    def call(self, inputs, mode='positive',scale = None):
        if scale is None:
            scale = self.scale
        if mode == 'positive':
            scale = self.tau + (1 - self.tau) * torch.sigmoid(self.scale)
        else:
            scale = (1 - self.tau) * torch.sigmoid(-self.scale)
        return inputs * self.sqrt(scale)

    # def hash_rand(self, mu, logvar, d):
    #     return mu + torch.exp(logvar) * d

    def valid_loss(self, loss, batch_cnt=None, step=None):
        step = batch_cnt

        # if step is not None:
        #     vae_kl_weight = kl_anneal_function(self.config.anneal_function, step,
        #                                        self.config.anneal_k, self.config.anneal_x0) * self.config.beta # Use beta here !
        # else:
        #     vae_kl_weight = 1.0
        vae_kl_weight = 1.0
        # return Pack(zkl=kld_loss, vae_nll=vae_nll, vae_PPL = vae_ppl, vae_emb=vae_embl*self.mse_scale, vae_rnll=vae_rnll)
        # mi_weight = 0.0 if self.config.use_mutual else 1.0

        # if loss.vae_rkl is None or torch.isinf(loss.vae_rkl) or torch.isnan(loss.vae_rkl):
        #     loss.vae_rkl = 0
        # if loss.vae_rnll is None or torch.isinf(loss.vae_rnll) or torch.isnan(loss.vae_rnll):
        #     loss.vae_rnll = 0
        # if loss.vae_zkl is None or torch.isinf(loss.vae_zkl) or torch.isnan(loss.vae_zkl):
        #     loss.vae_zkl = 0
        loss.vae_zkl = 0
        
        if loss.vae_nll is None:
            loss.vae_nll = 0

        if self.training:
            # vae_loss = loss.vae_nll + vae_kl_weight * self.config.beta * (loss.zkl) + loss.vae_rkl * 5 * vae_kl_weight + loss.vae_rnll
            # vae_loss = loss.vae_nll + loss.vae_zkl * vae_kl_weight + loss.vae_rnll  + loss.vae_rkl
            vae_loss = loss.vae_nll +  loss.vae_rkl
        else:
            vae_loss = loss.vae_nll
            # vae_loss = loss.vae_nll


        return vae_loss

    def set_pretrained_mode(self):
        self.mode = 0
        
    def set_normal_train_mode(self):
        self.mode = 1
        
    def pxz_forward(self, batch_size, results, out_utts, mode, gen_type, encoder_output=None,feature_emb=None):
        
        dec_init_state = self.x_init_connector(results.sample_z).squeeze(0) #ORI: 1,batch_size,dec_size
        dec_outs, dec_last, dec_ctx = self.x_decoder(dec_init_state, self.config, out_utts, mode)
        
        # dec_outs, dec_last, dec_ctx = self.x_decoder(batch_size, out_utts, dec_init_state,
        #                                              mode=mode, gen_type=gen_type,
        #                                              beam_size=self.config.beam_size,
        #                                              attn_context=None,latent_variable=feature_emb)
        results['dec_outs'] = dec_outs
        results['dec_ctx'] = dec_ctx
        return results
    
    @staticmethod
    def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
        # y = torch.mean(torch.nan_to_num(torch.sum(-0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2),1),0))
        y = torch.mean(torch.sum(-0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2),1),0)
        # Here we define a special version to pervernt number overflow
        # y = torch.sum(torch.mean(-0.5 + (logsigma2 - logsigma1)/2 + 0.5 * (logsigma1.exp() + (mu1 - mu2) ** 2) / (logsigma2.exp() ),0),0)
        return y
        # return torch.nan_to_num(y)
    
    def gaussian_analytical_standard(self, z_mean, z_logvar):
        z_mean = torch.nan_to_num(z_mean)
        z_logvar = torch.nan_to_num(z_logvar)
        z_mean = self.uni_layer_norm(z_mean)
        z_logvar = self.uni_layer_norm(z_logvar)

        # if self.config.use_bnvae:
        #     z_mean = self.call(self.bn(z_mean),self.scale2)
        #     z_logvar = self.call(self.bn(z_logvar),'n',self.scale2)
        return torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mean ** 2 - z_logvar.exp(), dim = 1), dim = 0)

    
    @staticmethod
    def analysis_mu(mu1, mu2):
        return torch.mean(torch.sum((mu1 - mu2) ** 2,1),0)

    @staticmethod
    def check_loss(loss):
        if torch.isnan(loss) or torch.isnan(loss):
            return 0
        return loss

    def bounded_mse(self, input, target, logvar):
        diff = torch.abs(target - input)
        return torch.mean(torch.sum(torch.clamp_min(diff - 2*logvar.exp(),0)**2,1))
    @staticmethod
    def radd(input):
        return torch.cat([input[:,0].unsqueeze(-1),input],-1)
    
    def forward(self, data_feed, mode, sample_n=1, gen_type='greedy', return_latent=False):
        
        real_feature_embs = None
        real_user_embs = self.user_embedding(data_feed['user_ids'])
        real_item_embs = self.item_embedding(data_feed['item_ids'])
        real_rating_embs = self.rating_embedding(data_feed['ratings'].ge(3).int())

        if self.config.use_feature:
            real_feature_embs = self.feature_embedding(data_feed['feature_ids'])
            
        #Don't not use input_ids
        enc_outs = self.x_encoder(self.radd(data_feed['sentence_embeddings']), self.radd(data_feed['attention_masks']), self.radd(torch.zeros_like(data_feed['attention_masks'],dtype=torch.long)))
        enc_last = enc_outs.last_hidden_state[:,:2,:]
        enc_last = enc_last.view(enc_last.size(0),-1)
        
        # h = enc_last.transpose(0,1).contiguous().squeeze(1)
        h = enc_last
        pos_embds = []

        for i in range(3):
            # Manually Do reverse here
            pos_embds.append(self.pos_embedding[2-i](data_feed['rd_ids'].unsqueeze(-1)))

        u_tmp = self.base_builder(torch.cat([real_user_embs, real_item_embs, real_rating_embs],-1))
        z_pri,pri_state = self.pri_encoder(self.cxtz(u_tmp))
        z_rec,rec_state = self.rec_encoder(self.xtz(h))
        
        # if self.training:
        #     cz_sample,rkl_loss = self.pri_encoder(self.cxtz(u_tmp), result, pos_embds)
        # else:
        #     cz_sample,rkl_loss = self.pri_encoder(self.cxtz(u_tmp), result, [torch.zeros_like(i) for i in pos_embds])
        
        z_sample,rkl_loss = self.sup_decoder(z_rec, rec_state, pri_state, pos_embds, self.restarted)
        result = Pack(sample_z = z_sample)

        # if self.restarted:
        #     # cz_sample = self.hash_rand(cz_mean, cz_logvar, self.pos_embedding_generate(data_feed['rd_ids'].unsqueeze(-1))) + xpp
        #     result = Pack(sample_z=cz_sample)
        # else:
        #     # z_sample = self.hash_rand(z_mean, z_logvar, self.pos_embedding_generate(data_feed['rd_ids'].unsqueeze(-1))) + xpp
        #     result = Pack(sample_z=z_sample)

        vae_resp = self.pxz_forward(data_feed['sentence_embeddings'].size(0), result, data_feed['ans_ids'], mode, gen_type,feature_emb=real_feature_embs)
        
        vae_nll = self.nll_loss(vae_resp['dec_outs'], data_feed['ans_ids'])
        vae_ppl = self.ppl(vae_resp['dec_outs'], data_feed['ans_ids'])
        vae_rkl = None

        if self.backLoss:
            if self.training:
                # vae_rkl = self.gaussian_analytical_kl(z_mean.detach(), cz_mean, z_logvar.detach(), cz_logvar)
                vae_rkl = rkl_loss
            # if not self.restarted and self.training:
            #     z_mean = self.call(self.bn(z_mean))
            #     z_logvar = self.call(self.bn(z_logvar),'n')
            #     # kld_loss = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mean ** 2 - z_logvar.exp(), dim = 1), dim = 0)
            #     kld_loss = self.gaussian_analytical_kl(z_mean,torch.zeros_like(z_mean),z_logvar,torch.ones_like(z_logvar))

            #     if self.config.emb_kl:
            #         tkl = self.gaussian_analytical_standard(uz_mean,uz_logvar) 
            #         tkl += self.gaussian_analytical_standard(iz_mean,iz_logvar) 
            #         tkl += self.gaussian_analytical_standard(rz_mean,rz_logvar) 
            #         kld_loss += tkl
            # else:
            #     kld_loss = None
            return Pack(vae_zkl=vae_rkl, vae_nll=vae_nll, vae_PPL = vae_ppl, vae_rnll=None, vae_rkl=vae_rkl)
        else:
            if self.config.do_pred:
                return vae_resp['dec_ctx'], data_feed['ans_ids'], self.dnn(torch.cat([real_user_embs, real_item_embs],-1)).squeeze(-1)
            else:
                return vae_resp['dec_ctx'], data_feed['ans_ids']

        return result


    def get_freeze_optimizer(self, config):
        if config.op == 'adamw':
            no_decay = ['bias', 'LayerNorm.weight']
            bert_compent_name = ['x_encoder','xtz','x_embedding','x_decoder','x_init_connector']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_compent_name)], 'weight_decay': config.weight_decay},
                {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_compent_name)], 'weight_decay': 0.0},
                ]
            print([n for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_compent_name)])
            print("Not use this optimizer!")
            exit(0)
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.init_lr, eps=1e-6)
        if config.op == 'adam':
            self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,
                                           self.parameters()), lr=config.init_lr)
        elif config.op == 'sgd':
            self.optimizer =torch.optim.SGD(self.parameters(), lr=config.init_lr,
                                   momentum=config.momentum)
        elif config.op == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=config.init_lr,
                                       momentum=config.momentum)
        self.scheduler = transformers.get_constant_schedule_with_warmup(self.optimizer,config.warmup_steps)
        return self.optimizer























from transformers.models.bert.modeling_bert import BertAttention,BertIntermediate,BertOutput,apply_chunking_to_forward
from torch import Tensor,device
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

def get_extended_attention_mask(attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.
        device: (:obj:`torch.device`):
            The device of the input to the model.

    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    # extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        ipt,
        # hidden_states,
        # attention_mask,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        hidden_states = ipt[0]
        attention_mask = ipt[1]

        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value


        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        return layer_output
        # outputs = (layer_output,) + outputs

        # # if decoder, return the attn key/values as the last output
        # if self.is_decoder:
        #     outputs = outputs + (present_key_value,)

        # return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output











from torch.functional import F

#The DisRec and DisPri can be totally same as both of them use token

class DisRec(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.layers = nn.ModuleList()
        scale = 1
        while(input_dim>=output_dim):
            self.layers.append(nn.Sequential(self.get_bert_layer(input_dim, config, scale), Block(input_dim, input_dim // 2, input_dim // 2)))
            input_dim //= 2
            scale += 1

    def get_bert_layer(self, input_dim, config, scale):
        bert_config = BertConfig()
        bert_config.hidden_size = input_dim
        bert_config.vocab_size = config.vocab_size
        bert_config.num_attention_heads = 16
        bert_config.num_hidden_layers = 2
        bert_config.max_position_embeddings = 22
        bert_config.intermediate_size = input_dim * 4
        bert_config.hidden_dropout_prob = 0
        return BertLayer(bert_config)

    def forward(self, output, at_mask):
        output_list = []
        for layer in self.layers:
            output_list.append(output[:,0]) # Get CLS
            # print(output.shape)
            # print(layer)
            output = layer((output, get_extended_attention_mask(at_mask, output.shape[:-1], output.device)))
        return output,output_list

class DisPri(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.layers = nn.ModuleList()
        scale = 0
        while(input_dim>=output_dim):
            self.layers.append(nn.Sequential(self.get_bert_layer(config, scale), Block(input_dim, input_dim // 2, input_dim // 2)))
            # self.mv_layers.append(Block(input_dim, input_dim)) # Use to Process [CLS]
            input_dim //= 2
            scale += 1

    def get_bert_layer(self, config, scale):
        bert_config = BertConfig()
        bert_config.hidden_size = config.user_token_size // (2 ** scale)
        bert_config.vocab_size = config.user_token_size
        bert_config.num_attention_heads = 16
        bert_config.num_hidden_layers = 2
        bert_config.max_position_embeddings = 22
        bert_config.intermediate_size = config.user_token_size * 4
        bert_config.hidden_dropout_prob = 0
        return BertLayer(bert_config)

    def forward(self, output, at_mask):
        output_list = []
        for layer in self.layers:
            output_list.append(output[:,0]) # Get CLS
            output = layer((output, get_extended_attention_mask(at_mask, output.shape[:-1], output.device)))
        return output,output_list

class DlsRec(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.layers = nn.ModuleList()
        scale = 1
        while(input_dim>=output_dim):
            self.layers.append(Block(input_dim, input_dim // 2, input_dim // 2))
            input_dim //= 2
            scale += 1

    def forward(self, output):
        output = output[:,0]
        output_list = []
        for layer in self.layers:
            output_list.append(output) # Get CLS
            output = layer(output)
        return output,output_list

class DlsPri(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.layers = nn.ModuleList()
        scale = 0
        while(input_dim>=output_dim):
            self.layers.append(nn.Sequential(Block(input_dim, input_dim // 2, input_dim // 2)))
            # self.mv_layers.append(Block(input_dim, input_dim)) # Use to Process [CLS]
            input_dim //= 2
            scale += 1

    def forward(self, output):
        output = output[:,0]
        output_list = []
        for layer in self.layers:
            output_list.append(output) # Get CLS
            output = layer(output)
        return output,output_list

from scipy.stats import norm
class DisDec(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dis_layers = nn.ModuleList()
        self.dis_layers2 = nn.ModuleList()
        self.y_z_info = nn.ModuleList()
        self.means = nn.ParameterList()
        self.logvars = nn.ParameterList()
        # tlg is a constant value to setup a logvar lower_bound
        # self.tlg = nn.ParameterList()
        self.config = config

        while(input_dim <= output_dim):
            self.layers.append(nn.Sequential(Block(input_dim, input_dim, input_dim * 2), Block(input_dim, input_dim * 2, input_dim * 2)))
            self.dis_layers.append(nn.Sequential(Block(input_dim * 2, input_dim * 2, input_dim * 4), Block(input_dim * 2, input_dim * 4, input_dim * 4)))
            self.dis_layers2.append(nn.Sequential(Block(input_dim * 2, input_dim * 2, input_dim * 4), Block(input_dim * 2, input_dim * 4, input_dim * 4)))

            self.means.append(nn.Parameter(torch.zeros(input_dim*2, requires_grad = True)))
            self.logvars.append(nn.Parameter(torch.zeros(input_dim*2, requires_grad = True)))
            self.tlg.append(nn.Parameter(-torch.log(2*torch.ones(input_dim*2, requires_grad = False)).unsqueeze(-1)))
            self.y_z_info.append(nn.Sequential(Block(input_dim * 4, input_dim * 4), Block(input_dim*4,1), nn.Sigmoid()))
            # self.z_x_info.append(nn.Sequential(Block(input_dim * 4, input_dim * 4), Block(input_dim*4,1), nn.Sigmoid()))
            input_dim *= 2

    def dis_reparameterization(self, mu, logvar, sample=True, z = None):
        # z = z * torch.exp(logvar) + mu
        return mu + torch.exp(logvar) * torch.rand_like(logvar)

    @staticmethod
    def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
        return torch.mean(torch.sum(-0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2),1),0)

    @staticmethod
    def gaussian_analytical_kl_standard(mu1, logsigma1):
        return torch.mean(torch.sum(0.5 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1) ** 2),1),0)

    def gaussian_analytical_js(self, mu1, mu2, log1, log2):
        return 0
    def forward(self, output, act1, act2, pos_embeddings, restarted):
        # act1 for rec, act2 for pri
        rkl = 0
        output = output[:,0] #Only Use CLS
        # output = torch.mean(output,1) 

        for index, (layer, dis_layer, dis_layer2, tm, tvar, tg, m_layer) in enumerate(zip(self.layers, self.dis_layers, self.dis_layers2, self.means, self.logvars, self.tlg, self.y_z_info)):
            if not restarted and not self.config.ost:
                mean,logvar = dis_layer(act1[-index-1]).chunk(2, -1)
                t_mean,t_logvar = dis_layer2(act2[-index-1]).chunk(2, -1)

                t_samples = self.dis_reparameterization(t_mean,t_logvar)
                pos_scores = m_layer(torch.cat([act2[-index-1], t_samples], -1))
                neg_scores = m_layer(torch.cat([act2[-index-1], t_samples[torch.randperm(t_samples.size(0))]], -1))
                rkl = rkl - torch.mean(torch.log(pos_scores + 1e-6) + torch.log(1 - neg_scores + 1e-6)) #Add mutual information loss

                # print(act2[-index-1])
                # print(act1[-index-1])
                # t_mean,t_logvar = dis_layer(act2[-index-1]).detach().chunk(2, -1) # Already detach here

                rkl = rkl + 0.1 * self.gaussian_analytical_kl(mean, t_mean, logvar, t_logvar) / (index + 1) 
                # rkl = rkl + 0.1 * self.gaussian_analytical_kl(t_mean, mean.detach(), t_logvar, logvar.detach()) / (index + 1) 
                # rkl = rkl + 0.8 * self.gaussian_analytical_kl(mean, t_mean.detach(), logvar, t_logvar.detach()) / (index + 1)
                rkl = rkl + 0.4 * self.gaussian_analytical_kl(mean, t_mean.detach(), logvar, t_logvar.detach()) / (index + 1)

                rkl = rkl + 0.01 * self.gaussian_analytical_kl(mean, torch.zeros_like(tm), logvar, torch.zeros_like(tvar))

                # rkl = rkl + 0.1 * self.gaussian_analytical_kl(mean, tm, logvar, tvar)


                # tlg = -torch.log(torch.ones_like(tvar, requires_grad=False)*2)
                # rkl += (tlg - torch.min(torch.cat([tvar.unsqueeze(-1),tlg.unsqueeze(-1)],-1),-1)) ** 2

                # rkl = rkl + torch.mean(torch.sum(tg.squeeze(-1) - torch.min(torch.cat([tg, tvar.unsqueeze(-1)], -1)))**2,-1)

                output = layer(output) + self.dis_reparameterization(mean, logvar)
            else:
                t_mean,t_logvar = dis_layer(act2[-index-1]).chunk(2, -1)
                if self.training:
                    output =  layer(output) + self.dis_reparameterization(t_mean, t_logvar)
                    # output =  layer(output) + self.dis_reparameterization(t_mean, t_logvar, pos_embeddings[index])
                else:
                    output = layer(output) + t_mean
        return output, rkl


class DlsDec_Auto(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dis_layers = nn.ModuleList()
        self.dis_layers2 = nn.ModuleList()
        # self.y_z_info = nn.ModuleList()
        # self.means = nn.ParameterList()
        # self.logvars = nn.ParameterList()
        # self.skip_emb = Block(config.dec_cell_size // 2, 64, 2**8)
        # tlg is a constant value to setup a logvar lower_bound
        self.tlg = nn.ParameterList()
        self.config = config
        self.cnt = 0

        self.var_list = []
        self.x_var_list = []
        self.dis_list = []
        self.x_y_dist_list = []
        self.center_var = []

        while(input_dim <= output_dim):
            self.layers.append(nn.Sequential(Block(input_dim, input_dim, input_dim * 2), Block(input_dim, input_dim * 2, input_dim * 2)))
            if self.cnt == 0:
                self.dis_layers.append(nn.Sequential(Block(input_dim * 2, input_dim * 2, input_dim * 4), Block(input_dim * 2, input_dim * 4, input_dim * 4)))
                self.dis_layers2.append(nn.Sequential(Block(input_dim * 2, input_dim * 2, input_dim * 4), Block(input_dim * 2, input_dim * 4, input_dim * 4)))
            else:
                # Here we additional consider sampled z
                self.dis_layers.append(nn.Sequential(Block(input_dim * 3, input_dim * 3, input_dim * 4), Block(input_dim * 3, input_dim * 4, input_dim * 4)))
                self.dis_layers2.append(nn.Sequential(Block(input_dim * 3, input_dim * 3, input_dim * 4), Block(input_dim * 3, input_dim * 4, input_dim * 4)))

            # self.means.append(nn.Parameter(torch.zeros(input_dim*2, requires_grad = True)))
            # self.logvars.append(nn.Parameter(torch.zeros(input_dim*2, requires_grad = True)))
            # self.tlg.append(nn.Parameter(-torch.log(2*torch.ones(input_dim*2, requires_grad = False)).unsqueeze(-1)))
            # self.y_z_info.append(nn.Sequential(Block(input_dim * 4, input_dim * 4), Block(input_dim*4,1), nn.Sigmoid()))
            # self.z_x_info.append(nn.Sequential(Block(input_dim * 4, input_dim * 4), Block(input_dim*4,1), nn.Sigmoid()))
            input_dim *= 2
            self.cnt += 1

    def dis_reparameterization(self, mu, logvar, sample=True, z = None):
        # z = z * torch.exp(logvar) + mu
        return mu + torch.exp(logvar) * torch.rand_like(logvar)

    def reparameterization_rand(self, mu, logvar, sample=True, z = None):
        rand = torch.rand_like(logvar)
        return mu + torch.exp(logvar) * rand, rand

    @staticmethod
    @torch.jit.script
    def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
        return torch.mean(torch.sum(-0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2),1),0)

    @staticmethod
    def gaussian_analytical_kl_standard(mu1, logsigma1):
        return torch.mean(torch.sum(0.5 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1) ** 2),1),0)

    @staticmethod
    @torch.jit.script
    def gumbel_sampling(samples):
        ratio = torch.sqrt(torch.sum(samples*samples, -1))
        samples = samples / ratio.unsqueeze(-1)
        sc = samples @ samples.transpose(0, 1)
        dig = torch.diag(torch.ones_like(ratio) * (torch.max(sc)))
        scores = torch.log(1 + 1e-6 +  sc - dig) # remove the diag scores
        gumbels = (-torch.empty_like(scores, memory_format=torch.legacy_contiguous_format).exponential_().log())
        # print(scores)
        index = (scores+gumbels).max(-1)[1]
        return index

    @staticmethod
    @torch.jit.script
    def gaussian_merge(mean1, mean2, logvar1, logvar2):
        var1, var2 = logvar1.exp(), logvar2.exp()
        mean = ( var1 * mean2 + var2 * mean1 ) / (var1 + var2)
        var = 2 * (var1*var2)/(var1+var2) # Var Non decrease
        return mean, var.log()

    def forward(self, output, act1, act2, pos_embeddings, restarted):
        rkl = 0
        last_t_sample = None
        last_sample = None

        for index, (layer, dis_layer, dis_layer2, tm, tvar, tg, m_layer) in enumerate(zip(self.layers, self.dis_layers, self.dis_layers2, self.means, self.logvars, self.tlg, self.y_z_info)):
            if not restarted and not self.config.ost:
                if last_sample is None:
                    mean, logvar = dis_layer(act1[-index-1] + act2[-index-1]).chunk(2, -1)
                    t_mean, t_logvar = dis_layer2(act2[-index-1]).chunk(2, -1)
                else:
                    mean, logvar = dis_layer(torch.cat([act1[-index-1] + act2[-index-1], last_sample], -1)).chunk(2, -1)
                    t_mean, t_logvar = dis_layer2(torch.cat([act2[-index-1], last_t_sample], -1)).chunk(2, -1)

                last_t_sample = t_mean # Just use t_mean
                # m_mean, m_logvar = self.gaussian_merge(t_mean, mean, t_logvar, logvar)
                rkl = rkl + self.config.expand_weight * self.gaussian_analytical_kl(t_mean, mean.detach(), t_logvar, logvar.detach()) 
                rkl = rkl + self.config.converge_weight * self.gaussian_analytical_kl(t_mean.detach(), mean, t_logvar.detach(), logvar) 
                # print(m_mean.shape,act2[-index-2].shape)
                last_sample = self.dis_reparameterization(m_mean, m_logvar) + act2[-index-1]

                # if not self.config.no_mutual_information:
                #     randperm = self.gumbel_sampling(last_sample.detach().clone())
                #     ng_t_mean = t_mean.detach()

                #     pos_scores = m_layer(torch.cat([last_t_sample, last_sample], -1))
                #     neg_scores = m_layer(torch.cat([last_t_sample, last_sample[randperm]], -1)) # Add constant to avoid too close to mean
                #     rkl = rkl - torch.mean(torch.log(pos_scores + 1e-6) + torch.log(1 - neg_scores + 1e-6)) #Add mutual information loss

                output = layer(output) + last_sample
            else:
                if last_sample is None:
                    t_mean,t_logvar = dis_layer2(act2[-index-1]).chunk(2, -1)
                    if restarted:
                        mean,logvar = dis_layer(act1[-index-1]).chunk(2, -1)
                        self.dis_list.append(self.gaussian_analytical_kl(mean.detach(), t_mean.detach(), logvar.detach(), t_logvar.detach()))
                        self.var_list.append(torch.mean(t_logvar.detach().clone().nan_to_num_()))
                        self.x_var_list.append(torch.mean(logvar.detach().clone().nan_to_num_()))
                        self.x_y_dist_list.append(torch.mean(torch.abs(t_mean.detach()-mean.detach())))
                        m_mean_var = torch.mean(t_mean.detach(), 0).unsqueeze(0) - t_mean.detach()
                        self.center_var.append(torch.mean(m_mean_var * m_mean_var))
                else:
                    mean,logvar = dis_layer(torch.cat([act1[-index-1], last_sample], -1)).chunk(2, -1)
                    t_mean,t_logvar = dis_layer2(torch.cat([act2[-index-1], last_sample], -1)).chunk(2, -1)
                    if restarted:
                        self.dis_list[-1] += self.gaussian_analytical_kl(mean.detach(), t_mean.detach(), logvar.detach(), t_logvar.detach())
                        self.var_list[-1] += torch.mean(t_logvar.detach().clone().nan_to_num_())
                        self.x_var_list[-1] += torch.mean(logvar.detach().clone().nan_to_num_())
                        self.x_y_dist_list[-1] += torch.mean(torch.abs(t_mean.detach()-mean.detach()))
                        m_mean_var = torch.mean(t_mean.detach(), 0).unsqueeze(0) - t_mean.detach()
                        self.center_var[-1] += torch.mean(m_mean_var * m_mean_var)

                if self.training:
                    last_sample = self.dis_reparameterization(t_mean, t_logvar) + act2[-index-1]
                    m_mean, m_logvar = self.gaussian_merge(t_mean, mean, t_logvar, logvar)
                    rkl = rkl + self.config.expand_weight * self.gaussian_analytical_kl(t_mean, m_mean.detach(), t_logvar, m_logvar.detach()) 
                    rkl = rkl + self.config.converge_weight * self.gaussian_analytical_kl(m_mean.detach(), mean, m_logvar.detach(), logvar) 
                else:
                    last_sample = t_mean + act2[-index-1]
                output = torch.nan_to_num(layer(output) + last_sample)
        # return output, rkl
        return output, rkl


    def sample_forward(self, output, act1, act2, sample_size):
        # act1 for rec, act2 for pri
        rkl = 0
        output = output.repeat(sample_size, 1) #Only Use CLS
        last_t_sample = None
        last_sample = None
        rand_id = []

        for index, (layer, dis_layer, dis_layer2, tm, tvar, tg, m_layer) in enumerate(zip(self.layers, self.dis_layers, self.dis_layers2, self.means, self.logvars, self.tlg, self.y_z_info)):
            if last_sample is None:
                t_mean,t_logvar = dis_layer2(act2[-index-1]).repeat(sample_size,1).chunk(2, -1)
                mean,logvar = dis_layer(act1[-index-1]).repeat(sample_size,1).chunk(2, -1)
                self.dis_list.append(self.gaussian_analytical_kl(mean.detach(), t_mean.detach(), logvar.detach(), t_logvar.detach()))
                self.var_list.append(torch.mean(t_logvar.detach().clone().nan_to_num_()))
                self.x_var_list.append(torch.mean(logvar.detach().clone().nan_to_num_()))
                self.x_y_dist_list.append(torch.mean(torch.abs(t_mean.detach()-mean.detach())))
                m_mean_var = torch.mean(t_mean.detach(), 0).unsqueeze(0) - t_mean.detach()
                self.center_var.append(torch.mean(m_mean_var * m_mean_var))
                last_sample, rand = self.reparameterization_rand(t_mean, t_logvar)
                rand_id.append(rand)
            else:
                mean,logvar = dis_layer(torch.cat([act1[-index-1].repeat(sample_size,1), last_sample], -1)).chunk(2, -1)
                t_mean,t_logvar = dis_layer2(torch.cat([act2[-index-1].repeat(sample_size,1), last_sample], -1)).chunk(2, -1)
                self.dis_list[-1] += self.gaussian_analytical_kl(mean.detach(), t_mean.detach(), logvar.detach(), t_logvar.detach())
                self.var_list[-1] += torch.mean(t_logvar.detach().clone().nan_to_num_())
                self.x_var_list[-1] += torch.mean(logvar.detach().clone().nan_to_num_())
                self.x_y_dist_list[-1] += torch.mean(torch.abs(t_mean.detach()-mean.detach()))
                m_mean_var = torch.mean(t_mean.detach(), 0).unsqueeze(0) - t_mean.detach()
                self.center_var[-1] += torch.mean(m_mean_var * m_mean_var)
                last_sample = self.dis_reparameterization(t_mean, t_logvar)
            output = torch.nan_to_num(layer(output) + last_sample)
        return output, rkl, rand_id

class DIS_VAE(GMMBase):
    def get_pos_layer(self, latent_size):
        return nn.Sequential(nn.Linear(1, latent_size), nn.GELU(), nn.Linear(latent_size, latent_size),nn.Tanh())

    # def get_gpt_layer(self, config, scale):
    #     return None

    def __init__(self, corpus, config, data_loader=None):
        super(DIS_VAE, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        config.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.unk_id = self.rev_vocab[UNK]
        
        self.user_dict = corpus.use_dict
        self.user_list = list(self.user_dict.keys())
        self.item_dict = corpus.item_dict
        self.item_list = list(self.item_dict.keys())
        self.feature_dict = corpus.feature_dict

        # require user_token_size == item_token_size
        # self.user_tokens = torch.nn.Embedding(config.user_token_cnt * config.user_token_cnt, config.user_token_size)
        # self.item_tokens = torch.nn.Embedding(config.item_token_cnt * config.item_token_cnt, config.item_token_size)
        self.ui_sp_tokens = torch.nn.Embedding(3, config.user_token_size)
        # self.rec_sp_tokens = torch.nn.Embedding(3, config.user_token_size)
        self.rkl_weight = 1

        # self.ui_cls_emb = self.ui_sp_tokens.weight[0]
        # self.ui_sep_emb = self.ui_sp_tokens.weight[1]
        self.ui_pos_emb = torch.nn.Embedding(2 + config.user_token_cnt + config.item_token_cnt, config.user_token_size)

        # self.rec_cls_emb = self.rec_sp_tokens.weight[0].expand(config.batch_size, 1, config.user_token_size)
        # self.rec_sep_emb = self.rec_sp_tokens.weight[1].expand(config.batch_size, 1, config.user_token_size)
        
        #Setting up GPT decoder model
        gpt_config = GPT2Config()
        gpt_config.bos_token_id = self.go_id
        gpt_config.eos_id = self.eos_id
        gpt_config.n_ctx = gpt_config.n_embd = config.embed_size
        gpt_config.vocab_size = self.vocab_size
        gpt_config.n_head = 16 
        gpt_config.n_layer = 2
        gpt_config.n_positions = config.max_seq_len + 1 #The first two token is used as a guide
        gpt_config.vocab_size = self.vocab_size
        gpt_config.scale_attn_weights = False #!!!
        gpt_config.padding_id = self.rev_vocab[PAD]

        #Setting up BERT encoder model
        # bert_config = BertConfig()
        # bert_config.hidden_size = config.embed_size
        # bert_config.vocab_size = self.vocab_size
        # bert_config.num_attention_heads = 16
        # bert_config.num_hidden_layers = 2
        # bert_config.max_position_embeddings = config.max_seq_len + 1
        # bert_config.intermediate_size = config.embed_size * 4

        
        self.x_decoder = GPT2Model(gpt_config)
        # self.x_encoder = BertModel(bert_config)

        self.x_embedding = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.rev_vocab[PAD]) #Word Embedding
        self.x_decoder.wte = self.x_embedding
        # self.x_encoder.embeddings.word_embeddings = self.x_embedding

        self.emb_size = config.dec_cell_size // 2
        emb_size = config.dec_cell_size // 2


        self.user_embedding = nn.Embedding(len(self.user_dict), config.user_token_cnt * config.user_token_size) 
        self.item_embedding = nn.Embedding(len(self.item_dict), config.item_token_cnt * config.user_token_size) 
        self.rating_embedding = nn.Embedding(2, config.user_token_size)


        #==============================New tranlsation here===========================
        # self.user_emb_trans = Block(config.user_token_size, config.user_token_size, config.user_token_size)
        # self.item_emb_trans = Block(config.item_token_size, config.item_token_size, config.item_token_size)


        # self.user_emb_trans = nn.Linear(config.user_token_cnt, config.user_token_size)
        # self.item_emb_trans = nn.Linear(config.item_token_cnt, config.item_token_size)

        # self.user_embedding = nn.Embedding(len(self.user_dict), self.emb_size) # Give 128
        # self.item_embedding = nn.Embedding(len(self.item_dict), self.emb_size) # Give 128
        # self.feature_embedding = nn.Embedding(len(self.feature_dict), self.emb_size)

        self.restarted = False
        
        if config.few_shot:
            self.user_store = torch.zeros(len(self.user_dict), config.dec_cell_size // 4).cuda()
            self.item_store = torch.zeros(len(self.item_dict), config.dec_cell_size // 4).cuda()
            self.rating_store = torch.zeros(5, config.dec_cell_size // 4).cuda()
            self.user_cnt = torch.zeros(len(self.user_dict)).cuda()
            self.item_cnt = torch.zeros(len(self.item_dict)).cuda()
            self.rating_cnt = torch.zeros(5).cuda()

        # self.feature_embedding.weight.requires_grad = False
    
        # nn.init.zeros_(self.feature_embedding.weight)
        

        # 512 word 1024 hidden  256 latent_size

        # self.rating_embedding.weight.requires_grad = False

        self.ls_size = config.dec_cell_size // 2
        self.latent_size =  config.dec_cell_size // 4
        self.latent_size2 =  self.latent_size // 2
        self.latent_size3 =  self.latent_size2 // 2
        self.latent_size4 =  self.latent_size3 // 2
        self.latent_size5 =  self.latent_size4 // 2
        self.latent_size6 =  self.latent_size5 // 2

        # self.rec_pos_emb = torch.nn.Embedding(config.max_seq_len, self.ls_size)
        # self.rec_token_emb = torch.nn.Embedding(len(self.vocab), self.ls_size)

        self.rec_pos_emb = torch.nn.Embedding(config.max_seq_len, self.latent_size)
        self.rec_token_emb = torch.nn.Embedding(len(self.vocab), self.latent_size)

        # self.pos_embedding_generate = nn.Sequential(nn.Linear(1, self.latent_size6), nn.GELU(), nn.Linear(self.latent_size6, self.latent_size6),nn.Tanh())

        # self.pos_embedding = nn.ModuleList()
        # self.pos_embedding.append(self.get_pos_layer(self.latent_size3))
        # self.pos_embedding.append(self.get_pos_layer(self.latent_size4))
        # self.pos_embedding.append(self.get_pos_layer(self.latent_size5))


        for dl in data_loader:
            # dl.embedding = self.x_embedding
            dl.rev_vocab = corpus.rev_vocab
            dl.sampler = RandomSampler(dl.data)
            dl.dataloader = DataLoader(dl.data, config.batch_size, sampler=dl.sampler, collate_fn=dl.batcher(dl.rev_vocab,config.max_seq_len, torch.cuda.current_device()), num_workers=1)

        
        self.bn = nn.BatchNorm1d(self.latent_size6,affine=False,eps=1e-8,track_running_stats=False)

        # self.scale = torch.nn.Parameter(torch.tensor(0.0)) 
        # self.scale2 = torch.nn.Parameter(torch.tensor(0.0))
        

        self.backLoss = True
        self.GEN_MODE = False
        self.sample_mode = False

        self.dropout = nn.Dropout(0.2)
        self.mse_loss = torch.nn.MSELoss(reduction='sum')
        self.mse_scale = 20
        self.zkl_scale = 1
        self.nlgeval = None
        
        # self.white = nn.Linear(bert_config.hidden_size, self.latent_size)
        # 101 == [CLS]

        self.function = F.log_softmax
        self.decoder_size = config.embed_size

        if self.config.use_feature:
            dec_emb_size = config.dec_cell_size // 4 * 3
        else:
            dec_emb_size = config.dec_cell_size // 2


        self.rec_encoder = DisRec(self.latent_size, self.latent_size5, config)
        self.pri_encoder = DisPri(config.user_token_size, self.latent_size5, config)
        self.sup_decoder = DisDec_Auto(self.latent_size6, self.latent_size5, config)
        # self.out_maker = Block(self.latent_size6, self.latent_size5, self.latent_size5)
        

        # self.xtz = nn.Sequential(Block(config.dec_cell_size, config.dec_cell_size), 
        # Block(config.dec_cell_size, self.latent_size, config.dec_cell_size),
        # Block(self.latent_size , self.latent_size),
        # Block(self.latent_size , self.latent_size2, self.latent_size),
        # Block(self.latent_size2, self.latent_size2),
        # Block(self.latent_size2, self.latent_size3))


        # self.cxtz = nn.Sequential(Block(self.latent_size, self.latent_size), 
        # Block(self.latent_size, self.latent_size2),
        # Block(self.latent_size2, self.latent_size2),
        # Block(self.latent_size2, self.latent_size3))

        emb_size = config.dec_cell_size // 2
        emb3 = emb_size * 3

        # self.base_builder = nn.Sequential(Block(emb3, emb3, self.latent_size), Block(emb3, self.latent_size))

        self.tau = config.tau

        self.x_init_connector = nn.Sequential(
            # Block(self.latent_size6, self.latent_size6, self.latent_size5),
            # Block(self.latent_size6, self.latent_size5, self.latent_size5),
            # Block(self.latent_size5, self.latent_size5, self.latent_size4),
            # Block(self.latent_size5, self.latent_size4, self.latent_size4),
            Block(self.latent_size4, self.latent_size4, self.latent_size3),
            Block(self.latent_size4, self.latent_size3, self.latent_size3),
            Block(self.latent_size3, self.latent_size3, self.latent_size2),
            Block(self.latent_size3, self.latent_size2, self.latent_size3),
            Block(self.latent_size2, self.latent_size2, self.latent_size),
            Block(self.latent_size2, self.latent_size, self.latent_size),
            # nn_lib.LinearConnector(self.latent_size,config.dec_cell_size, config.rnn_cell == 'lstm')
            )
        self.expand_weight = config.expand_weight
        self.greedy_cat_connector = nn_lib.GreedyConnector()
        self.nll_loss = criterions.NLLEntropy(0, self.config)
        self.rating_loss = nn.MSELoss()
        self.nll_loss_word = criterions.NLLEntropy(0, self.config, avg_type="word")
        self.ppl = criterions.Perplexity(0, self.config)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.log_uniform_y = Variable(torch.log(torch.ones(1) / config.k))
        self.entropy_loss = criterions.Entropy()
        # self.uni_layer_norm = nn.LayerNorm(self.emb_size, elementwise_affine=False)

        self.steps = None
        self.E_steps = None

        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
        self.kl_w = 0.0
        self.mode = None
        self.sqrt = torch.sqrt

    def call(self, inputs, mode='positive',scale = None):
        if scale is None:
            scale = self.scale
        if mode == 'positive':
            scale = self.tau + (1 - self.tau) * torch.sigmoid(self.scale)
        else:
            scale = (1 - self.tau) * torch.sigmoid(-self.scale)
        return inputs * self.sqrt(scale)

    # def hash_rand(self, mu, logvar, d):
    #     return mu + torch.exp(logvar) * d

    def valid_loss(self, loss, batch_cnt=None, step=None):
        step = batch_cnt

        # if step is not None:
        #     vae_kl_weight = kl_anneal_function(self.config.anneal_function, step,
        #                                        self.config.anneal_k, self.config.anneal_x0) # Use beta here !
        # else:
        #     vae_kl_weight = 1.0
        vae_kl_weight = 1.0

        # return Pack(zkl=kld_loss, vae_nll=vae_nll, vae_PPL = vae_ppl, vae_emb=vae_embl*self.mse_scale, vae_rnll=vae_rnll)
        # mi_weight = 0.0 if self.config.use_mutual else 1.0

        # if loss.vae_rkl is None or torch.isinf(loss.vae_rkl) or torch.isnan(loss.vae_rkl):
        #     loss.vae_rkl = 0
        # if loss.vae_rnll is None or torch.isinf(loss.vae_rnll) or torch.isnan(loss.vae_rnll):
        #     loss.vae_rnll = 0
        # if loss.vae_zkl is None or torch.isinf(loss.vae_zkl) or torch.isnan(loss.vae_zkl):
        #     loss.vae_zkl = 0
        loss.vae_zkl = 0
        
        if loss.vae_nll is None:
            loss.vae_nll = 0

        if self.training:
            # vae_loss = loss.vae_nll + vae_kl_weight * self.config.beta * (loss.zkl) + loss.vae_rkl * 5 * vae_kl_weight + loss.vae_rnll
            # vae_loss = loss.vae_nll + loss.vae_zkl * vae_kl_weight + loss.vae_rnll  + loss.vae_rkl
            vae_loss = loss.vae_nll +  loss.vae_rkl
        else:
            vae_loss = loss.vae_nll
            # vae_loss = loss.vae_nll


        return vae_loss

    def set_pretrained_mode(self):
        self.mode = 0
        
    def set_normal_train_mode(self):
        self.mode = 1
        
    def pxz_forward(self, batch_size, results, out_utts, mode, gen_type, encoder_output=None,feature_emb=None):

        dec_init_state = self.x_init_connector(results.sample_z).squeeze(0) #ORI: 1,batch_size,dec_size
        dec_outs, dec_last, dec_ctx = self.x_decoder(dec_init_state, self.config, out_utts, mode)
        
        # dec_outs, dec_last, dec_ctx = self.x_decoder(batch_size, out_utts, dec_init_state,
        #                                              mode=mode, gen_type=gen_type,
        #                                              beam_size=self.config.beam_size,
        #                                              attn_context=None,latent_variable=feature_emb)
        results['dec_outs'] = dec_outs
        results['dec_ctx'] = dec_ctx
        return results
    
    @staticmethod
    def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
        y = torch.mean(torch.sum(-0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2),1),0)
        return y
        # return torch.nan_to_num(y)
    
    def gaussian_analytical_standard(self, z_mean, z_logvar):
        z_mean = torch.nan_to_num(z_mean)
        z_logvar = torch.nan_to_num(z_logvar)
        z_mean = self.uni_layer_norm(z_mean)
        z_logvar = self.uni_layer_norm(z_logvar)

        # if self.config.use_bnvae:
        #     z_mean = self.call(self.bn(z_mean),self.scale2)
        #     z_logvar = self.call(self.bn(z_logvar),'n',self.scale2)
        return torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mean ** 2 - z_logvar.exp(), dim = 1), dim = 0)

    
    @staticmethod
    def analysis_mu(mu1, mu2):
        return torch.mean(torch.sum((mu1 - mu2) ** 2,1),0)

    @staticmethod
    def check_loss(loss):
        if torch.isnan(loss) or torch.isnan(loss):
            return 0
        return loss

    def bounded_mse(self, input, target, logvar):
        diff = torch.abs(target - input)
        return torch.mean(torch.sum(torch.clamp_min(diff - 2*logvar.exp(),0)**2,1))

    @staticmethod
    def radd(input):
        return torch.cat([input[:,0].unsqueeze(-1),input],-1)

    def anneal_score(self):
        return float(1 / np.exp((self.steps/(self.E_steps * 0.7))**2 + np.log(1+self.steps)))
        # return float(1 / np.exp((self.steps/(self.E_steps * 0.7))**2))
    
    def forward(self, data_feed, mode, sample_n=1, gen_type='greedy', return_latent=False):
        
        real_feature_embs = None
        batch_size = data_feed['user_ids'].size(0)

        user_matrix = self.user_embedding(data_feed['user_ids']).view(batch_size, self.config.user_token_cnt, -1)
        item_matrix = self.item_embedding(data_feed['item_ids']).view(batch_size, self.config.item_token_cnt, -1)

        # if self.training and not self.restarted:
        #     user_matrix = F.gumbel_softmax(user_matrix, hard=True, tau=self.anneal_score())
        #     item_matrix = F.gumbel_softmax(item_matrix, hard=True, tau=self.anneal_score())
        # else:
        #     user_index = user_matrix.max(1, keepdim=True)[1]
        #     user_matrix = torch.zeros_like(user_matrix, memory_format=torch.legacy_contiguous_format).scatter_(1, user_index, 1.0)
        #     item_index = item_matrix.max(1, keepdim=True)[1]
        #     item_matrix = torch.zeros_like(item_matrix, memory_format=torch.legacy_contiguous_format).scatter_(1, item_index, 1.0)


        # user_token = self.user_tokens.weight.view(user_matrix.size(1), user_matrix.size(2), -1).unsqueeze(0)
        # item_token = self.item_tokens.weight.view(item_matrix.size(1), item_matrix.size(2), -1).unsqueeze(0)
        
        # user_token = torch.sum(user_token * user_matrix.unsqueeze(-1), 1)
        # item_token = torch.sum(item_token * item_matrix.unsqueeze(-1), 1)

        # user_token = self.user_emb_trans(user_matrix)
        # item_token = self.item_emb_trans(item_matrix)
        user_token = user_matrix
        item_token = item_matrix

        real_rating_embs = self.rating_embedding(data_feed['ratings'].ge(3).int())

        #ignore feature embedding
        # if self.config.use_feature:
        #     real_feature_embs = self.feature_embedding(data_feed['feature_ids'])
            
        pos_embds = []
        #Ignore Position Embeddings
        # for i in range(3):
        #     # Manually Do reverse here
        #     pos_embds.append(self.pos_embedding[2-i](data_feed['rd_ids'].unsqueeze(-1)))

        # u_tmp = self.base_builder(torch.cat([real_user_embs, real_item_embs, real_rating_embs],-1))
        idx = torch.zeros_like(user_token[:,0,0]).unsqueeze(-1).long()
        # ui = torch.cat([self.ui_sp_tokens(idx), user_token, self.ui_sp_tokens(idx+1), item_token, self.ui_sp_tokens(idx+2)], 1) + self.ui_pos_emb.weight
        ui = torch.cat([self.ui_sp_tokens(idx), user_token, item_token, real_rating_embs.unsqueeze(1)], 1) + self.ui_pos_emb.weight

        # if user_token.size(0) != self.ui_cls_emb.size(0):
        #     tmp_batch_size = user_token.size(0)
        #     ui = torch.cat([self.ui_cls_emb[:tmp_batch_size], user_token, self.ui_sep_emb[:tmp_batch_size], item_token, self.ui_sep_emb[:tmp_batch_size]], 1) + self.ui_pos_emb.weight
        # else:
        #     ui = torch.cat([self.ui_cls_emb, user_token, self.ui_sep_emb, item_token, self.ui_sep_emb], 1) + self.ui_pos_emb.weight

        words =  self.rec_token_emb(data_feed['sentence_embeddings']) + self.rec_pos_emb.weight
        z_pri,pri_state = self.pri_encoder(ui, torch.ones_like(ui[:,:,0]))
        z_rec,rec_state = self.rec_encoder(words, data_feed['attention_masks'])
        
        vae_nll = None
        vae_ppl = None
        if not self.sample_mode:
            # z_sample, rkl_loss = self.sup_decoder(z_rec, rec_state, pri_state, pos_embds, self.restarted)
            # if self.training:
            #     rec_mean, rec_logvar = self.out_maker(z_rec[:, 0]).chunk(2, -1)
            #     pri_mean, pri_logvar = self.out_maker(z_pri[:, 0]).chunk(2, -1)
            #     rec_sample = self.sup_decoder.dis_reparameterization(rec_mean, rec_logvar)
            #     mi_kl = self.gaussian_analytical_kl(pri_mean, rec_mean.detach(), pri_logvar, rec_logvar.detach()) + self.gaussian_analytical_kl(pri_mean.detach(), rec_mean, pri_logvar.detach(), rec_logvar) * 0.5
            #     z_sample, rkl_loss = self.sup_decoder(rec_sample, rec_state, pri_state, pos_embds, self.restarted)
            # else:
            #     if self.restarted:
            #         pri_mean, pri_logvar = self.out_maker(z_pri[:, 0]).chunk(2, -1)
            #         z_sample, rkl_loss = self.sup_decoder(pri_mean, rec_state, pri_state, pos_embds, self.restarted)
            #     else:
            #         rec_mean, rec_logvar = self.out_maker(z_rec[:, 0]).chunk(2, -1)
            #         rec_sample = self.sup_decoder.dis_reparameterization(rec_mean, rec_logvar)
            #         z_sample, rkl_loss = self.sup_decoder(rec_sample, rec_state, pri_state, pos_embds, self.restarted)
            #     # z_sample, rkl_loss = self.sup_decoder((z_pri[:,0] + z_rec[:,0]) / 2, rec_state, pri_state, pos_embds, self.restarted)
            z_sample, rkl_loss = self.sup_decoder(z_pri[:, 0], rec_state, pri_state, pos_embds, self.restarted)

            result = Pack(sample_z = z_sample)
            vae_resp = self.pxz_forward(data_feed['sentence_embeddings'].size(0), result, data_feed['ans_ids'], mode, gen_type,feature_emb=real_feature_embs)

            # if self.restarted:
            #     vae_nll = None
            #     vae_ppl = None
            # else:
            vae_nll = self.nll_loss(vae_resp['dec_outs'], data_feed['ans_ids'])
            vae_ppl = self.ppl(vae_resp['dec_outs'], data_feed['ans_ids'])
            # vae_nll = 0
            # vae_ppl = 0
        else:
            sample_size = 2
            # pri_mean, pri_logvar = self.out_maker(z_pri[:, 0]).chunk(2, -1)
            # z_sample, rkl_loss, rand_id = self.sup_decoder.sample_forward(pri_mean, rec_state, pri_state, sample_size)
            z_sample, rkl_loss, rand_id = self.sup_decoder.sample_forward(z_pri[:, 0], rec_state, pri_state, sample_size)
            # z_sample, rkl_loss = self.sup_decoder.sample_forward(z_rec, rec_state, pri_state, sample_size)
            result = Pack(sample_z = z_sample, rand_id = rand_id)
            vae_resp = self.pxz_forward(data_feed['sentence_embeddings'].size(0) * sample_size, result, data_feed['ans_ids'].repeat(sample_size, 1), mode, gen_type,feature_emb=real_feature_embs)

        vae_rkl = None

        # print(rkl_loss, self.rkl_weight)

        if self.backLoss:
                # vae_rkl = rkl_loss + mi_kl
            if self.training:
                # vae_rkl = rkl_loss + self.rating_loss(z_pri[:,0].detach(), z_rec[:,0]) * 0.1 + self.rating_loss(z_pri[:,0], z_rec[:,0].detach()) + mi_kl
                vae_rkl = rkl_loss
                if  vae_rkl < self.config.rkl_bound and self.rkl_weight >= 1e-6:
                    self.rkl_weight *= 0.125
                elif self.rkl_weight < self.config.rkl_bound:
                    self.rkl_weight *= 2
                    # vae_rkl = rkl_loss * self.rkl_weight + mi_kl
                return Pack(vae_zkl=None, vae_nll=vae_nll, vae_PPL = vae_ppl, vae_rnll=None, vae_rkl=vae_rkl * self.rkl_weight)
            else:
                return Pack(vae_zkl=None, vae_nll=vae_nll, vae_PPL = vae_ppl, vae_rnll=None, vae_rkl=None)
        else:
            # if self.config.do_pred:
            #     return vae_resp['dec_ctx'], data_feed['ans_ids'], self.dnn(torch.cat([real_user_embs, real_item_embs],-1)).squeeze(-1)
            # else:
            if not self.sample_mode:
                return vae_resp['dec_ctx'], data_feed['ans_ids']
            else:
                return vae_resp['dec_ctx'], data_feed['ans_ids'], result['rand_id']
        return result


    def get_freeze_optimizer(self, config):
        if config.op == 'adamw':
            no_decay = ['bias', 'LayerNorm.weight']
            bert_compent_name = ['x_encoder','xtz','x_embedding','x_decoder','x_init_connector']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_compent_name)], 'weight_decay': config.weight_decay},
                {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_compent_name)], 'weight_decay': 0.0},
                ]
            print([n for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_compent_name)])
            print("Not use this optimizer!")
            exit(0)
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.init_lr, eps=1e-6)
        if config.op == 'adam':
            self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,
                                           self.parameters()), lr=config.init_lr)
        elif config.op == 'sgd':
            self.optimizer =torch.optim.SGD(self.parameters(), lr=config.init_lr,
                                   momentum=config.momentum)
        elif config.op == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=config.init_lr,
                                       momentum=config.momentum)
        self.scheduler = transformers.get_constant_schedule_with_warmup(self.optimizer,config.warmup_steps)
        return self.optimizer


class DeN_Auto(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dis_layers = nn.ModuleList()
        self.dis_layers2 = nn.ModuleList()
        self.y_z_info = nn.ModuleList()
        # self.means = nn.ParameterList()
        # self.logvars = nn.ParameterList()
        # self.skip_emb = Block(config.dec_cell_size // 2, 64, 2**8)
        # tlg is a constant value to setup a logvar lower_bound
        # self.tlg = nn.ParameterList()
        self.config = config
        self.cnt = 0

        self.var_list = []
        self.x_var_list = []
        self.dis_list = []
        self.x_y_dist_list = []
        self.center_var = []

        while(input_dim <= output_dim):
            # self.layers.append(nn.Sequential(Block(input_dim, input_dim, input_dim * 2), Block(input_dim, input_dim * 2, input_dim * 2)))
            self.layers.append(Block(input_dim, input_dim * 2, input_dim * 2))
            if self.cnt == 0:
                # self.dis_layers.append(nn.Sequential(Block(input_dim * 2, input_dim * 2, input_dim * 4), Block(input_dim * 2, input_dim * 4, input_dim * 4)))
                # self.dis_layers2.append(nn.Sequential(Block(input_dim * 2, input_dim * 2, input_dim * 4), Block(input_dim * 2, input_dim * 4, input_dim * 4)))
                self.dis_layers.append(Block(input_dim * 2, input_dim * 4, input_dim * 4))
                self.dis_layers2.append(Block(input_dim * 2, input_dim * 4, input_dim * 4))
            else:
                # Here we additional consider sampled z
                # self.dis_layers.append(nn.Sequential(Block(input_dim * 3, input_dim * 3, input_dim * 4), Block(input_dim * 3, input_dim * 4, input_dim * 4)))
                # self.dis_layers2.append(nn.Sequential(Block(input_dim * 3, input_dim * 3, input_dim * 4), Block(input_dim * 3, input_dim * 4, input_dim * 4)))
                self.dis_layers.append(Block(input_dim * 3, input_dim * 4, input_dim * 4))
                self.dis_layers2.append(Block(input_dim * 3, input_dim * 4, input_dim * 4))

            # self.means.append(nn.Parameter(torch.zeros(input_dim*2, requires_grad = True)))
            # self.logvars.append(nn.Parameter(torch.zeros(input_dim*2, requires_grad = True)))
            # self.tlg.append(nn.Parameter(-torch.log(2*torch.ones(input_dim*2, requires_grad = False)).unsqueeze(-1)))
            self.y_z_info.append(nn.Sequential(Block(input_dim * 4, input_dim * 4), Block(input_dim*4, 1, input_dim), nn.Sigmoid()))
            # self.z_x_info.append(nn.Sequential(Block(input_dim * 4, input_dim * 4), Block(input_dim*4,1), nn.Sigmoid()))
            input_dim *= 2
            self.cnt += 1

    @staticmethod
    def dis_reparameterization(mu, logvar):
        # z = z * torch.exp(logvar) + mu
        return mu + torch.exp(logvar) * torch.rand_like(logvar)

    def reparameterization_rand(self, mu, logvar, sample=True, z = None):
        rand = torch.rand_like(logvar)
        return mu + torch.exp(logvar) * rand, rand

    @staticmethod
    @torch.jit.script
    def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
        # return torch.mean(torch.nan_to_num(torch.sum(-0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2),1)),0)
        return torch.mean(torch.sum(-0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2),1),0)

    @staticmethod
    def gaussian_analytical_kl_standard(mu1, logsigma1):
        return torch.mean(torch.sum(0.5 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1) ** 2),1),0)

    @staticmethod
    @torch.jit.script
    def gumbel_sampling(samples):
        ratio = torch.sqrt(torch.sum(samples*samples, -1))
        samples = samples / ratio.unsqueeze(-1)
        sc = samples @ samples.transpose(0, 1)
        dig = torch.diag(torch.ones_like(ratio) * (torch.max(sc)))
        scores = torch.log(1 + 1e-6 +  sc - dig) # remove the diag scores
        gumbels = (-torch.empty_like(scores, memory_format=torch.legacy_contiguous_format).exponential_().log())
        # print(scores)
        index = (scores+gumbels).max(-1)[1]
        return index

    @staticmethod
    @torch.jit.script
    def gaussian_merge(mean1, mean2, logvar1, logvar2):
        var1, var2 = logvar1.exp(), logvar2.exp()
        mean = (var1 * mean2 + var2 * mean1 ) / (var1 + var2)
        var = 2 * (var1*var2)/(var1+var2) # Var Non decrease
        return mean, var.log()

    def forward(self, output, act1, act2, pos_embeddings, restarted):
        # act1 for rec, act2 for pri
        rkl = 0
        # output = output[:,0] #Only Use CLS
        last_t_sample = None
        last_sample = None
        # output = torch.mean(output,1) 
        # Skip_Emb = act1[0]

        for index, (layer, dis_layer, dis_layer2, m_layer) in enumerate(zip(self.layers, self.dis_layers, self.dis_layers2, self.y_z_info)):
            if not restarted and not self.config.ost:
                if last_sample is None:
                    mean, logvar = dis_layer(act1[-index-1] + act2[-index-1]).chunk(2, -1)
                    t_mean, t_logvar = dis_layer2(act2[-index-1]).chunk(2, -1)
                else:
                    mean, logvar = dis_layer(torch.cat([act1[-index-1] + act2[-index-1], last_sample], -1)).chunk(2, -1)
                    t_mean, t_logvar = dis_layer2(torch.cat([act2[-index-1], last_t_sample], -1)).chunk(2, -1)

                last_t_sample = t_mean # Just use t_mean

                m_mean, m_logvar = self.gaussian_merge(t_mean, mean, t_logvar, logvar)
                rkl = rkl + self.config.expand_weight * self.gaussian_analytical_kl(t_mean, m_mean.detach(), t_logvar, m_logvar.detach()) 
                rkl = rkl + self.config.converge_weight * self.gaussian_analytical_kl(m_mean.detach(), mean, m_logvar.detach(), logvar) 
                # print(m_mean.shape,act2[-index-2].shape)
                last_sample = self.dis_reparameterization(m_mean, m_logvar) + act2[-index-1]
                
                if not self.config.no_mutual_information:
                    randperm = self.gumbel_sampling(last_sample.detach().clone())
                    ng_t_mean = t_mean.detach()

                    pos_scores = m_layer(torch.cat([last_t_sample, last_sample], -1))
                    neg_scores = m_layer(torch.cat([last_t_sample, last_sample[randperm]], -1)) # Add constant to avoid too close to mean
                    rkl = rkl - torch.mean(torch.log(pos_scores + 1e-6) + torch.log(1 - neg_scores + 1e-6)) #Add mutual information loss

                output = layer(output) + last_sample
            else:
                if last_sample is None:
                    t_mean,t_logvar = dis_layer2(act2[-index-1]).chunk(2, -1)
                    if restarted:
                        mean,logvar = dis_layer(act1[-index-1]).chunk(2, -1)
                        self.dis_list.append(self.gaussian_analytical_kl(mean.detach(), t_mean.detach(), logvar.detach(), t_logvar.detach()))
                        self.var_list.append(torch.mean(t_logvar.detach().clone().nan_to_num_()))
                        self.x_var_list.append(torch.mean(logvar.detach().clone().nan_to_num_()))
                        self.x_y_dist_list.append(torch.mean(torch.abs(t_mean.detach()-mean.detach())))
                        m_mean_var = torch.mean(t_mean.detach(), 0).unsqueeze(0) - t_mean.detach()
                        self.center_var.append(torch.mean(m_mean_var * m_mean_var))
                else:
                    mean,logvar = dis_layer(torch.cat([act1[-index-1], last_sample], -1)).chunk(2, -1)
                    t_mean,t_logvar = dis_layer2(torch.cat([act2[-index-1], last_sample], -1)).chunk(2, -1)
                    if restarted:
                        self.dis_list[-1] += self.gaussian_analytical_kl(mean.detach(), t_mean.detach(), logvar.detach(), t_logvar.detach())
                        self.var_list[-1] += torch.mean(t_logvar.detach().clone().nan_to_num_())
                        self.x_var_list[-1] += torch.mean(logvar.detach().clone().nan_to_num_())
                        self.x_y_dist_list[-1] += torch.mean(torch.abs(t_mean.detach()-mean.detach()))
                        m_mean_var = torch.mean(t_mean.detach(), 0).unsqueeze(0) - t_mean.detach()
                        self.center_var[-1] += torch.mean(m_mean_var * m_mean_var)

                # rkl = rkl + self.gaussian_analytical_kl(t_mean, mean, t_logvar, logvar) / (index + 1) 
                # rkl = rkl + self.config.converge_weight * self.gaussian_analytical_kl(t_mean.detach(), mean, t_logvar.detach(), logvar) / (index + 1) 

                if self.training:
                    last_sample = self.dis_reparameterization(t_mean, t_logvar) + act2[-index-1]
                    m_mean, m_logvar = self.gaussian_merge(t_mean, mean, t_logvar, logvar)
                    rkl = rkl + self.config.expand_weight * self.gaussian_analytical_kl(t_mean, m_mean.detach(), t_logvar, m_logvar.detach()) 
                    rkl = rkl + self.config.converge_weight * self.gaussian_analytical_kl(m_mean.detach(), mean, m_logvar.detach(), logvar) 
                else:
                    last_sample = t_mean + act2[-index-1]
                output = torch.nan_to_num(layer(output) + last_sample)
        # return output, rkl
        return output, rkl


    def sample_forward(self, output, act1, act2, sample_size):
        # act1 for rec, act2 for pri
        rkl = 0
        output = output.repeat(sample_size, 1) #Only Use CLS
        last_t_sample = None
        last_sample = None
        rand_id = []

        for index, (layer, dis_layer, dis_layer2, m_layer) in enumerate(zip(self.layers, self.dis_layers, self.dis_layers2, self.y_z_info)):
            if last_sample is None:
                t_mean,t_logvar = dis_layer2(act2[-index-1]).repeat(sample_size,1).chunk(2, -1)
                mean,logvar = dis_layer(act1[-index-1]).repeat(sample_size,1).chunk(2, -1)
                self.dis_list.append(self.gaussian_analytical_kl(mean.detach(), t_mean.detach(), logvar.detach(), t_logvar.detach()))
                self.var_list.append(torch.mean(t_logvar.detach().clone().nan_to_num_()))
                self.x_var_list.append(torch.mean(logvar.detach().clone().nan_to_num_()))
                self.x_y_dist_list.append(torch.mean(torch.abs(t_mean.detach()-mean.detach())))
                m_mean_var = torch.mean(t_mean.detach(), 0).unsqueeze(0) - t_mean.detach()
                self.center_var.append(torch.mean(m_mean_var * m_mean_var))
                last_sample, rand = self.reparameterization_rand(t_mean, t_logvar)
                rand_id.append(rand)
            else:
                mean,logvar = dis_layer(torch.cat([act1[-index-1].repeat(sample_size,1), last_sample], -1)).chunk(2, -1)
                t_mean,t_logvar = dis_layer2(torch.cat([act2[-index-1].repeat(sample_size,1), last_sample], -1)).chunk(2, -1)
                self.dis_list[-1] += self.gaussian_analytical_kl(mean.detach(), t_mean.detach(), logvar.detach(), t_logvar.detach())
                self.var_list[-1] += torch.mean(t_logvar.detach().clone().nan_to_num_())
                self.x_var_list[-1] += torch.mean(logvar.detach().clone().nan_to_num_())
                self.x_y_dist_list[-1] += torch.mean(torch.abs(t_mean.detach()-mean.detach()))
                m_mean_var = torch.mean(t_mean.detach(), 0).unsqueeze(0) - t_mean.detach()
                self.center_var[-1] += torch.mean(m_mean_var * m_mean_var)
                last_sample = self.dis_reparameterization(t_mean, t_logvar)
            output = torch.nan_to_num(layer(output) + last_sample)
        return output, rkl, rand_id


class DeN_VAE(GMMBase):
    def get_pos_layer(self, latent_size):
        return nn.Sequential(nn.Linear(1, latent_size), nn.GELU(), nn.Linear(latent_size, latent_size),nn.Tanh())

    def __init__(self, corpus, config, data_loader=None):
        super(DeN_VAE, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        config.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.unk_id = self.rev_vocab[UNK]
        
        self.user_dict = corpus.use_dict
        self.user_list = list(self.user_dict.keys())
        self.item_dict = corpus.item_dict
        self.item_list = list(self.item_dict.keys())
        self.feature_dict = corpus.feature_dict

        # require user_token_size == item_token_size
        self.ui_sp_tokens = torch.nn.Embedding(3, config.user_token_size)
        self.rkl_weight = 1

        self.ui_pos_emb = torch.nn.Embedding(2 + config.user_token_cnt + config.item_token_cnt, config.user_token_size)

        # self.x_encoder = BertModel(bert_config)

        self.x_embedding = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.rev_vocab[PAD]) #Word Embedding
        # self.x_decoder.wte = self.x_embedding
        # self.x_encoder.embeddings.word_embeddings = self.x_embedding

        self.emb_size = config.dec_cell_size // 2
        emb_size = config.dec_cell_size // 2


        self.user_embedding = nn.Embedding(len(self.user_dict), config.user_token_cnt * config.user_token_size) 
        self.item_embedding = nn.Embedding(len(self.item_dict), config.item_token_cnt * config.user_token_size) 
        self.rating_embedding = nn.Embedding(2, config.user_token_size)

        self.restarted = False
        
        if config.few_shot:
            self.user_store = torch.zeros(len(self.user_dict), config.dec_cell_size // 4).cuda()
            self.item_store = torch.zeros(len(self.item_dict), config.dec_cell_size // 4).cuda()
            self.rating_store = torch.zeros(5, config.dec_cell_size // 4).cuda()
            self.user_cnt = torch.zeros(len(self.user_dict)).cuda()
            self.item_cnt = torch.zeros(len(self.item_dict)).cuda()
            self.rating_cnt = torch.zeros(5).cuda()

        # 512 word 1024 hidden  256 latent_size

        # self.rating_embedding.weight.requires_grad = False

        self.ls_size = config.dec_cell_size // 2
        self.latent_size =  config.dec_cell_size // 4
        self.latent_size2 =  self.latent_size // 2
        self.latent_size3 =  self.latent_size2 // 2
        self.latent_size4 =  self.latent_size3 // 2
        self.latent_size5 =  self.latent_size4 // 2
        self.latent_size6 =  self.latent_size5 // 2

        #Setting up GPT decoder model
        gpt_config = GPT2Config()
        gpt_config.bos_token_id = self.go_id
        gpt_config.eos_id = self.eos_id
        gpt_config.n_ctx = gpt_config.n_embd = gpt_config.hidden_size = self.latent_size4
        gpt_config.vocab_size = self.vocab_size
        gpt_config.n_head = 4 
        gpt_config.n_layer = 2
        gpt_config.n_positions = config.max_seq_len + 1 #The first two token is used as a guide
        gpt_config.vocab_size = self.vocab_size
        gpt_config.scale_attn_weights = False #!!!
        gpt_config.padding_id = self.rev_vocab[PAD]

        self.x_decoder = GPT2Model(gpt_config)

        # self.rec_pos_emb = torch.nn.Embedding(config.max_seq_len, self.ls_size)
        # self.rec_token_emb = torch.nn.Embedding(len(self.vocab), self.ls_size)

        self.rec_pos_emb = torch.nn.Embedding(config.max_seq_len, self.latent_size)
        self.rec_token_emb = torch.nn.Embedding(len(self.vocab), self.latent_size)

        for dl in data_loader:
            # dl.embedding = self.x_embedding
            dl.rev_vocab = corpus.rev_vocab
            dl.sampler = RandomSampler(dl.data)
            dl.dataloader = DataLoader(dl.data, config.batch_size, sampler=dl.sampler, collate_fn=dl.batcher(dl.rev_vocab,config.max_seq_len, torch.cuda.current_device()), num_workers=1)

        
        self.bn = nn.BatchNorm1d(self.latent_size6,affine=False,eps=1e-8,track_running_stats=False)

        self.backLoss = True
        self.GEN_MODE = False
        self.sample_mode = False

        self.dropout = nn.Dropout(0.2)
        self.mse_loss = torch.nn.MSELoss(reduction='sum')
        self.mse_scale = 20
        self.zkl_scale = 1
        self.nlgeval = None
        
        # self.white = nn.Linear(bert_config.hidden_size, self.latent_size)
        # 101 == [CLS]

        self.function = F.log_softmax
        self.decoder_size = config.embed_size

        if self.config.use_feature:
            dec_emb_size = config.dec_cell_size // 4 * 3
        else:
            dec_emb_size = config.dec_cell_size // 2


        self.rec_encoder = DisRec(self.latent_size, self.latent_size5, config)
        self.pri_encoder = DisPri(config.user_token_size, self.latent_size5, config)
        self.sup_decoder = DeN_Auto(self.latent_size6, self.latent_size5, config)
        # self.direct_z = nn.Linear(self.latent_size6, self.latent_size)
        # self.out_maker = Block(self.latent_size6, self.latent_size5, self.latent_size5)
        
        emb_size = config.dec_cell_size // 2
        emb3 = emb_size * 3

        self.tau = config.tau
        self.x_init_connector = nn.Sequential(
            Block(self.latent_size4, self.latent_size3, self.latent_size3),
            # Block(self.latent_size4, self.latent_size3, self.latent_size3),
            Block(self.latent_size3, self.latent_size2, self.latent_size2),
            # Block(self.latent_size3, self.latent_size2, self.latent_size3),
            # Block(self.latent_size2, self.latent_size2, self.latent_size),
            Block(self.latent_size2, self.latent_size, self.latent_size),
            )
        
        self.expand_weight = config.expand_weight
        self.greedy_cat_connector = nn_lib.GreedyConnector()
        self.nll_loss = criterions.NLLEntropy(0, self.config)
        self.rating_loss = nn.MSELoss()
        self.nll_loss_word = criterions.NLLEntropy(0, self.config, avg_type="word")
        self.ppl = criterions.Perplexity(0, self.config)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.log_uniform_y = Variable(torch.log(torch.ones(1) / config.k))
        self.entropy_loss = criterions.Entropy()
        # self.uni_layer_norm = nn.LayerNorm(self.emb_size, elementwise_affine=False)

        self.steps = None
        self.E_steps = None

        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
        self.kl_w = 0.0
        self.mode = None
        self.sqrt = torch.sqrt

    def call(self, inputs, mode='positive',scale = None):
        if scale is None:
            scale = self.scale
        if mode == 'positive':
            scale = self.tau + (1 - self.tau) * torch.sigmoid(self.scale)
        else:
            scale = (1 - self.tau) * torch.sigmoid(-self.scale)
        return inputs * self.sqrt(scale)

    # def hash_rand(self, mu, logvar, d):
    #     return mu + torch.exp(logvar) * d

    def valid_loss(self, loss, batch_cnt=None, step=None):
        step = batch_cnt

        vae_kl_weight = 1.0

        loss.vae_zkl = 0
        
        if loss.vae_nll is None:
            loss.vae_nll = 0

        if self.training:
            vae_loss = loss.vae_nll +  loss.vae_rkl
        else:
            vae_loss = loss.vae_nll
        return vae_loss

    def set_pretrained_mode(self):
        self.mode = 0
        
    def set_normal_train_mode(self):
        self.mode = 1
        
    def pxz_forward(self, batch_size, results, out_utts, mode, gen_type, encoder_output=None,feature_emb=None):
        # dec_init_state = self.x_init_connector(results.sample_z).squeeze(0) #ORI: 1,batch_size,dec_size
        dec_outs, dec_last, dec_ctx = self.x_decoder(results.sample_z.squeeze(0), self.config, out_utts, mode)
        # dec_outs, dec_last, dec_ctx = self.x_decoder(dec_init_state, self.config, out_utts, mode)

        results['dec_outs'] = dec_outs
        results['dec_ctx'] = dec_ctx
        return results
    
    @staticmethod
    def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
        y = torch.mean(torch.sum(-0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2),1),0)
        return y
        # return torch.nan_to_num(y)
    
    def gaussian_analytical_standard(self, z_mean, z_logvar):
        z_mean = torch.nan_to_num(z_mean)
        z_logvar = torch.nan_to_num(z_logvar)
        z_mean = self.uni_layer_norm(z_mean)
        z_logvar = self.uni_layer_norm(z_logvar)

        # if self.config.use_bnvae:
        #     z_mean = self.call(self.bn(z_mean),self.scale2)
        #     z_logvar = self.call(self.bn(z_logvar),'n',self.scale2)
        return torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mean ** 2 - z_logvar.exp(), dim = 1), dim = 0)

    
    @staticmethod
    def analysis_mu(mu1, mu2):
        return torch.mean(torch.sum((mu1 - mu2) ** 2,1),0)

    @staticmethod
    def check_loss(loss):
        if torch.isnan(loss) or torch.isnan(loss):
            return 0
        return loss

    def bounded_mse(self, input, target, logvar):
        diff = torch.abs(target - input)
        return torch.mean(torch.sum(torch.clamp_min(diff - 2*logvar.exp(),0)**2,1))

    @staticmethod
    def radd(input):
        return torch.cat([input[:,0].unsqueeze(-1),input],-1)

    def anneal_score(self):
        return float(1 / np.exp((self.steps/(self.E_steps * 0.7))**2 + np.log(1+self.steps)))
        # return float(1 / np.exp((self.steps/(self.E_steps * 0.7))**2))
    
    def forward(self, data_feed, mode, sample_n=1, gen_type='greedy', return_latent=False):
        
        real_feature_embs = None
        batch_size = data_feed['user_ids'].size(0)

        user_token = self.user_embedding(data_feed['user_ids']).view(batch_size, self.config.user_token_cnt, -1)
        item_token = self.item_embedding(data_feed['item_ids']).view(batch_size, self.config.item_token_cnt, -1)

        # if self.training and not self.restarted:
        #     user_matrix = F.gumbel_softmax(user_matrix, hard=True, tau=self.anneal_score())
        #     item_matrix = F.gumbel_softmax(item_matrix, hard=True, tau=self.anneal_score())
        # else:
        #     user_index = user_matrix.max(1, keepdim=True)[1]
        #     user_matrix = torch.zeros_like(user_matrix, memory_format=torch.legacy_contiguous_format).scatter_(1, user_index, 1.0)
        #     item_index = item_matrix.max(1, keepdim=True)[1]
        #     item_matrix = torch.zeros_like(item_matrix, memory_format=torch.legacy_contiguous_format).scatter_(1, item_index, 1.0)


        # user_token = self.user_tokens.weight.view(user_matrix.size(1), user_matrix.size(2), -1).unsqueeze(0)
        # item_token = self.item_tokens.weight.view(item_matrix.size(1), item_matrix.size(2), -1).unsqueeze(0)
        
        # user_token = torch.sum(user_token * user_matrix.unsqueeze(-1), 1)
        # item_token = torch.sum(item_token * item_matrix.unsqueeze(-1), 1)

        # user_token = self.user_emb_trans(user_matrix)
        # item_token = self.item_emb_trans(item_matrix)
        # user_token = user_matrix
        # item_token = item_matrix

        real_rating_embs = self.rating_embedding(data_feed['ratings'].ge(3).int())

        #ignore feature embedding
        # if self.config.use_feature:
        #     real_feature_embs = self.feature_embedding(data_feed['feature_ids'])
            
        pos_embds = []
        #Ignore Position Embeddings
        # for i in range(3):
        #     # Manually Do reverse here
        #     pos_embds.append(self.pos_embedding[2-i](data_feed['rd_ids'].unsqueeze(-1)))

        # u_tmp = self.base_builder(torch.cat([real_user_embs, real_item_embs, real_rating_embs],-1))
        idx = torch.zeros_like(user_token[:,0,0]).unsqueeze(-1).long()
        ui = torch.cat([self.ui_sp_tokens(idx), user_token, item_token, real_rating_embs.unsqueeze(1)], 1) + self.ui_pos_emb.weight

        words =  self.rec_token_emb(data_feed['sentence_embeddings']) + self.rec_pos_emb.weight
        z_pri,pri_state = self.pri_encoder(ui, torch.ones_like(ui[:,:,0]))
        z_rec,rec_state = self.rec_encoder(words, data_feed['attention_masks'])
        
        vae_nll = None
        vae_ppl = None
        if not self.sample_mode:
            z_sample, rkl_loss = self.sup_decoder(z_pri[:, 0], rec_state, pri_state, pos_embds, self.restarted)

            result = Pack(sample_z = z_sample, z_pri=z_pri)
            vae_resp = self.pxz_forward(data_feed['sentence_embeddings'].size(0), result, data_feed['ans_ids'], mode, gen_type,feature_emb=real_feature_embs)

            # if self.restarted:
            #     vae_nll = None
            #     vae_ppl = None
            # else:
            vae_nll = self.nll_loss(vae_resp['dec_outs'], data_feed['ans_ids'])
            vae_ppl = self.ppl(vae_resp['dec_outs'], data_feed['ans_ids'])
            # vae_nll = 0
            # vae_ppl = 0
        else:
            sample_size = 2
            # pri_mean, pri_logvar = self.out_maker(z_pri[:, 0]).chunk(2, -1)
            # z_sample, rkl_loss, rand_id = self.sup_decoder.sample_forward(pri_mean, rec_state, pri_state, sample_size)
            z_sample, rkl_loss, rand_id = self.sup_decoder.sample_forward(z_pri[:, 0], rec_state, pri_state, sample_size)
            # z_sample, rkl_loss = self.sup_decoder.sample_forward(z_rec, rec_state, pri_state, sample_size)
            result = Pack(sample_z = z_sample, rand_id = rand_id)
            vae_resp = self.pxz_forward(data_feed['sentence_embeddings'].size(0) * sample_size, result, data_feed['ans_ids'].repeat(sample_size, 1), mode, gen_type,feature_emb=real_feature_embs)

        vae_rkl = None

        if self.backLoss:
                # vae_rkl = rkl_loss + mi_kl
            if self.training:
                # vae_rkl = rkl_loss + self.rating_loss(z_pri[:,0].detach(), z_rec[:,0]) * 0.1 + self.rating_loss(z_pri[:,0], z_rec[:,0].detach()) + mi_kl
                vae_rkl = rkl_loss
                if  vae_rkl < self.config.rkl_bound and self.rkl_weight >= 1e-6:
                    self.rkl_weight *= 0.125
                elif self.rkl_weight < self.config.rkl_bound:
                    self.rkl_weight *= 2
                    # vae_rkl = rkl_loss * self.rkl_weight + mi_kl
                return Pack(vae_zkl=None, vae_nll=vae_nll, vae_PPL = vae_ppl, vae_rnll=None, vae_rkl=vae_rkl * self.rkl_weight)
            else:
                return Pack(vae_zkl=None, vae_nll=vae_nll, vae_PPL = vae_ppl, vae_rnll=None, vae_rkl=None)
        else:
            if not self.sample_mode:
                return vae_resp['dec_ctx'], data_feed['ans_ids']
            else:
                return vae_resp['dec_ctx'], data_feed['ans_ids'], result['rand_id']
        return result


    def get_freeze_optimizer(self, config):
        if config.op == 'adamw':
            no_decay = ['bias', 'LayerNorm.weight']
            bert_compent_name = ['x_encoder','xtz','x_embedding','x_decoder','x_init_connector']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_compent_name)], 'weight_decay': config.weight_decay},
                {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_compent_name)], 'weight_decay': 0.0},
                ]
            print([n for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_compent_name)])
            print("Not use this optimizer!")
            exit(0)
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.init_lr, eps=1e-6)
        if config.op == 'adam':
            self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,
                                           self.parameters()), lr=config.init_lr)
        elif config.op == 'sgd':
            self.optimizer =torch.optim.SGD(self.parameters(), lr=config.init_lr,
                                   momentum=config.momentum)
        elif config.op == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=config.init_lr,
                                       momentum=config.momentum)
        self.scheduler = transformers.get_constant_schedule_with_warmup(self.optimizer,config.warmup_steps)
        return self.optimizer


from torch.distributions.normal import Normal

class DLS_Auto(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dis_layers = nn.ModuleList()
        self.dis_layers2 = nn.ModuleList()
        self.f_cs = nn.ModuleList()
        self.gau_list = []
        self.n_batch = -1 
        self.gen_mode = False
        # self.y_z_info = nn.ModuleList()

        self.config = config
        self.cnt = 0

        self.var_list = []
        self.x_var_list = []
        self.dis_list = []
        self.x_y_dist_list = []
        self.center_var = []
        self.align_loss_list = []
        self.uniform_loss_list = []
        self.mini_emb_list = []
        self.rec_emb_list = []
        self.nll_loss = F.nll_loss
        self.batch_arrange = torch.nn.Parameter(torch.arange(config.batch_size), requires_grad=False)
        # self.cl = Block(input_dim, config.feature_cnt, input_dim) # feature classifier

        used_dim = 0
        while(input_dim <= output_dim):
            self.layers.append(Block(input_dim, input_dim * 2, input_dim * 2))
            if self.cnt == 0:
                self.dis_layers.append(Block(input_dim * 2, input_dim * 4, input_dim * 4))
                self.dis_layers2.append(Block(input_dim * 2, input_dim * 4, input_dim * 4))
            else:
                self.dis_layers.append(Block(input_dim * 3, input_dim * 4, input_dim * 4))
                self.dis_layers2.append(Block(input_dim * 3, input_dim * 4, input_dim * 4))
            self.gau_list.append(Normal(torch.zeros(input_dim*2).to(torch.cuda.current_device()),torch.ones(input_dim*2).to(torch.cuda.current_device())))
            used_dim += input_dim * 2

            input_dim *= 2
            self.cnt += 1
        self.gau_list.append(Normal(torch.zeros(used_dim).to(torch.cuda.current_device()), torch.ones(used_dim).to(torch.cuda.current_device())))

    @staticmethod
    def dis_reparameterization(mu, logvar):
        return mu + torch.exp(logvar) * torch.rand_like(logvar)

    def reparameterization_rand(self, mu, logvar, sample=True, z = None):
        rand = torch.rand_like(logvar)
        return mu + torch.exp(logvar) * rand, rand

    @staticmethod
    @torch.jit.script
    def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
        # return torch.mean(torch.nan_to_num(torch.sum(-0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2),1)),0)
        return torch.mean(torch.sum(-0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2),1),0)

    @staticmethod
    def gaussian_analytical_kl_standard(mu1, logsigma1):
        return torch.mean(torch.sum(0.5 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1) ** 2),1),0)

    # @staticmethod
    # @torch.jit.script
    @staticmethod
    @torch.jit.script
    def _density_estimate(mean, logvar, sample):
         var = logvar.exp()
         rs = sample.expand(sample.size(0), sample.size(0), sample.size(1)).transpose(0,1)
         return (rs-mean)/var

    def density_estimate(self, mean, logvar, sample, dis):
         var = logvar.exp()
        #  rs = sample.expand(sample.size(0), sample.size(0), sample.size(1)).transpose(0,1)
         prob = dis.log_prob(self._density_estimate(mean, logvar, sample)).sum(-1).sum(-1) - torch.log(self.n_batch * var.size(0))
         return prob.mean()

    @staticmethod
    @torch.jit.script
    def gaussian_merge(mean1, mean2, logvar1, logvar2):
        var1, var2 = logvar1.exp(), logvar2.exp()
        mean = (var1 * mean2 + var2 * mean1 ) / (var1 + var2)
        var = 2 * (var1*var2)/(var1+var2) # Var Non decrease
        return mean, var.log()

    @staticmethod
    @torch.jit.script
    def align_loss(x, y):
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @staticmethod
    @torch.jit.script
    def align_metric(x, y):
        return (x/x.norm(p=2,dim=1).unsqueeze(-1) - y/y.norm(p=2,dim=1).unsqueeze(-1)).norm(p=2, dim=1)

    @staticmethod
    @torch.jit.script
    def uniform_loss(x):
        return torch.pdist(x, p=2.0).pow(2).mul(-2).exp().mean().log()

    @staticmethod
    @torch.jit.script
    def uniform_metric(x):
        return torch.pdist(x, p=2.0).pow(2).mul(-2).exp().log()

    @staticmethod
    @torch.jit.script
    def cons_loss(v1, v2, batch_arrange):
        v1 = v1 / torch.norm(v1, dim=1).unsqueeze(-1)
        v2 = v2 / torch.norm(v2, dim=1).unsqueeze(-1)
        scores = v1@v2.transpose(0, 1) * 1.1 
        return (F.nll_loss(F.log_softmax(scores, dim=1), batch_arrange) + F.nll_loss(F.log_softmax(scores, dim=0), batch_arrange))/2

    def forward(self, output, rec_output, act1, act2, restarted):
        # act1 for rec, act2 for pri
        rkl = 0
        # output = output[:,0] #Only Use CLS
        last_t_sample = None
        last_sample = None
        qgroup = 0
        amean = None
        alogvar = None
        asample = None

        if not restarted and not self.config.ost:
            # rkl = self.nll_loss(F.log_softmax(self.cl(output)), feature_ids) * self.config.cip_weight # feature classifer loss
            rkl = self.cons_loss(output, rec_output, self.batch_arrange) * self.config.cip_weight # Use constrative loss
            for index, (layer, dis_layer, dis_layer2) in enumerate(zip(self.layers, self.dis_layers, self.dis_layers2)):
                if last_sample is None:
                    mean, logvar = dis_layer(act1[-index-1] + act2[-index-1]).chunk(2, -1)
                    t_mean, t_logvar = dis_layer2(act2[-index-1]).chunk(2, -1)
                    last_sample = self.dis_reparameterization(mean, logvar) + act2[-index-1]
                    amean = t_mean
                    alogvar = t_logvar
                    asample = last_sample
                else:
                    mean, logvar = dis_layer(torch.cat([act1[-index-1] + act2[-index-1], last_sample], -1)).chunk(2, -1)
                    t_mean, t_logvar = dis_layer2(torch.cat([act2[-index-1], last_t_sample], -1)).chunk(2, -1)
                    last_sample = self.dis_reparameterization(mean, logvar) + act2[-index-1]
                    amean = torch.cat([amean, t_mean], -1)
                    alogvar = torch.cat([alogvar, t_logvar], -1)
                    asample = torch.cat([asample, last_sample], -1)
                
                last_t_sample = t_mean # Just use t_mean
                rkl = rkl + self.config.expand_weight * self.gaussian_analytical_kl(t_mean, mean.detach(), t_logvar, logvar.detach())
                rkl = rkl + self.config.converge_weight * self.gaussian_analytical_kl(t_mean.detach(), mean, t_logvar.detach(), logvar) 
                rkl = rkl + self.cons_loss(act1[-index-1], act2[-index-1], self.batch_arrange)  * self.config.cip_weight
                qgroup = qgroup + self.density_estimate(t_mean, t_logvar, last_sample, self.gau_list[index])
                
                output = layer(output) + last_sample
            rkl = rkl + torch.abs((qgroup - self.density_estimate(amean, alogvar, asample, self.gau_list[-1])) * self.config.gir_weight)
        else:
            rkl = self.cons_loss(output, rec_output, self.batch_arrange) * self.config.cip_weight # Use this still in testing
            if self.gen_mode:
                self.align_loss_list.append(self.align_metric(output.detach(), rec_output.detach()).cpu())
                self.uniform_loss_list.append(self.uniform_metric(output.detach()).cpu())
                self.mini_emb_list.append(output.detach().clone().cpu()) # Prevent large GPU memory cost
                self.rec_emb_list.append(rec_output.detach().clone().cpu())

            for index, (layer, dis_layer, dis_layer2) in enumerate(zip(self.layers, self.dis_layers, self.dis_layers2)):
                if last_sample is None:
                    t_mean,t_logvar = dis_layer2(act2[-index-1]).chunk(2, -1)
                    if restarted:
                        mean,logvar = dis_layer(act1[-index-1]+act2[-index-1]).chunk(2, -1)
                        # self.dis_list.append(self.gaussian_analytical_kl(mean.detach(), t_mean.detach(), logvar.detach(), t_logvar.detach()))
                        # self.var_list.append(torch.mean(t_logvar.detach().clone().nan_to_num_()))
                        # self.x_var_list.append(torch.mean(logvar.detach().clone().nan_to_num_()))
                        # self.x_y_dist_list.append(torch.mean(torch.abs(t_mean.detach()-mean.detach())))
                        # m_mean_var = torch.mean(t_mean.detach(), 0).unsqueeze(0) - t_mean.detach()
                        # self.center_var.append(torch.mean(m_mean_var * m_mean_var))
                else:
                    mean,logvar = dis_layer(torch.cat([act1[-index-1]+act2[-index-1], last_sample], -1)).chunk(2, -1)
                    t_mean,t_logvar = dis_layer2(torch.cat([act2[-index-1], last_sample], -1)).chunk(2, -1)
                    # if restarted:
                        # self.dis_list[-1] += self.gaussian_analytical_kl(mean.detach(), t_mean.detach(), logvar.detach(), t_logvar.detach())
                        # self.var_list[-1] += torch.mean(t_logvar.detach().clone().nan_to_num_())
                        # self.x_var_list[-1] += torch.mean(logvar.detach().clone().nan_to_num_())
                        # self.x_y_dist_list[-1] += torch.mean(torch.abs(t_mean.detach()-mean.detach()))
                        # m_mean_var = torch.mean(t_mean.detach(), 0).unsqueeze(0) - t_mean.detach()
                        # self.center_var[-1] += torch.mean(m_mean_var * m_mean_var)

                # rkl = rkl + self.gaussian_analytical_kl(t_mean, mean, t_logvar, logvar) / (index + 1) 
                # rkl = rkl + self.config.converge_weight * self.gaussian_analytical_kl(t_mean.detach(), mean, t_logvar.detach(), logvar) / (index + 1) 

                if self.training:
                    last_sample = self.dis_reparameterization(t_mean, t_logvar) + act2[-index-1]
                    rkl = rkl + self.config.expand_weight * self.gaussian_analytical_kl(t_mean, mean.detach(), t_logvar, logvar.detach()) 
                    # rkl = rkl + self.config.converge_weight * self.gaussian_analytical_kl(t_mean.detach(), mean, t_logvar.detach(), logvar) 
                else:
                    last_sample = t_mean + act2[-index-1]
                output = torch.nan_to_num(layer(output) + last_sample)
        return output, rkl

from transformers import AutoConfig,AutoModel

class DLS_VAE(GMMBase):
    def __init__(self, corpus, config, data_loader=None):
        super(DLS_VAE, self).__init__(config)

        self.restarted = False
        self.latent_size =  config.dec_cell_size
        self.latent_size2 =  self.latent_size // 2 
        self.latent_size3 =  self.latent_size2 // 2
        self.latent_size4 =  self.latent_size3 // 2
        self.latent_size5 =  self.latent_size4 // 2
        self.latent_size6 =  self.latent_size5 // 2

        #Setting up GPT decoder model
        self.x_decoder = GPT2Model.from_pretrained('distilgpt2')
        sefl.x_decoder.config.n_positions = config.decode_max_seq
        self.x_decoder.config.user_cache = True
        # gpt_config = AutoConfig.from_pretrained('distilgpt2')
        # gpt_config.n_positions = config.decode_max_seq
        # self.x_decoder = GPT2Model(gpt_config)

        self.encoder = AutoModel.from_pretrained('distilbert-base-uncased')
        self.encoder.config.n_positions = config.encode_max_seq

        for dl in data_loader:
            # dl.embedding = self.x_embedding
            dl.sampler = RandomSampler(dl.data)
            dl.dataloader = DataLoader(dl.data, config.batch_size, sampler=dl.sampler, collate_fn=dl.batcher(config.max_seq_len, torch.cuda.current_device()), num_workers=1, drop_last=True)

        
        # self.bn = nn.BatchNorm1d(self.latent_size6,affine=False,eps=1e-8,track_running_stats=False)

        self.backLoss = True
        self.GEN_MODE = False
        self.sample_mode = False

        self.dropout = nn.Dropout(0.2)
        self.mse_loss = torch.nn.MSELoss(reduction='sum')
        self.mse_scale = 20
        self.zkl_scale = 1
        self.nlgeval = None
        
        # self.white = nn.Linear(bert_config.hidden_size, self.latent_size)
        # 101 == [CLS]

        self.function = F.log_softmax
        self.decoder_size = config.embed_size

        self.rec_encoder = DlsRec(self.latent_size, self.latent_size5, config)
        self.pri_encoder = DlsPri(self.latent_size, self.latent_size5, config)
        self.sup_decoder = DLS_Auto(self.latent_size6, self.latent_size4, config)
        # self.direct_z = nn.Linear(self.latent_size6, self.latent_size)
        # self.out_maker = Block(self.latent_size6, self.latent_size5, self.latent_size5)
        
        self.tau = config.tau
        self.x_init_connector = nn.Sequential(
            # Block(self.latent_size4, self.latent_size3, self.latent_size3),
            # Block(self.latent_size4, self.latent_size3, self.latent_size3),
            Block(self.latent_size3, self.latent_size2, self.latent_size2),
            # Block(self.latent_size3, self.latent_size2, self.latent_size3),
            # Block(self.latent_size2, self.latent_size2, self.latent_size),
            Block(self.latent_size2, self.latent_size, self.latent_size),
            )
        
        self.expand_weight = config.expand_weight
        self.greedy_cat_connector = nn_lib.GreedyConnector()
        self.nll_loss = criterions.NLLEntropy(0, self.config)
        self.rating_loss = nn.MSELoss()
        self.nll_loss_word = criterions.NLLEntropy(0, self.config, avg_type="word")
        self.ppl = criterions.Perplexity(0, self.config)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.log_uniform_y = Variable(torch.log(torch.ones(1) / config.k))
        self.entropy_loss = criterions.Entropy()
        # self.uni_layer_norm = nn.LayerNorm(self.emb_size, elementwise_affine=False)

        self.steps = None
        self.E_steps = None

        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
        self.kl_w = 0.0
        self.mode = None
        self.sqrt = torch.sqrt

    def call(self, inputs, mode='positive',scale = None):
        if scale is None:
            scale = self.scale
        if mode == 'positive':
            scale = self.tau + (1 - self.tau) * torch.sigmoid(self.scale)
        else:
            scale = (1 - self.tau) * torch.sigmoid(-self.scale)
        return inputs * self.sqrt(scale)

    # def hash_rand(self, mu, logvar, d):
    #     return mu + torch.exp(logvar) * d

    def valid_loss(self, loss, batch_cnt=None, step=None):
        step = batch_cnt

        vae_kl_weight = 1.0

        loss.vae_zkl = 0
        
        if loss.vae_nll is None:
            loss.vae_nll = 0

        if self.training:
            vae_loss = loss.vae_nll +  loss.vae_rkl
        else:
            vae_loss = loss.vae_nll
        return vae_loss

    def set_pretrained_mode(self):
        self.mode = 0
        
    def set_normal_train_mode(self):
        self.mode = 1
        
    def pxz_forward(self, batch_size, results, out_utts, mode, gen_type, encoder_output=None,feature_emb=None):
        dec_init_state = self.x_init_connector(results.sample_z).squeeze(0) #ORI: 1,batch_size,dec_size
        dec_outs, dec_last, dec_ctx = self.x_decoder(dec_init_state, self.config, out_utts, mode)

        results['dec_outs'] = dec_outs
        results['dec_ctx'] = dec_ctx
        return results
    
    @staticmethod
    def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
        y = torch.mean(torch.sum(-0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2),1),0)
        return y
        # return torch.nan_to_num(y)
    
    def gaussian_analytical_standard(self, z_mean, z_logvar):
        z_mean = torch.nan_to_num(z_mean)
        z_logvar = torch.nan_to_num(z_logvar)
        z_mean = self.uni_layer_norm(z_mean)
        z_logvar = self.uni_layer_norm(z_logvar)

        # if self.config.use_bnvae:
        #     z_mean = self.call(self.bn(z_mean),self.scale2)
        #     z_logvar = self.call(self.bn(z_logvar),'n',self.scale2)
        return torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mean ** 2 - z_logvar.exp(), dim = 1), dim = 0)
    
    @staticmethod
    def analysis_mu(mu1, mu2):
        return torch.mean(torch.sum((mu1 - mu2) ** 2,1),0)

    @staticmethod
    def check_loss(loss):
        if torch.isnan(loss) or torch.isnan(loss):
            return 0
        return loss

    def bounded_mse(self, input, target, logvar):
        diff = torch.abs(target - input)
        return torch.mean(torch.sum(torch.clamp_min(diff - 2*logvar.exp(),0)**2,1))

    @staticmethod
    def radd(input):
        return torch.cat([input[:,0].unsqueeze(-1),input],-1)

    def anneal_score(self):
        return float(1 / np.exp((self.steps/(self.E_steps * 0.7))**2 + np.log(1+self.steps)))
        # return float(1 / np.exp((self.steps/(self.E_steps * 0.7))**2))
    
    def forward(self, data_feed, mode, sample_n=1, gen_type='greedy', return_latent=False):
        
        real_feature_embs = None
        batch_size = data_feed['user_ids'].size(0)

        context = self.x_encoder(data_feed['context_ids'], data_feed['context_attention_mask'])
        respond = self.x_encoder(data_feed['encode_respond_ids'], data_feed['encode_respond_attention_mask'])


        z_pri,pri_state = self.pri_encoder(context)
        z_rec,rec_state = self.rec_encoder(respond)
        
        vae_nll = None
        vae_ppl = None
        if not self.sample_mode:
            z_sample, rkl_loss = self.sup_decoder(z_pri, z_rec, rec_state, pri_state, self.restarted)

            result = Pack(sample_z = z_sample, z_pri=z_pri)
            vae_resp = self.pxz_forward(data_feed['decode_respond_ids'].size(0), result, data_feed['decode_respond_ids'], mode, gen_type)

            # if self.restarted:
            #     vae_nll = None
            #     vae_ppl = None
            # else:
            vae_nll = self.nll_loss(vae_resp['dec_outs'], data_feed['ans_ids'])
            vae_ppl = self.ppl(vae_resp['dec_outs'], data_feed['ans_ids'])
            # vae_nll = 0
            # vae_ppl = 0
        else:
            sample_size = 2
            # pri_mean, pri_logvar = self.out_maker(z_pri[:, 0]).chunk(2, -1)
            # z_sample, rkl_loss, rand_id = self.sup_decoder.sample_forward(pri_mean, rec_state, pri_state, sample_size)
            z_sample, rkl_loss, rand_id = self.sup_decoder.sample_forward(z_pri[:, 0], rec_state, pri_state, sample_size)
            # z_sample, rkl_loss = self.sup_decoder.sample_forward(z_rec, rec_state, pri_state, sample_size)
            result = Pack(sample_z = z_sample, rand_id = rand_id)
            vae_resp = self.pxz_forward(data_feed['sentence_embeddings'].size(0) * sample_size, result, data_feed['ans_ids'].repeat(sample_size, 1), mode, gen_type,feature_emb=real_feature_embs)

        vae_rkl = None

        if self.backLoss:
                # vae_rkl = rkl_loss + mi_kl
            if self.training:
                # vae_rkl = rkl_loss + self.rating_loss(z_pri[:,0].detach(), z_rec[:,0]) * 0.1 + self.rating_loss(z_pri[:,0], z_rec[:,0].detach()) + mi_kl
                vae_rkl = rkl_loss
                if  vae_rkl < self.config.rkl_bound and self.rkl_weight >= 1e-6:
                    self.rkl_weight *= 0.125
                elif self.rkl_weight < self.config.rkl_bound:
                    self.rkl_weight *= 2
                    # vae_rkl = rkl_loss * self.rkl_weight + mi_kl
                return Pack(vae_zkl=None, vae_nll=vae_nll, vae_PPL = vae_ppl, vae_rnll=None, vae_rkl=vae_rkl * self.rkl_weight)
            else:
                return Pack(vae_zkl=None, vae_nll=vae_nll, vae_PPL = vae_ppl, vae_rnll=None, vae_rkl=None)
        else:
            if not self.sample_mode:
                return vae_resp['dec_ctx'], data_feed['ans_ids']
            else:
                return vae_resp['dec_ctx'], data_feed['ans_ids'], result['rand_id']
        return result


    def get_freeze_optimizer(self, config):
        if config.op == 'adamw':
            no_decay = ['bias', 'LayerNorm.weight']
            bert_compent_name = ['x_encoder','xtz','x_embedding','x_decoder','x_init_connector']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_compent_name)], 'weight_decay': config.weight_decay},
                {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_compent_name)], 'weight_decay': 0.0},
                ]
            print([n for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in bert_compent_name)])
            print("Not use this optimizer!")
            exit(0)
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.init_lr, eps=1e-6)
        if config.op == 'adam':
            self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,
                                           self.parameters()), lr=config.init_lr)
        elif config.op == 'sgd':
            self.optimizer =torch.optim.SGD(self.parameters(), lr=config.init_lr,
                                   momentum=config.momentum)
        elif config.op == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=config.init_lr,
                                       momentum=config.momentum)
        self.scheduler = transformers.get_constant_schedule_with_warmup(self.optimizer,config.warmup_steps)
        return self.optimizer
