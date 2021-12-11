import sys
import os
import errno
import logging
import re
import time
from math import floor
import codecs
import operator
import nltk
import pickle as pkl
import numpy as np
from datetime import timedelta
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr, kendalltau

from nea.quadratic_weighted_kappa import quadratic_weighted_kappa as qwk
from nea.quadratic_weighted_kappa import linear_weighted_kappa as lwk

logger = logging.getLogger(__name__)


def set_logger(config):
    console_format = BColors.OKBLUE + '[%(levelname)s]' + BColors.ENDC + ' (%(name)s) %(message)s'
    # datefmt='%Y-%m-%d %Hh-%Mm-%Ss'
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(console_format))
    logger.addHandler(console)
    if config.out_dir:
        file_format = '[%(levelname)s] (%(name)s) %(message)s'
        log_file = logging.FileHandler(config.out_dir + f'/log{config.prompt_id}_{config.fold_num}.txt', mode='w')
        log_file.setLevel(logging.DEBUG)
        log_file.setFormatter(logging.Formatter(file_format))
        logger.addHandler(log_file)


def mkdir_p(path):
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    WHITE = '\033[37m'
    YELLOW = '\033[33m'
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    RED = '\033[31m'
    MAGENTA = '\033[35m'
    BLACK = '\033[30m'
    BHEADER = BOLD + '\033[95m'
    BOKBLUE = BOLD + '\033[94m'
    BOKGREEN = BOLD + '\033[92m'
    BWARNING = BOLD + '\033[93m'
    BFAIL = BOLD + '\033[91m'
    BUNDERLINE = BOLD + '\033[4m'
    BWHITE = BOLD + '\033[37m'
    BYELLOW = BOLD + '\033[33m'
    BGREEN = BOLD + '\033[32m'
    BBLUE = BOLD + '\033[34m'
    BCYAN = BOLD + '\033[36m'
    BRED = BOLD + '\033[31m'
    BMAGENTA = BOLD + '\033[35m'
    BBLACK = BOLD + '\033[30m'

    @staticmethod
    def cleared(s):
        return re.sub("\033\[[0-9][0-9]?m", "", s)


def red(message):
    return BColors.RED + str(message) + BColors.ENDC


def b_red(message):
    return BColors.BRED + str(message) + BColors.ENDC


def blue(message):
    return BColors.BLUE + str(message) + BColors.ENDC


def b_yellow(message):
    return BColors.BYELLOW + str(message) + BColors.ENDC


def green(message):
    return BColors.GREEN + str(message) + BColors.ENDC


def b_green(message):
    return BColors.BGREEN + str(message) + BColors.ENDC


def tokenize(string):
    tokens = nltk.word_tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index + 1) < len(tokens):
            tokens[index + 1] = '@' + re.sub('[0-9]+.*', '', tokens[index + 1])
            tokens.pop(index)
    return tokens


def is_number(token):
    num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
    return bool(num_regex.match(token))


def print_args(args, path=None):
    if path:
        output_file = open(path, 'w')
    logger = logging.getLogger(__name__)
    logger.info("Arguments:")
    args.command = ' '.join(sys.argv)
    items = vars(args)
    for key in sorted(items.keys(), key=lambda s: s.lower()):
        value = items[key]
        if not value:
            value = "None"
        logger.info("  " + key + ": " + str(items[key]))
        if path is not None:
            output_file.write("  " + key + ": " + str(items[key]) + "\n")
    if path:
        output_file.close()
    del args.command


def get_time_dif(start_time):
    return timedelta(seconds=int(round(time.time() - start_time)))


def calc_correl(obs, pred):
    prs, _ = pearsonr(obs, pred)
    spr, _ = spearmanr(obs, pred)
    tau, _ = kendalltau(obs, pred)
    return prs, spr, tau


def calc_kappa(obs, pred, low, high):
    # Kappa only supports integer values

    obs_int = np.rint(obs).astype('int32')
    pred_int = np.rint(pred).astype('int32')
    qw_kappa = qwk(obs_int, pred_int, low, high)
    lw_kappa = lwk(obs_int, pred_int, low, high)
    return qw_kappa, lw_kappa


def get_cnn_padding_size(input_size, output_size, kernel_size, stride):
    """
    :param input_size: 2-elem tuple [H_in, W_in]
    :param output_size: 2-elem tuple [H_out, W_out]
    :param kernel_size: 2-elem tuple [Ks_h, Ks_w]
    :param stride: 2-elem tuple [S_h, S_w]
    :return: padding_size: 2-elem tuple [pad_h, pad_w]
    """
    pad_h = ((output_size[0] - 1) * stride[0] - input_size[0] + kernel_size[0]) / 2
    pad_w = ((output_size[1] - 1) * stride[1] - input_size[1] + kernel_size[1]) / 2

    pad_h = pad_h if pad_h > 0 else 0
    pad_w = pad_w if pad_w > 0 else 0

    return floor(pad_h), floor(pad_w)


class Configuration(object):
    def __init__(self, args):
        self.fold_num = args.fold_num
        self.data_path = args.data_path
        self.train_path = args.data_path + f'/fold_{args.fold_num}/train.tsv'
        self.dev_path = args.data_path + f'/fold_{args.fold_num}/dev.tsv'
        self.test_path = args.data_path + f'/fold_{args.fold_num}/test.tsv'
        self.model_name = args.model_type
        self.out_dir = args.out_dir_path
        self.save_path = args.out_dir_path + '/saved_dict/' + self.model_name + '.pth'
        self.vis_path = args.out_dir_path + '/log_tensorboard/' + self.model_name

        self.asap_ranges = {
            0: (0, 60),
            1: (2, 12),
            2: (1, 6),
            3: (0, 3),
            4: (0, 3),
            5: (0, 4),
            6: (0, 4),
            7: (0, 30),
            8: (0, 60)
        }
        self.prompt_id = args.prompt_id
        self.low, self.high = self.get_score_range(self.prompt_id)

        self.batch_size = args.batch_size
        self.vocab_size = args.vocab_size
        self.default_maxlen = args.maxlen
        self.overal_maxlen = args.maxlen

        if args.vocab_path:
            self.vocab_path = args.vocab_path
            self.vocab = self.load_vocab(args.vocab_path)
            if len(self.vocab) != args.vocab_size:
                logger.warning('The vocabulary includes %i words which is different from given: %i'
                               % (len(self.vocab), self.vocab_size))
        else:
            self.vocab = self.build_vocab(self.train_path, args.prompt_id,
                                          args.maxlen, args.vocab_size)
            if len(self.vocab) < self.vocab_size:
                logger.warning('The vocabulary includes only %i words (less than %i)'
                               % (len(self.vocab), self.vocab_size))
            else:
                assert self.vocab_size == 0 or len(self.vocab) == self.vocab_size

        # model setting
        self.window_size = args.cnn_window_size
        # ====================================================== #
        self.num_filters = args.cnn_dim
        # self.num_filters = 64  # 测试是否能跑通用的参数
        # ====================================================== #
        self.cnn_stride = 1
        self.cnn_pad_size = get_cnn_padding_size((0, 0), (0, 0), (args.cnn_window_size, 0),
                                                 (self.cnn_stride, self.cnn_stride))
        self.rnn_type = args.rnn_type.upper()  # upper to enable initialize rnn with getattr()
        self.rnn_nlayers = args.rnn_nlr
        # ====================================================== #
        self.hidden_size = args.rnn_dim
        # self.hidden_size = 64  # 测试是否能跑通用的参数
        # ====================================================== #
        self.dropout_W = 0.5  # dropout rate (before feeding rnn)
        self.dropout_U = 0.1 if args.rnn_nlr > 1 else 0.  # dropout rate (between rnn layers)
        self.skip_init_bias = args.skip_init_bias
        self.learning_rate = 1e-3
        self.initial_mean_value = 0.

        if args.emb_path:
            logger.info('Initializing lookup table...')
            self.embedding_pre = torch.tensor(self.get_pretrained_emb(args.emb_path, args.emb_dim), dtype=torch.float32)
            logger.info('  Done')
            self.emb_dim = self.embedding_pre.size(1)
        else:
            self.emb_dim = args.emb_dim

        # optimizer setting
        self.algorithm = args.algorithm
        self.clipvalue = 0
        self.clipnorm = 10

        self.num_epochs = 20
        self.require_improvement = 1000

        self.metric = args.loss

        if args.loss == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.L1Loss()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.USE_CUDA = True if self.device != 'cpu' else False

    def get_score_range(self, prompt_id):
        return self.asap_ranges[prompt_id]

    def get_dataset_friendly_scores(self, scores_array, prompt_id_array):
        """
        :param scores_array: scores of essays, whose scale [0, 1]
        :param prompt_id_array: prompt(s) of essays whose predicted scores lie in scores_array
        :return: an array of scores(keeping sequence), whose scale as those prompts do in data-set
        """
        arg_type = type(prompt_id_array)
        assert arg_type in {int, np.ndarray}
        if arg_type is int:
            low, high = self.asap_ranges[prompt_id_array]
            scores_array = scores_array * (high - low) + low
            assert np.all(scores_array >= low) and np.all(scores_array <= high)
        else:
            assert scores_array.shape[0] == prompt_id_array.shape[0]
            dim = scores_array.shape[0]
            low = np.zeros(dim)
            high = np.zeros(dim)
            for ii in range(dim):
                low[ii], high[ii] = self.asap_ranges[prompt_id_array[ii]]
            scores_array = scores_array * (high - low) + low
        return scores_array

    def load_vocab(self, vocab_path):
        logger.info('Loading vocabulary from: ' + vocab_path)

        with open(vocab_path, 'rb') as vocab_file:
            vocab = pkl.load(vocab_file)

        return vocab

    def build_vocab(self, file_path, prompt_id, maxlen, vocab_size, to_tokenize=True, to_lower=True):

        logger.info('Building vocabulary from: ' + file_path)
        if maxlen > 0:
            logger.warning('Skipping sequences with more than ' + str(maxlen) + ' words')

        total_words, unique_words = 0., 0.
        word_freqs = dict()
        with codecs.open(file_path, mode='r', encoding='gb18030') as input_file:
            input_file.__next__()  # skip the header
            for line in input_file:
                line = line.strip().split('\t')
                essay_set = int(line[1])
                content = line[2].strip()

                if 0 < maxlen < len(content):
                    continue

                if essay_set == prompt_id or prompt_id <= 0:

                    if to_lower:
                        content = content.lower()
                    if to_tokenize:
                        content = tokenize(content)
                    else:
                        content = content.split()

                    for word in content:
                        try:
                            word_freqs[word] += 1
                        except KeyError:
                            unique_words += 1
                            word_freqs[word] = 1

                        total_words += 1

        logger.info('  %i total words, %i unique words' % (total_words, unique_words))
        sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)
        if vocab_size <= 0 or vocab_size is None:
            # calculate vocab size automatically by removing all singletons
            vocab_size = 0
            for word, freq in sorted_word_freqs:
                if freq > 1:
                    vocab_size += 1
        vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
        init_size = len(vocab)
        index = init_size
        for word, freq in sorted_word_freqs[:vocab_size - init_size]:
            if freq > 1:
                vocab[word] = index
                index += 1

        self.vocab_path = self.out_dir
        pkl.dump(vocab, open(self.vocab_path + '/vocab.pkl', 'wb'))
        logger.info('Saving vocabulary to: ' + self.vocab_path)

        self.vocab_size = len(vocab)
        logger.info('  Vocab size: %i' % (len(vocab)))

        return vocab

    def get_pretrained_emb(self, emb_path, emb_dim=None):
        logger.info('Loading embeddings from: ' + emb_path)
        with codecs.open(emb_path, 'r', encoding='gb18030') as emb_file:
            line = emb_file.__next__().split()
            assert len(line) == 2, 'The first line in W2V embeddings must be the pair (vocab_size, emb_dim)'

            try:
                int(line[0])
                int(line[1])
            except ValueError:
                raise ValueError('Incorrect header!')

            self.emb_dim = int(line[1])
            if emb_dim and self.emb_dim != emb_dim:
                logger.warning('The embeddings dimension does not match with the requested dimension')

            embeddings = np.random.rand(self.vocab_size, self.emb_dim)

            for line in emb_file:
                line = line.split()
                word = line[0]
                vec = line[1].split(',')  # NOTE here vec is list of str type elem
                assert len(vec) == self.emb_dim, 'The number of dimensions does not match the header info'

                if word in self.vocab:
                    embeddings[self.vocab[word]] = np.asarray(vec, dtype='float32')

        logger.info('  #vectors: %i, #dimensions: %i' % (self.vocab_size, self.emb_dim))

        return embeddings


# ----------------------------------------------------------------------------------------------------------- #
