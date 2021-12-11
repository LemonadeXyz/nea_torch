import argparse
import logging
import numpy as np
import torch
from importlib import import_module
from torch.utils.data import DataLoader

from nea.asap_dataloader import DatasetNea, collate_fn
from nea.train_eval import train
from nea.utils import mkdir_p, set_logger, Configuration, print_args

# -----------------------------------------------------------------------------------------------------------#

# import warnings
# warnings.simplefilter("error")

# -----------------------------------------------------------------------------------------------------------#

logger = logging.getLogger(__name__)

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-dp", "--data-pth", dest="data_path", type=str, metavar='<str>', required=True,
                    help="The path to the data-set")
parser.add_argument("-f", "--fold", dest="fold_num", type=str, metavar='<str>', required=True,
                    help="The number of the data folder")
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True,
                    help="The path to the output directory")
parser.add_argument("-p", "--prompt", dest="prompt_id", type=int, metavar='<int>', required=True, default=0,
                    help="Promp ID for ASAP dataset.")
parser.add_argument("-t", "--type", dest="model_type", type=str, metavar='<str>', default='regp',
                    help="Model type (reg|regp|breg|bregp) (default=regp)")
parser.add_argument("-u", "--rec-unit", dest="rnn_type", type=str, metavar='<str>', default='lstm',
                    help="Recurrent unit type (lstm|gru|rnn_tanh|rnn_relu') (default=lstm)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='rmsprop',
                    help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
parser.add_argument("-l", "--loss", dest="loss", type=str, metavar='<str>', default='mse',
                    help="Loss function (mse|mae) (default=mse)")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=50,
                    help="Embeddings dimension (default=50)")
parser.add_argument("-c", "--cnndim", dest="cnn_dim", type=int, metavar='<int>', default=50,
                    help="CNN output dimension (default=50)")
parser.add_argument("-w", "--cnnwin", dest="cnn_window_size", type=int, metavar='<int>', default=3,
                    help="CNN window size. (default=3)")
parser.add_argument("-r", "--rnndim", dest="rnn_dim", type=int, metavar='<int>', default=300,
                    help="RNN dimension (default=300)")
parser.add_argument("-y", "--rnnnlr", dest="rnn_nlr", type=int, metavar='<int>', default=1,
                    help="RNN number of layers. '2' means bidirectional (default=1)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=32,
                    help="Batch size (default=32)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000,
                    help="Vocab size (default=4000)")
parser.add_argument("--aggregation", dest="aggregation", type=str, metavar='<str>', default='mot',
                    help="The aggregation method for regp and bregp types (mot|attsum|attmean) (default=mot)")
parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.5,
                    help="The dropout probability. To disable, give a negative number (default=0.5)")
parser.add_argument("--vocab-path", dest="vocab_path", type=str, metavar='<str>', default='',
                    help="(Optional) The path to the existing vocab file (*.pkl)")
parser.add_argument("--skip-init-bias", dest="skip_init_bias", action='store_true',
                    help="Skip initialization of the last layer bias")
parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>',
                    help="The path to the word embeddings file (Word2Vec format)")
parser.add_argument("--epochs", dest="num_epochs", type=int, metavar='<int>', default=50,
                    help="Number of epochs (default=50)")
parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0,
                    help="Maximum allowed number of words during training. '0' means no limit (default=0)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="Random seed (default=1234)")
args = parser.parse_args()

out_dir = args.out_dir_path
mkdir_p(out_dir + '/preds')
mkdir_p(out_dir + '/saved_dict')

config = Configuration(args)
set_logger(config)
print_args(args)

assert args.model_type in {'reg', 'regp', 'breg', 'bregp'}
assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}
assert args.loss in {'mse', 'mae'}
assert args.rnn_type in {'lstm', 'gru', 'rnn_tanh', 'rnn_relu'}
assert args.aggregation in {'mot', 'attsum', 'attmean'}

if args.seed > 0:
    np.random.seed(args.seed)

assert args.prompt_id > 0, 'Given prompt ID INVALID!'

assert args.cnn_dim > 0, 'This NEA version dose NOT support network without CNN!'
assert args.rnn_dim > 0, 'This NEA version dose NOT support network without RNN!'

assert args.cnn_window_size % 2 != 0, 'EVEN win_size can NOT give an output with the SAME size as input, make it ODD!'

if args.model_type == 'reg':
    config.model_name = 'reg'
    logger.info('Building a REGRESSION model')
elif args.model_type == 'regp':
    config.model_name = 'regp'
    logger.info('Building a REGRESSION model with POOLING')
elif args.model_type == 'breg':
    config.model_name = 'breg'
    logger.info('Building a BIDIRECTIONAL REGRESSION model')
else:
    config.model_name = 'bregp'
    logger.info('Building a BIDIRECTIONAL REGRESSION model with POOLING')

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True

# Parameters
params = {'batch_size': args.batch_size,
          'shuffle': True,
          'collate_fn': lambda x: collate_fn(x, config),
          'drop_last': True,
          'num_workers': 0}
config.num_epochs = args.num_epochs

# Datasets
train_set = DatasetNea(config, config.train_path)
dev_set = DatasetNea(config, config.dev_path)
test_set = DatasetNea(config, config.test_path)

if args.maxlen == 0.:
    config.overal_maxlen = max(train_set.true_maxlen, dev_set.true_maxlen, test_set.true_maxlen)
else:
    config.overal_maxlen = args.maxlen

print('config.overal_maxlen =', config.overal_maxlen)

# Generators
train_generator = DataLoader(train_set, **params)
dev_generator = DataLoader(dev_set, **params)
test_generator = DataLoader(test_set, **params)

mean_score = 0.
# mean_square = 0.
for _, _, score in train_generator:
    # initialize mean_score with mean score of the first batch of training set
    mean_score = np.array(score).mean()
    # mean_square = (np.array(score) ** 2).mean()
    break

# stdev_score = torch.sqrt(mean_score - mean_square ** 2)
config.initial_mean_value = mean_score

hdl = import_module('models.' + args.model_type)
model = hdl.Model(config).to(config.device)
model.init_weight(config)
logger.info(model.parameters)

train(config, model, train_generator, dev_generator, test_generator)
