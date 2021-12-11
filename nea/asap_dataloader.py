import logging
import numpy as np
import codecs
import torch
from torch.utils.data import Dataset

from nea.utils import tokenize, is_number

logger = logging.getLogger(__name__)


def read_dataset(file_path, output_path, prompt_id, maxlen, to_tokenize, to_lower):
    # according to ASAP dataset
    score_idx = 6

    if 'train' in file_path:
        essays_path = output_path + '/preds/train_essays.txt'
    elif 'dev' in file_path:
        essays_path = output_path + '/preds/dev_essays.txt'
    elif 'test' in file_path:
        essays_path = output_path + '/preds/test_essays.txt'
    else:
        raise OSError('No such file or dirctory to store the essays!')

    if maxlen > 0:
        logger.warning('Removing sequences with more than ' + str(maxlen) + ' words')

    essays, scores, prompt_ids = [], [], []
    true_maxlen = -1

    with codecs.open(file_path, mode='r', encoding='gb18030') as input_file:
        input_file.__next__()  # skip the header

        for line in input_file:
            line = line.strip().split('\t')

            content = line[2].strip()
            if 0 < maxlen < len(content):
                continue  # skip too long essays before further operations
            essay_id = int(line[0])
            essay_set = int(line[1])

            if essay_set == prompt_id or prompt_id <= 0:

                if true_maxlen < len(essays):  # get REAL max length
                    true_maxlen = len(essays)

                if to_lower:
                    content = content.lower()

                if to_tokenize:
                    content = tokenize(content)
                else:
                    content = content.split()

                essays.append(content)
                scores.append(float(line[score_idx]))
                prompt_ids.append(essay_set)

    logger.info('Saving essays to: ' + essays_path)
    # np.savetxt(essays_path, np.array(essays, dtype=object), fmt='%s')
    logger.info('Total number of essays: %d' % len(essays))

    return essays, scores, prompt_ids, true_maxlen


def collate_fn(batch, config):
    (texts, labels) = zip(*batch)
    seq_lengths = torch.LongTensor(list(map(len, texts)))
    labels = torch.FloatTensor(labels)

    # make tensor of text sequences, size (batch_size * seq_len)
    seq_tensor = torch.zeros(len(texts), config.overal_maxlen).long()
    for idx, (seq, seq_len) in enumerate(zip(texts, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    return seq_tensor.permute(1, 0), seq_lengths, labels


class DatasetNea(Dataset):
    def __init__(self, config, source_path, to_tokenize=True, to_lower=True):

        self.source_path = source_path
        self.vocab = config.vocab
        self.essays, self.scores, self.prompt_ids, self.true_maxlen = read_dataset(
            source_path, config.out_dir, config.prompt_id, config.default_maxlen, to_tokenize, to_lower)
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

        self.labels = self.get_model_friendly_scores(np.array(self.scores, dtype='float32'),
                                                     np.array(self.prompt_ids, dtype='int32'))

    def __getitem__(self, idx):
        # return the seq and score 
        seq = self.encode(self.essays[idx])
        label = self.labels[idx]
        return seq, label

    def __len__(self):
        return len(self.essays)

    def encode(self, text):
        seq = list()  # for storing essay encoding
        for word in text:
            # encoding process
            if is_number(word):
                seq.append(self.vocab['<num>'])
            elif word in self.vocab:
                seq.append(self.vocab[word])
            else:
                seq.append(self.vocab['<unk>'])
        return seq

    def get_model_friendly_scores(self, scores_array, prompt_id_array):
        """
        map scores from asap_range[prompt_id_array] to [0, 1]
        """
        arg_type = type(prompt_id_array)
        assert arg_type in {int, np.ndarray}
        if arg_type is int:
            low, high = self.asap_ranges[prompt_id_array]
            scores_array = (scores_array - low) / (high - low)
        else:
            assert scores_array.shape[0] == prompt_id_array.shape[0]
            dim = scores_array.shape[0]
            low = np.zeros(dim)
            high = np.zeros(dim)
            for ii in range(dim):
                low[ii], high[ii] = self.asap_ranges[prompt_id_array[ii]]
            scores_array = (scores_array - low) / (high - low)
        assert np.all(scores_array >= 0) and np.all(scores_array <= 1)

        return scores_array
