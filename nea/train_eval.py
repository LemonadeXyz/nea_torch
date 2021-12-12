import time
import logging
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.tensorboard import SummaryWriter

from nea.optimizers import *
from nea.utils import get_time_dif, calc_correl, calc_kappa
from nea.quadratic_weighted_kappa import quadratic_weighted_kappa as qwk

logger = logging.getLogger(__name__)


def train(config, model, train_generator, dev_generator, test_generator):
    start_time = time.time()
    model.train()

    optimizer = get_optimizer(config.algorithm, model)

    # Decays the learning rate of each parameter group by gamma every epoch.
    # lr = gamma * lr
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    total_batch = 0

    best_dev_loss = float('inf')
    best_dev_metric = -1
    best_dev_qwk = -1
    best_dev_correl = (-1., -1., -1.)
    best_dev_kappa = (-1., -1.)
    best_dev_epoch = -1

    test_echo_metric = -1
    test_echo_correl = (-1., -1., -1.)
    test_echo_kappa = (-1., -1.)

    best_test_metric = -1
    best_test_qwk = -1
    best_test_correl = (-1., -1., -1.)
    best_test_kappa = (-1., -1.)
    best_test_epoch = -1

    last_improve = 0  # save the NUM of BATCH when loss decrease on dev set
    flag = False  # record if there is no improvement for long

    writer = SummaryWriter(log_dir=config.vis_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

    for epoch in range(config.num_epochs):
        logger.info('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        hidden = model.init_hidden(config)

        for i, (texts, seq_lengths, labels) in enumerate(train_generator):
            model.train()
            hidden = model.repackage_hidden(hidden)
            model.zero_grad()
            outputs = model(texts, hidden, seq_lengths)
            loss = config.loss_fn(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.clipnorm)
            optimizer.step()

            if i % 5 == 0:
                predict_rescale = config.get_dataset_friendly_scores(outputs.data.cpu().numpy(),
                                                                     config.prompt_id)
                labels_rescale = config.get_dataset_friendly_scores(labels.data.cpu().numpy(),
                                                                    config.prompt_id)
                train_qwk = qwk(predict_rescale, labels_rescale)
                dev_loss, dev_metric, dev_correl, dev_kappa = evaluate(config, model, dev_generator)
                test_loss, test_metric, test_correl, test_kappa = evaluate(config, model, test_generator)

                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    last_improve = total_batch
                    improve = '*'
                    torch.save(model.state_dict(), config.save_path)
                else:
                    improve = ''

                time_dif = get_time_dif(start_time)
                msg_train = 'Iter: {0:>5},  Train Loss: {1:>5.3},  Train QWK: {2:>6.3},  ' \
                            'Dev Loss: {3:>5.3},  Dev QWK: {4:>6.3},  Time: {5} {6}'
                logger.info(msg_train.format(i * config.batch_size, loss.item(), train_qwk,
                                             dev_loss, dev_kappa[0], time_dif, improve))

                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("qwk/train", train_qwk, total_batch)
                writer.add_scalar("qwk/dev", dev_kappa[0], total_batch)

                if dev_kappa[0] > best_dev_qwk:
                    best_dev_epoch = epoch + 1
                    best_dev_metric = dev_metric
                    best_dev_qwk = dev_kappa[0]
                    best_dev_correl = dev_correl
                    best_dev_kappa = dev_kappa

                    test_echo_metric = test_metric
                    test_echo_correl = test_correl
                    test_echo_kappa = test_kappa

                if test_kappa[0] > best_test_qwk:
                    best_test_qwk = test_kappa[0]
                    best_test_metric = test_metric
                    best_test_correl = test_correl
                    best_test_kappa = test_kappa
                    best_test_epoch = epoch + 1

            total_batch += 1

            if total_batch - last_improve > config.require_improvement:
                # End training if valid loss stops decreasing after 1000 batches
                logger.info("No improvement for a long time, auto-stopping...\n")
                flag = True
                break
        if flag:
            break
        # scheduler.step()
    writer.close()

    msg_best_dev = '[Best DEV @ep{0:>2}]  ' \
                   'metric: {1:>5.4}, QWK: {2:>6.4}, LWK: {3:>5.3}, PRS: {4:>4.3}, SPR: {5:>4.3}, Tau: {6:>4.3}'

    msg_echo_test = '[TEST(echo)]  ' \
                    'metric: {0:>5.4}, QWK: {1:>6.4}, LWK: {2:>5.3}, PRS: {3:>4.3}, SPR: {4:>4.3}, Tau: {5:>4.3}'

    msg_best_test = '[Best TEST @ep{0:>2}]  ' \
                    'metric: {1:>5.4}, QWK: {2:>6.4}, LWK: {3:>5.3}, PRS: {4:>4.3}, SPR: {5:>4.3}, Tau: {6:>4.3}'

    logger.info(msg_best_dev.format(best_dev_epoch, best_dev_metric, best_dev_kappa[0], best_dev_kappa[1],
                                    best_dev_correl[0], best_dev_correl[1], best_dev_correl[2]))

    logger.info(msg_echo_test.format(test_echo_metric, test_echo_kappa[0], test_echo_kappa[1],
                                     test_echo_correl[0], test_echo_correl[1], test_echo_correl[2]))

    logger.info(msg_best_test.format(best_test_epoch, best_test_metric, best_test_kappa[0], best_test_kappa[1],
                                     best_test_correl[0], best_test_correl[1], best_test_correl[2]))

    test(config, model, test_generator)


def test(config, model, test_generator):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_loss, test_metric, test_correl, test_kappa = evaluate(config, model, test_generator)

    msg_info = '[TEST (final)]  ' \
               'loss: {0:>5.4}, metric: {1:>5.4}, QWK: {2:>6.4}, LWK: {3:>5.3}, ' \
               'PRS: {4:>4.3}, SPR: {5:>4.3}, Tau: {6:>4.3}'
    logger.info(msg_info.format(test_loss, test_metric, test_kappa[0], test_kappa[1],
                                test_correl[0], test_correl[1], test_correl[2]))
    time_dif = get_time_dif(start_time)
    logger.info("Time for TEST process: {0}".format(time_dif))


def evaluate(config, model, data_set):
    model.eval()
    loss_total = 0
    hidden = model.init_hidden(config)
    predict_all = np.array([], dtype=float)
    labels_all = np.array([], dtype=float)
    with torch.no_grad():
        for texts, seq_lengths, labels in data_set:
            outputs = model(texts, hidden, seq_lengths)
            loss = config.loss_fn(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = outputs.data.cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    labels_all = config.get_dataset_friendly_scores(labels_all, config.prompt_id)
    predict_all = config.get_dataset_friendly_scores(predict_all, config.prompt_id)

    if config.metric == 'mse':
        metric = mean_squared_error(labels_all, predict_all)
    else:
        metric = mean_absolute_error(labels_all, predict_all)

    prs, spr, tau = calc_correl(labels_all, predict_all)
    qwk_out, lwk_out = calc_kappa(labels_all, predict_all, config.low, config.high)

    return loss_total / len(data_set), metric, (prs, spr, tau), (qwk_out, lwk_out)
