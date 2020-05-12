"""Helper functions for train/validation loop."""
import logging
import io
import torch
import tensorflow as tf
import matplotlib
import os
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.misc
from time import time

from tqdm import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
from texttable import Texttable
from copy import deepcopy


from utils.encoder_decoder import get_encoder_decoder
from utils.optimizers import get_optimizer
from utils.metrics import metrics
from utils.dataloader import get_dataloaders
from utils.loss_utils import loss_meters, MultiTaskLoss, averageMeter
from utils.im_utils import cat_labels, prob_labels, inst_labels, labels
from utils.writer_utils import add_images_to_writer
from utils.data_utils import get_weights, get_class_weights_from_data, \
    TargetGenerator, get_predictions
from utils.constants import TRAIN, VAL, EXPERIMENT_NAME, BEST_LOSS, MILLION, \
    START_ITER, PLATEAU_COUNT, STATE, GPU_STR, GPU_ID, CPU, PARAMS, MODEL, \
    ENCODER, DECODER, DATA, IM_SIZE, PRETRAINED_PATH, ROOT_PATH, RESULTS_DIR, \
    MULTIGPU, TASKS, MODEL_NAME, LOGS, SAVE_LOGS, PKL, EPOCHS, EARLY_STOP, \
    PATIENCE, PRINT_INTERVAL, RESUME, MODEL_STATE, OPTIMIZER_STATE, LOSS_FN, \
    CRITERION_STATE, EPOCH, EPOCH_STR, OUTPUTS, OPTIMIZER, LOSS_STR, \
    TASK_LOSS, TRAIN_STR, VAL_STR, VAL_LOSS_STR, METRICS_STR, PATIENCE_STR, \
    EARLY_STOP_STR, TASK_STR, SEMANTIC, INSTANCE_PROBS, INSTANCE_CONTOUR, \
    POSTPROC, POSTPROCS, PANOPTIC, BATCH_LOG, LENGTH


class ExperimentLoop():
    """Loops experiment in train/val modes."""

    def __init__(self, cfg, mode=TRAIN):
        """Initialize ExperimentLoop class.

        Arguments:
            cfg {[dict]} -- [configuration settings for the experiment]

        Keyword Arguments:
            mode {[str]} -- [train/val mode to run experiment]
            (default: {TRAIN})
        """
        self.cfg = cfg
        self.mode = mode
        self.setup_experiment()

    def setup_experiment(self):
        """Set up required parameters, variables etc for experiment."""
        # Define Configuration parameters
        self.get_config_params()

        # Get device and writer
        self.writer = get_writer(self.cfg)

        # Get dataloaders, model, weights, optimizers and metrics
        self.dataloaders = get_dataloaders(self.cfg)
        self.model = get_model(self.cfg)  # TODO:
        self.weights = get_weights(self.cfg)
        self.criterion = self.get_criterion()
        self.optimizer = self.init_optimizer()
        self.get_losses_and_metrics()

        # Load Checkpoint
        self.load_checkpoint()
        self.model = place_on_multi_gpu(self.cfg, self.model)

    def get_config_params(self):
        """Get configuration parameters for the experiment."""
        cfg = self.cfg
        params = cfg[PARAMS]
        self.params = params
        self.epochs = params[EPOCHS]
        self.patience = params[PATIENCE]
        self.early_stop = params[EARLY_STOP]
        self.print_interval = params[PRINT_INTERVAL]
        self.resume_training = params[RESUME]
        self.exp_dir, _, _ = get_exp_dir(cfg)
        self.plateau_count = 0
        self.start_iter = 0
        self.best_loss = MILLION

    def get_criterion(self):
        """Get criterion for the current experiment."""
        cfg = self.cfg
        loss_fn = cfg[MODEL][LOSS_FN]
        criterion = MultiTaskLoss(cfg[MODEL][OUTPUTS],
                                  self.weights, loss_fn)
        criterion = place_on_multi_gpu(self.cfg, criterion)
        return criterion

    def init_optimizer(self):
        """Initialize the optimizer."""
        optimizer_cls = get_optimizer(self.params[TRAIN])
        optimizer_params = {k: v for k, v in
                            self.params[TRAIN][OPTIMIZER].items()
                            if k != "name"}
        criterion_params = list(self.criterion.parameters())
        if criterion_params is not None:
            model_params = list(self.model.parameters()) + criterion_params
        else:
            model_params = self.model.parameters()
        optimizer = optimizer_cls(model_params, **optimizer_params)

        return optimizer

    def get_losses_and_metrics(self):
        """Get loss and metric meters to log them during experiments."""
        out_model_cfg = self.cfg[MODEL][OUTPUTS]
        post_proc_cfg = self.cfg[POSTPROCS]

        self.train_loss_meters = loss_meters(out_model_cfg)
        self.val_loss_meters = loss_meters(out_model_cfg)
        self.val_metrics = metrics(out_model_cfg)
        self.post_proc_metrics = metrics(post_proc_cfg)

    def load_checkpoint(self):
        """Load a checkpoint from experiments dir."""
        self.check_point_exists = if_checkpoint_exists(self.exp_dir)
        if self.check_point_exists and self.resume_training:
            checkpoint_name, checkpoint = get_checkpoint(self.exp_dir)
            self.model.load_state_dict(checkpoint[MODEL_STATE])
            self.optimizer.load_state_dict(checkpoint[OPTIMIZER_STATE])
            self.criterion.load_state_dict(checkpoint[CRITERION_STATE])
            self.start_iter = checkpoint[EPOCH]
            self.best_loss = checkpoint[BEST_LOSS]
            logging.info("Loaded checkpoint {} from epoch {} with loss {}"
                         .format(checkpoint_name, self.start_iter,
                                 self.best_loss))
        else:
            if self.mode == TRAIN:
                logging.info("Begining Training from Scratch")
            else:
                raise ValueError('Cannot load checkpoint')

    def train(self):
        """Train a model on a entire train data for several epochs."""
        for epoch in range(self.epochs):
            logging.info(EPOCH_STR.format(epoch=epoch+1).center(LENGTH, '='))
            logging.info(TRAIN_STR.center(LENGTH, '='))
            self.loss_weights = self.criterion.module.sigma \
                if self.cfg[PARAMS][MULTIGPU] else self.criterion.sigma
            #self.loss_weights =  self.criterion.sigma
            logging.info('MTL Loss type: {}, Loss weights: {}'.
                         format(self.cfg[MODEL][LOSS_FN], self.loss_weights)
                         .center(LENGTH, '='))
            self.n_batches = len(self.dataloaders[TRAIN])
            # _ = get_class_weights_from_data(
            #    self.dataloaders[TRAIN], 9, self.cfg, task=INSTANCE_CONTOUR)
            self.train_running_loss = averageMeter()
            self.train_generator = TargetGenerator(self.cfg)
            self.val_generator = TargetGenerator(self.cfg)
            start = time()
            for batch_id, data in enumerate(self.dataloaders[TRAIN]):
                self.train_step(data, epoch, batch_id)
                # break
            train_s = time() - start
            loss_str = LOSS_STR.format(epoch=epoch+1,
                                       loss=self.train_running_loss.avg)
            for task, task_loss in self.train_loss_meters.meters.items():
                loss_str += TASK_LOSS.format(task, task_loss.avg)
            logging.info(loss_str + 'Time: {:05.3f}'.format(train_s))

            self.validation_step(epoch)

            # if epoch % self.print_interval == 0 and epoch > 0:
            #    self.val_generator = TargetGenerator(self.cfg)
            #    self.mode == VAL
            #    self.validation_step(epoch)
            #    self.mode == TRAIN

            self.train_running_loss.reset()
            self.train_loss_meters.reset()

            self.save_model(epoch)
            if self.stop_training(epoch):
                self.writer.close()

        self.writer.close()
        logging.info('Stopping Training. Maximum Number of epochs: {} reached'
                     .format(self.epochs))

    def validation(self):
        """Validate a model on a entire validation data."""
        self.validation_step(epoch=1)

    def train_step(self, data, epoch, batch):
        """Train a model on a batch of data.

        Arguments:
            data {tuple} -- [inputs and groundtruth for the network]
            epoch {[int]} -- [Current epoch number]
            batch {[int]} -- [current batch number]

        """

        self.optimizer.zero_grad()
        start = time()
        inputs, targets = data
        inputs = inputs.cuda()
        #labels = get_labels(targets, self.cfg)
        labels = self.train_generator.generate_targets(targets[0], targets[1])
        loader_s = time() - start

        start = time()
        self.model = self.model.cuda()
        logits = self.model(inputs)
        predictions = get_predictions(logits, self.cfg[MODEL][OUTPUTS],
                                      labels)
        forward_s = time() - start

        start = time()
        losses, loss = self.criterion(logits, labels)
        self.train_running_loss.update(loss)
        self.train_loss_meters.update(losses)
        loss_s = time() - start

        start = time()
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        # torch.cuda.empty_cache()
        backward_s = time() - start

        start = time()
        self.batch_size = inputs.shape[0]
        self.sample_idx = np.random.randint(0, self.batch_size, size=1)
        self.writer.add_scalar('Loss/train', loss.mean().data,
                               epoch*self.n_batches + batch)
        for task, task_loss in losses.items():
            self.writer.add_scalar('Loss/train_{}'.format(task),
                                   task_loss.mean().data,
                                   epoch*self.n_batches + batch)
        if batch == 0:
            self.send_predictions_to_writer(inputs, predictions, labels,
                                            self.train_loss_meters, epoch,
                                            TRAIN)
        if batch % self.print_interval == 0 or batch == self.n_batches:
            writer_s = time() - start
            postproc_s = 0.0
            metric_s = 0.0
            train_s = loader_s + forward_s + backward_s + \
                loss_s + postproc_s + writer_s + metric_s
            logging.info(BATCH_LOG.format(epoch+1, batch, self.n_batches,
                                          train_s, loader_s, forward_s,
                                          backward_s, loss_s,
                                          postproc_s, metric_s, writer_s))

    def validation_step(self, epoch):
        """Validate a model on a batch of data.

        Arguments:
            epoch {[int]} -- [Current epoch number]
        """
        val_start = deepcopy(time())

        logging.info(VAL_STR.center(LENGTH, '='))
        running_val_loss = averageMeter()
        with torch.no_grad():
            for i, data in enumerate(self.dataloaders[VAL]):
                get_postprocs = True if self.mode == VAL else False
                start = time()
                inputs, targets = data
                inputs = inputs.cuda()
                self.batch_size = inputs.shape[0]
                self.sample_idx = np.random.randint(0, self.batch_size, size=1)

                labels = self.val_generator.generate_targets(
                    targets[0], targets[1])
                loader_s = time() - start

                start = time()
                logits = self.model(inputs)
                predictions = get_predictions(logits, self.cfg[MODEL][OUTPUTS],
                                              labels)
                forward_s = time() - start
                backward_s = 0.0
                postproc_s = 0.0
                writer_s = 0.0

                start = time()
                val_losses, val_loss = self.criterion(logits, labels)
                val_loss = val_loss.mean().data
                # torch.cuda.empty_cache()
                self.val_loss_meters.update(val_losses)
                running_val_loss.update(val_loss)
                loss_s = time() - start

                start = time()
                self.val_metrics.update(labels, predictions)
                metric_s = time() - start

                if self.mode == VAL:
                    start = time()
                    outputs = post_process_predictions(predictions,
                                                       self.cfg[POSTPROCS])
                    postproc_s = time() - start

                    start = time()
                    self.post_proc_metrics.update(labels, outputs)
                    metric_s += time() - start

                    start = time()
                    self.send_predictions_to_writer(inputs, predictions,
                                                    labels,
                                                    self.val_loss_meters, i,
                                                    VAL, save_to_disk=True)
                    self.send_outputs_to_writer(outputs, labels,
                                                self.post_proc_metrics, i, VAL,
                                                save_to_disk=True)
                    writer_s = time() - start
                # break
                val_s = loader_s + forward_s + backward_s + \
                    loss_s + postproc_s + writer_s + metric_s
                logging.info(BATCH_LOG.format(0, i,
                                              len(self.dataloaders[VAL]),
                                              val_s, loader_s, forward_s,
                                              backward_s, loss_s,
                                              postproc_s, metric_s, writer_s))
        if self.mode != VAL:
            start = time()
            loss_str = VAL_LOSS_STR.format(epoch=epoch + 1,
                                           loss=running_val_loss.avg)
            self.writer.add_scalar('Loss/Val', running_val_loss.avg, epoch)
            for task, loss in self.val_loss_meters.meters.items():
                loss_str += TASK_LOSS.format(task, loss.avg.mean())
                self.writer.add_scalar('Loss/Validation_{}'.format(task),
                                       loss.avg.mean(), epoch)
                self.send_predictions_to_writer(inputs, predictions,
                                                labels, self.val_loss_meters,
                                                epoch, VAL)
            writer_s = time() - start
            logging.info(loss_str + 'Time: {:05.3f}'.format(time()-val_start))

        print_metrics(self.val_metrics)
        if self.mode == VAL:
            print_metrics(self.post_proc_metrics)
        self.val_loss_meters.reset()
        self.val_metrics.reset()
        self.post_proc_metrics.reset()
        self.val_loss = running_val_loss.avg
        running_val_loss.reset()

    def send_predictions_to_writer(self, inputs, predictions, labels, meters,
                                   epoch=1, state=VAL, save_to_disk=False):
        for task in meters.meters.keys():
            add_images_to_writer(inputs, predictions[task], labels[task],
                                 self.writer, epoch, task, state=state,
                                 sample_idx=self.sample_idx,
                                 save_to_disk=save_to_disk)

    def send_outputs_to_writer(self, outputs, labels, metrics, epoch=1,
                               state=VAL, save_to_disk=False):
        for task in metrics.metrics.keys():
            add_images_to_writer(None, outputs[task], labels[task],
                                 self.writer, epoch, task, state=state,
                                 sample_idx=self.sample_idx,
                                 save_to_disk=save_to_disk)

    def save_model(self, epoch):
        """Save model to checkpoint.

        Arguments:
            epoch {[int]} -- [Current epoch number]
        """
        if self.val_loss <= self.best_loss:
            self.best_loss = self.val_loss
            self.state = {"epoch": self.start_iter + epoch + 1,
                          "model_state": self.model.state_dict(),
                          "criterion_state": self.criterion.state_dict(),
                          "optimizer_state": self.optimizer.state_dict(),
                          "best_loss": self.best_loss}
            save_path, _, _, time_stamp = get_save_path(self.cfg,
                                                        self.best_loss)
            torch.save(self.state, save_path)
            deleted_old = delete_old_checkpoint(save_path)
            model_name = MODEL_NAME.format(time_stamp=time_stamp,
                                           best_loss=self.best_loss)

            logging.info("Saving checkpoint {} at epoch {}"
                         .format(model_name, epoch+1).center(LENGTH, '='))
            if deleted_old:
                logging.info('Deleted old checkpoints'.center(LENGTH, '='))
                self.plateau_count = 0
        else:
            self.plateau_count += 1
            logging.info("""Best Loss: {}. Current Loss: {}. /
                          No Checkpoint saved. Plateau_count: {}/{}"""
                         .format(self.best_loss, self.val_loss,
                                 self.plateau_count, self.patience))

    def stop_training(self, epoch):
        """Stop training if stop criteria is met.

        Arguments:
            epoch {[int]} -- [Current epoch number]

        Returns:
            [bool] -- [true/false to stop training]
        """
        if self.plateau_count == self.patience and self.early_stop:
            early_stop_str = EARLY_STOP_STR.format(epochs=epoch+1)
            patience_str = PATIENCE_STR.format(patience=self.plateau_count)
            logging.info((early_stop_str+patience_str).center(LENGTH, '='))

        return True


def get_exp_dir(cfg):
    """Get the name of experiments directory.

    Arguments:
        cfg {[dict]} -- [configuration settings for the experiment]

    Returns:
        exp_dir_path{[str]} -- [path to experiment directory]
        network_name{[str]} -- [Name of the network that includes encoder_name
        and decoder_name]
        exp_name{[str]} -- [Name of the experiment that includes all tasks
        learnt by network]
    """
    encoder_name = cfg[MODEL][ENCODER]
    decoder_name = cfg[MODEL][DECODER]
    imsize = cfg[DATA][IM_SIZE]
    base_dir = cfg[MODEL][PRETRAINED_PATH]

    # create results directory if it's not specified in cfg.
    if base_dir is None:
        base_dir = RESULTS_DIR

    network_name = EXPERIMENT_NAME.format(encoder_name=encoder_name,
                                          decoder_name=decoder_name,
                                          imsize=imsize)
    exp_name = '_'.join(cfg[TASKS].keys())

    # create experiment directory based on tasks and network type.
    exp_dir_path = os.path.join(base_dir, exp_name, network_name)
    if not os.path.exists(exp_dir_path):
        os.makedirs(exp_dir_path)

    return exp_dir_path, exp_name, network_name


def get_save_path(cfg, best_loss=None):
    """Get path to save best model.

    Arguments:
        cfg {[dict]} -- [configuration settings for the experiment]

    Keyword Arguments:
        best_loss {[float]} -- [best loss of network] (default: {None})

    Returns:
        path[str] -- path to save model
        network_name{[str]} -- [Name of the network that includes encoder_name
        and decoder_name]
        exp_name{[str]} -- [Name of the experiment that includes all tasks
        learnt by network]
        time_stamp[str] -- [Timestamp in YYYY-MM-DD-HH:MM:SS]
    """
    exp_dir_path, exp_name, network_name = get_exp_dir(cfg)
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    path = os.path.join(exp_dir_path, MODEL_NAME.format(time_stamp=time_stamp,
                                                        best_loss=best_loss))

    return path, network_name, exp_name, time_stamp


def get_writer(cfg):
    """Get tf.writer to log experiments data to Tensorboard.

    Arguments:
        cfg {[dict]} -- [configuration settings for the experiment]

    Returns:
        [tf.SummaryWriter] -- [Object to log experiments data to Tensorboard]
    """
    base_dir = cfg[MODEL][PRETRAINED_PATH]
    savelogs = cfg[PARAMS][SAVE_LOGS]
    _, exp_name, task_name, time_stamp = get_save_path(cfg)
    log_dir = os.path.join(base_dir, LOGS, task_name,
                           exp_name, "{}".format(time_stamp))
    if not os.path.exists(log_dir) and savelogs:
        os.makedirs(log_dir)
        return SummaryWriter(log_dir=log_dir)
    else:
        return None


def get_best_model(list_of_models):
    """Get best model from given list of models.

    Arguments:
        list_of_models {[list]} -- [List of trained models]

    Returns:
        [.pkl file] -- [Neural Network model with best loss]
    """
    best_loss = MILLION
    best_model = None
    for model in list_of_models:
        checkpoint_name = model.split('/')[-1].split(PKL)[0]
        loss = float(checkpoint_name.split('_')[-1])
        if loss < best_loss:
            best_loss = loss
            best_model = model
    return best_model


def get_checkpoint_list(exp_dir):
    """Get list of checkpoint model files.

    Arguments:
        exp_dir {[str]} -- [Path to current experiment directory]

    Returns:
        [list] -- [List of trained models]
    """
    return glob.glob(os.path.join(exp_dir, '*.pkl'))


def get_checkpoint(exp_dir):
    """Get checkpoint model file.

    Arguments:
        exp_dir {[str]} -- [Path to current experiment directory]

    Returns:
        checkpoint_name[str] -- [Name of the checkpoint file with best_loss]
        model(torch.nn.Module) -- Neural network module with best_loss
    """
    list_of_models = get_checkpoint_list(exp_dir)
    checkpoint_name = get_best_model(list_of_models)
    model = torch.load(checkpoint_name)
    return checkpoint_name, model


def if_checkpoint_exists(exp_dir):
    """Check if a checkpoint exists in given path.

    Arguments:
        exp_dir {[str]} -- [Path to current experiment directory]


    Returns:
        [bool] -- [True/False based on checkpoint existence]
    """
    list_of_models = get_checkpoint_list(exp_dir)

    return True if len(list_of_models) != 0 else False


def get_model(cfg):
    """Get the model and place it on devices specified in config.

    Arguments:
        cfg {[dict]} -- [configuration settings for the experiment]
        device {[torch.device]} -- [device to place data.]

    Returns:
        [model(torch.nn.Module) -- Neural network module defined for the
        experiment
    """
    model = get_encoder_decoder(cfg)
    return model


def place_on_multi_gpu(cfg, model):
    """Place model on multiple GPUS.

    Arguments:
        cfg {[dict]} -- [configuration settings for the experiment]
        model {[torch.nn.Module]} -- [Neural Network Model to place on GPUs]

    Returns:
        [torch.nn.Module] -- [Neural Network Model on Multiple GPUs]
    """
    gpus = list(range(torch.cuda.device_count()))
    multigpu = cfg[PARAMS][MULTIGPU]
    if len(gpus) > 1 and multigpu:
        model = nn.DataParallel(model, device_ids=gpus, dim=0)
    return model


def delete_old_checkpoint(path_to_checkpoint):
    """Delete an old checkpoint in given path.

    Arguments:
        path_to_checkpoint[str] -- path to save model

    Returns:
        [bool] -- [True/False]
    """
    removed = False
    path = '/'.join(path_to_checkpoint.split('/')[:-1])
    list_of_models = get_checkpoint_list(path)
    for model in list_of_models:
        if model not in path_to_checkpoint:
            os.remove(model)
            removed = True
    return removed


def print_metrics(metrics):
    """Print metrics.

    Arguments:
        metrics {dict} -- [metrics to be printed]
    """
    logging.info(METRICS_STR.center(LENGTH, '='))
    for task in metrics.metrics.keys():
        if metrics.metrics[task] is not None:
            score, class_iou = metrics.metrics[task].get_scores()
            if isinstance(score, dict):
                header = score.keys()
                t = PrettyTable(header)
                logging.info(task.center(LENGTH, '='))
                t.add_row(['{:05.3f}'.format(score[k] * 100)
                           for k in score.keys()])
                print(t)
                if task in [SEMANTIC, 'semantic_with_instance']:
                    print_table(class_iou, cat_labels)
                elif task == INSTANCE_PROBS:
                    print_table(class_iou, prob_labels)
                elif task == INSTANCE_CONTOUR:
                    print_table(class_iou, inst_labels)
                elif task == PANOPTIC:
                    t = PrettyTable(['Category']+list(class_iou.keys()))
                    for label in labels:
                        if label.ignoreInEval:
                            continue
                        row = ([label.name] + ['{:05.3f}'.format(
                            class_iou[metric][label.id]*100)
                            for metric in class_iou.keys()])
                        t.add_row(row)
                    print(t)


def print_table(data, labels):
    t = PrettyTable([labels[k].name for k in data.keys()])
    t.add_row(['{:05.3f}'.format(v*100) for v in data.values()])
    print(t)
