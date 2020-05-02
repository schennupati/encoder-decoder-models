"""Writer utilities to support sending data to Tensorboard or disk ."""

import os
import torch
import numpy as np
import matplotlib
from utils.im_utils import decode_segmap, labels, inst_labels, prob_labels, \
    get_color_inst
from utils.constants import PLOTS_DIR, TRAIN, VAL, HWC, CHW, INPUT_IMAGE, \
    IMAGES, SEMANTIC, INSTANCE_CONTOUR, INSTANCE_REGRESSION, INSTANCE_HEATMAP,\
    INSTANCE_PROBS, INPUT_SAVE_NAME, OUTPUT_SAVE_NAME, INPUT_WRITER_NAME, \
    OUTPUT_WRITER_NAME, DET, GT, PNG, PANOPTIC, PANOPTIC_IMAGE


def add_input_to_writer(inputs, writer, epoch, state=TRAIN, sample_idx=0,
                        save_to_disk=False, dataformats=HWC):
    """Add network input to writer.

    Arguments:
        inputs {torch.tensor} -- [Network input]
        writer {tf.writer} -- [Summary writer which writes to Tensorboard]
        epoch {int} -- [Current epoch of training]

    Keyword Arguments:
        state {str} -- [training or validation state] (default: {TRAIN})
        sample_idx {int} -- [Index of data in batch to be visualized]
         (default: {0})
        save_to_disk {bool} -- Whether to save image to disk (default: {False})
        dataformats {str} -- [Channel first of last] (default: {HWC})
    """
    name_to_writer = INPUT_WRITER_NAME.format(state=state)
    name_to_save = INPUT_SAVE_NAME.format(epoch=epoch, state=state)
    img = inputs[sample_idx, ...].squeeze()
    writer.add_image(name_to_writer, img, epoch, dataformats=dataformats)
    if save_to_disk:
        save_image_to_disk(img, name_to_save, sample_idx, dataformats)


def add_output_to_writer(predictions, targets, writer, epoch,
                         task=SEMANTIC, state=TRAIN, sample_idx=0,
                         save_images_to_disk=False,
                         dataformats=HWC):
    """Add network output to writer.

    Arguments:
        predictions {dict} -- [prediction made by network]
        targets {dict} -- [ground truth]
        writer {tf.writer} -- [Summary writer which writes to Tensorboard]
        epoch {int} -- [Current epoch of training]

    Keyword Arguments:
        task {str} -- [Type of task] (default: {SEMANTIC})
        state {str} -- [training or validation state] (default: {TRAIN})
        sample_idx {int} -- [Index of data in batch to be visualized]
         (default: {0})
        save_images_to_disk {bool} -- Whether to save images to disk
         (default: {False})
        dataformats {str} -- [Channel first of last] (default: {HWC})
    """

    write_task_data(predictions, writer, epoch, task,
                    state, sample_idx, data_type=DET,
                    save_images_to_disk=save_image_to_disk)
    write_task_data(targets, writer, epoch, task,
                    state, sample_idx, data_type=GT,
                    save_images_to_disk=save_images_to_disk)


def write_task_data(data, writer, epoch, task=SEMANTIC,
                    state=TRAIN, sample_idx=0, data_type=GT,
                    save_images_to_disk=False):
    """Writes task data to tf.writer

    Arguments:
        data {dict} -- [Predictions(DET) or Targets(GT)]
        writer {tf.writer} -- [Summary writer which writes to Tensorboard]
        epoch {int} -- [Current epoch of training]

    Keyword Arguments:
        task {str} -- [Type of task] (default: {SEMANTIC})
        state {str} -- [training or validation state] (default: {TRAIN})
        sample_idx {int} -- [Index of data in batch to be visualized]
         (default: {0})
        data_type {str} -- [Ground truth or prediction] (default: {GT})
        save_images_to_disk {bool} -- Whether to save images to disk
         (default: {False})
    """
    data = get_sample(data, state, sample_idx)
    output = generate_task_visuals(data.squeeze(), task)
    name_to_writer = OUTPUT_WRITER_NAME.format(
        state=state, data_type=data_type, task=task)
    name_to_save = OUTPUT_SAVE_NAME.format(epoch=epoch, data_type=data_type,
                                           task=task, state=state)
    add_image_to_writer(output, writer, epoch, name_to_writer)
    if save_images_to_disk:
        save_image_to_disk(output, name_to_save, sample_idx)


def add_image_to_writer(img, writer, epoch, name_to_writer, dataformats=HWC):
    """Adds an image to writer.

    Arguments:
        img {np.ndarray} -- [Image to be written to Tensorboard]
        writer {tf.writer} -- [Summary writer which writes to Tensorboard]
        epoch {int} -- [Current epoch of training]
        name_to_writer {str} -- [Identifier for Tensorboard]

    Keyword Arguments:
        dataformats {str} -- [Channel first of last] (default: {HWC})
    """
    writer.add_image(name_to_writer, img, epoch, dataformats=dataformats)


def get_sample(data, state=TRAIN, sample_idx=0):
    """Get a sample from batch of data

    Arguments:
        data {dict} -- [Predictions(DET) or Targets(GT)]

    Keyword Arguments:
        state {str} -- [training or validation state] (default: {TRAIN})
        sample_idx {int} -- [Index of data in batch to be visualized]
         (default: {0})

    Returns:
        [np.ndarray] -- [Selected sample placed on cpu in numpy datatype]
    """
    data = data[sample_idx, ...]
    if torch.is_tensor(data):
        data = data.cpu().numpy()
    return data


def generate_task_visuals(data, task):
    """[Generate visualization image for a task specific data.]

    Arguments:
        data {dict} -- [Predictions(DET) or Targets(GT)]
        task {str} -- [Type of task] (default: {SEMANTIC})

    Returns:
        [np.ndarray] -- [Visualization image for the given task]
    """
    if task == SEMANTIC:
        data = decode_segmap(data, nc=19, labels=labels)
    elif task == INSTANCE_CONTOUR:
        data = decode_segmap(data, nc=9, labels=inst_labels)
    elif task == INSTANCE_REGRESSION:
        data = get_color_inst(data.transpose(1, 2, 0))
    elif task == INSTANCE_HEATMAP:
        data = np.clip(data, 0, np.max(data))
    elif task == INSTANCE_PROBS:
        data = decode_segmap(data, nc=2, labels=prob_labels)
    else:
        data = data
    return data


def save_image_to_disk(image, name, sample_idx=0,
                       dataformats=HWC, path=PLOTS_DIR):
    """Save give image to disk

    Arguments:
        image {np.ndarray} -- [Image to be saved to disk]
        name {str} -- [name of the file to be saved.]

    Keyword Arguments:
        sample_idx {int} -- [Index of data in batch to be visualized]
         (default: {0})
        dataformats {str} -- [Channel first of last] (default: {HWC})
        path {str} -- [root_path to save images] (default: {RESULTS_DIR})
    """
    shape = image.shape
    if len(shape) > 3:
        image = image[sample_idx, ...]
        image = image.squeeze()
    elif len(shape) == 2:
        image = image.unsqueeze(0)
    image = image.permute(1, 2, 0) if dataformats == CHW else image
    image = image.numpy() if torch.is_tensor(image) else image
    fname = os.path.join(path, name + PNG)
    matplotlib.image.imsave(fname, (image*225).astype(np.uint8))


def add_images_to_writer(inputs, predictions, targets, writer,
                         epoch, task=SEMANTIC, state=TRAIN, sample_idx=0,
                         save_to_disk=False):
    """[Adds images of inputs/outputs to tf.writer]

    Arguments:
        inputs {torch.tensor} - - [Network input]
        predictions {dict} - - [prediction made by network]
        targets {dict} -- [ground truth]
        writer {tf.writer} -- [Summary writer which writes to Tensorboard]
        epoch {int} -- [Current epoch of training]

    Keyword Arguments:
        task {str} -- [Type of task] (default: {SEMANTIC})
        state {str} -- [training or validation state] (default: {TRAIN})
        sample_idx {int} -- [Index of data in batch to be visualized]
        save_to_disk {bool} -- Whether to save images to disk (default: {False})
    """
    if task == PANOPTIC:
        predictions = predictions[PANOPTIC_IMAGE]
        targets = targets[PANOPTIC_IMAGE]
    if inputs is not None:
        add_input_to_writer(inputs, writer, epoch, state, sample_idx,
                            save_to_disk, dataformats=CHW)
    add_output_to_writer(predictions, targets, writer, epoch,
                         task, state, sample_idx, save_to_disk,
                         dataformats=HWC)
