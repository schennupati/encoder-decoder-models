"""Writer utilities to support sending data to Tensorboard or disk ."""

import os
import numpy as np
import matplotlib
from utils.im_utils import decode_segmap, labels, inst_labels, prob_labels, \
    get_color_inst
from utils.constants import PLOTS_DIR, TRAIN, VAL, HWC, CHW, INPUT_IMAGE, \
    IMAGES, SEMANTIC, INSTANCE_CONTOUR, INSTANCE_REGRESSION, INSTANCE_HEATMAP,\
    INSTANCE_PROBS, INPUT_SAVE_NAME, OUTPUT_SAVE_NAME, INPUT_WRITER_NAME, \
    OUTPUT_WRITER_NAME, DET, GT, PNG


def add_input_to_writer(inputs, writer, epoch,
                        state=TRAIN, sample_idx=0, save_to_disk=False,
                        dataformats=HWC):
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
    img = inputs[sample_idx, ...]
    writer.add_image(name_to_writer, img, epoch, dataformats=dataformats)
    if save_to_disk:
        save_image_to_disk(img, name_to_save, dataformats)


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
        save_to_disk {bool} -- Whether to save images to disk (default: {False})
        dataformats {str} -- [Channel first of last] (default: {HWC})
    """

    write_task_data(predictions, writer, epoch, task,
                    state, sample_idx, data_type=DET)
    write_task_data(targets, writer, epoch, task,
                    state, sample_idx, data_type=GT)


def write_task_data(data, writer, epoch, task=SEMANTIC,
                    state=TRAIN, sample_idx=0, data_type=GT):
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
    """
    data = get_sample(data, task, sample_idx, state)
    output = generate_task_visuals(data, task)
    name_to_writer = OUTPUT_WRITER_NAME.format(
        state=state, data_type=DET, task=task)
    add_image_to_writer(output, epoch, writer, name_to_writer)


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


def get_sample(data, task, state=TRAIN, sample_idx=0):
    """Get a sample from batch of data

    Arguments:
        data {dict} -- [Predictions(DET) or Targets(GT)]
        task {str} -- [Type of task] (default: {SEMANTIC})

    Keyword Arguments:
        state {str} -- [training or validation state] (default: {TRAIN})
        sample_idx {int} -- [Index of data in batch to be visualized]
         (default: {0})

    Returns:
        [np.ndarray] -- [Selected sample placed on cpu in numpy datatype]
    """
    data = data[task][sample_idx, ...]
    data = data.detach() if state == TRAIN else data
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
    if len(data.shape) == 2:
        data = data.unsqueeze(0)
    if task == SEMANTIC:
        data = decode_segmap(data, nc=19, labels=labels)
    elif task == INSTANCE_CONTOUR:
        data = decode_segmap(data, nc=11, labels=inst_labels)
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
    image = (image.permute(1, 2, 0)).numpy() if dataformats == CHW else image
    fname = os.path.join(path, name + PNG)
    matplotlib.image.imsave(fname, image.astype(np.uint8))


def add_images_to_writer(inputs, predictions, targets, writer,
                         epoch, task=SEMANTIC, state=TRAIN, save_to_disk=False):
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
        save_to_disk {bool} -- Whether to save images to disk (default: {False})
    """
    batch_size = predictions.shape[0]
    sample_idx = np.random.randint(0, batch_size, size=1)

    add_input_to_writer(inputs, writer, epoch, sample_idx, state,
                        save_to_disk, dataformats=CHW)
    add_output_to_writer(predictions, targets, writer, epoch,
                         task, state, sample_idx, save_to_disk, HWC)
