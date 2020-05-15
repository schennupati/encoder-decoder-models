"""Writer utilities to support sending data to Tensorboard or disk ."""

import os
import torch
import numpy as np
import matplotlib
import torch.nn.functional as F
from utils.im_utils import decode_segmap, labels, inst_labels, prob_labels, \
    get_color_inst, to_rgb
import cv2
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
                         dataformats=HWC, inputs=None, scale_factor=0.25):
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
                    save_images_to_disk=save_image_to_disk, inputs=inputs,
                    scale_factor=scale_factor)
    write_task_data(targets, writer, epoch, task,
                    state, sample_idx, data_type=GT,
                    save_images_to_disk=save_images_to_disk, inputs=inputs,
                    scale_factor=scale_factor)


def write_task_data(data, writer, epoch, task=SEMANTIC,
                    state=TRAIN, sample_idx=0, data_type=GT,
                    save_images_to_disk=False, inputs=None, scale_factor=0.25):
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
    data = get_sample(data, task, data_type, sample_idx)
    rgb = inputs[sample_idx, ...].squeeze()
    rgb = normalize(rgb, scale_factor)
    data = data.squeeze() if not isinstance(data, list) else data
    if len(data) != 0:
        output = generate_task_visuals(data, task, rgb)
    else:
        output = rgb
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


def get_sample(data, task, data_type=GT, sample_idx=0):
    """Get a sample from batch of data

    Arguments:
        data {dict} -- [Predictions(DET) or Targets(GT)]

    Keyword Arguments:
        data_type {str} -- [Ground truth or prediction] (default: {GT})
        sample_idx {int} -- [Index of data in batch to be visualized]
         (default: {0})

    Returns:
        [np.ndarray] -- [Selected sample placed on cpu in numpy datatype]
    """
    if data_type == GT and isinstance(data, dict):
        if task == 'bounding_box':
            data = data['bboxes'][sample_idx[0]]
        else:
            data = data['targets'][sample_idx[0]]
    else:
        data = data[sample_idx[0]]

    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    return data


def generate_task_visuals(data, task, rgb=None):
    """[Generate visualization image for a task specific data.]

    Arguments:
        data {dict} -- [Predictions(DET) or Targets(GT)]
        task {str} -- [Type of task] (default: {SEMANTIC})

    Returns:
        [np.ndarray] -- [Visualization image for the given task]
    """
    if task == SEMANTIC:
        data = decode_segmap(data, nc=19, labels=labels)
    elif task == 'semantic_with_instance':
        data = decode_segmap(data, nc=20, labels=labels)
    elif task == INSTANCE_CONTOUR:
        data = decode_segmap(data, nc=9, labels=inst_labels)
    elif task == INSTANCE_REGRESSION:
        data = get_color_inst(data)
    elif task == INSTANCE_HEATMAP:
        data = np.clip(data, 0, np.max(data))
    elif task == INSTANCE_PROBS:
        data = decode_segmap(data, nc=2, labels=prob_labels)
    elif task == 'bounding_box':
        data = get_bbox(rgb.astype(np.uint8), data)
    elif task == PANOPTIC:
        data = data[..., 0] + \
            256 * data[..., 1] + \
            256 * 256 * data[..., 2]
        data = to_rgb(data)
    else:
        data = data
    return data


def get_bbox(rgb, bboxes):
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    if bboxes.shape[0] == 0:
        return rgb
    color = (255, 255, 0)
    thickness = 1
    if len(bboxes.shape) == 1:
        pt1 = (int(bboxes[0]), int(bboxes[1]))
        pt2 = (int(bboxes[2]), int(bboxes[3]))
        rgb = cv2.rectangle(rgb, pt1, pt2, color, thickness)
    else:
        for bbox in bboxes:
            pt1 = (int(bbox[0]), int(bbox[1]))
            pt2 = (int(bbox[2]), int(bbox[3]))

            rgb = cv2.rectangle(rgb, pt1, pt2, color, thickness)
    return torch.tensor(rgb.get())


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
                         save_to_disk=False, scale_factor=0.25):
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
                             dataformats=HWC, inputs=inputs,
                             scale_factor=scale_factor)


def normalize(rgb, scale_factor):
    rgb = F.interpolate(rgb.unsqueeze(1), scale_factor=scale_factor,
                        mode='bilinear')
    rgb = rgb.cpu().numpy() if torch.is_tensor(rgb) else to_rgb
    rgb = rgb.squeeze().transpose(1, 2, 0)
    rgb *= [0.229, 0.224, 0.225]
    rgb += [0.485, 0.456, 0.406]
    rgb *= 255
    return rgb
