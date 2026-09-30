"""RC-Net inference: quasi-dense depth from a trained RC-Net.

The run(), forward() and log_network_settings() functions below are copied
verbatim from RadarCam-Depth/RCNet/rcnet_main.py (the baseline's train() half
is replaced by rcnet_train_rice.py and is not vendored, which also drops the
tensorboard dependency).
"""

import os

import numpy as np
import torch
import torch.utils.data
import torchvision
from tqdm.auto import tqdm

from data import data_utils, datasets
from rcnet_model import RCNetModel
from rcnet_transforms import Transforms
from utils.log_utils import log


def run(save_root,

        image_paths,
        radar_paths,
        gt_paths,

        restore_path,
        patch_size,
        normalized_image_range,

        encoder_type,
        n_filters_encoder_image,
        n_neurons_encoder_depth,
        decoder_type,
        n_filters_decoder,
        weight_initializer,
        activation_func,
        response_thr=0.5):

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    Read input paths
    '''

    n_sample = len(image_paths)

    assert n_sample == len(radar_paths)
    assert n_sample == len(gt_paths)

    '''
    Set up inputs and outputs
    '''
    depth_predicted_paths = []
    response_predicted_paths = []
    depth_predicted_color_paths = []

    inputs_outputs = [
        [
            'training',
            image_paths,
            radar_paths,
            gt_paths,
            depth_predicted_paths,
            depth_predicted_color_paths,
            response_predicted_paths,
        ]
    ]

    '''
    Set up the model
    '''
    # Build network
    rcnet_model = RCNetModel(
        input_channels_image=3,
        input_channels_depth=3,
        input_patch_size_image=patch_size,
        encoder_type=encoder_type,
        n_filters_encoder_image=n_filters_encoder_image,
        n_neurons_encoder_depth=n_neurons_encoder_depth,
        decoder_type=decoder_type,
        n_filters_decoder=n_filters_decoder,
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        device=device)

    rcnet_model.eval()
    rcnet_model.to(device)
    rcnet_model.data_parallel()

    parameters_rcnet_model = rcnet_model.parameters()

    step, _ = rcnet_model.restore_model(restore_path)

    log('Restoring checkpoint from: \n{}\n'.format(restore_path))

    log_network_settings(
        log_path=None,
        # Network settings
        encoder_type=encoder_type,
        n_filters_encoder_image=n_filters_encoder_image,
        n_neurons_encoder_depth=n_neurons_encoder_depth,
        decoder_type=decoder_type,
        n_filters_decoder=n_filters_decoder,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_model=parameters_rcnet_model)

    '''
    Process each set of input and outputs
    '''
    for paths in inputs_outputs:
        # Unpack inputs and outputs
        tag, \
        image_paths, \
        radar_paths, \
        ground_truth_paths, \
        depth_predicted_paths, \
        depth_predicted_color_paths, \
        response_predicted_paths, = paths

        # Create output paths for depth and response
        for radar_path in radar_paths:
            # Create path and store
            file_id = os.path.basename(radar_path).split('.')[0]
            depth_predicted_path = os.path.join(save_root, 'depth_predicted', file_id + '.png')
            depth_predicted_paths.append(depth_predicted_path)

            depth_predicted_color_path = os.path.join(save_root, 'depth_predicted_colors', file_id + '.png')
            depth_predicted_color_paths.append(depth_predicted_color_path)

            response_predicted_path = os.path.join(save_root, 'response_predicted', file_id + '.png')
            response_predicted_paths.append(response_predicted_path)

        # Create directories
        depth_predicted_dirpaths = np.unique([os.path.dirname(path) for path in depth_predicted_paths])
        depth_predicted_color_dirpaths = np.unique([os.path.dirname(path) for path in depth_predicted_color_paths])
        response_predicted_dirpaths = np.unique([os.path.dirname(path) for path in response_predicted_paths])
        for dirpaths in [depth_predicted_dirpaths, depth_predicted_color_dirpaths, response_predicted_dirpaths]:
            for dirpath in dirpaths:
                os.makedirs(dirpath, exist_ok=True)

        # Set up dataloader
        dataloader = torch.utils.data.DataLoader(
            datasets.RCNetInferenceDataset(
                image_paths=image_paths,
                radar_paths=radar_paths,
                ground_truth_paths=ground_truth_paths),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False)

        transforms = Transforms(
            normalized_image_range=normalized_image_range)

        n_sample = len(dataloader)

        print('Processing {} samples...'.format(n_sample))

        # Iterate through data loader
        progress = tqdm(dataloader, total=n_sample, desc=f'{tag} inference', unit='frame')
        for sample_idx, data in enumerate(progress):
            with torch.no_grad():
                data = [
                    datum.to(device) for datum in data
                ]

                image, radar_points, ground_truth = data
                bounding_boxes_list = []

                pad_size_x = patch_size[1] // 2
                radar_points = radar_points.squeeze(dim=0)

                if radar_points.ndim == 1:
                    # Expand to 1 x 3
                    radar_points = np.expand_dims(radar_points, axis=0)

                # get the shifted radar points after padding
                for radar_point_idx in range(0, radar_points.shape[0]):
                    # Set radar point to the center of the patch
                    radar_points[radar_point_idx, 0] = radar_points[radar_point_idx, 0] + pad_size_x
                    bounding_box = torch.zeros(4)
                    bounding_box[0] = radar_points[radar_point_idx, 0] - pad_size_x
                    bounding_box[1] = 0
                    bounding_box[2] = radar_points[radar_point_idx, 0] + pad_size_x
                    bounding_box[3] = image.shape[-2]
                    bounding_boxes_list.append(bounding_box)

                bounding_boxes_list = [torch.stack(bounding_boxes_list, dim=0)]

                [image], [radar_points], [bounding_boxes_list] = transforms.transform(
                    images_arr=[image],
                    points_arr=[radar_points],
                    bounding_boxes_arr=[bounding_boxes_list],
                    random_transform_probability=0.0)

                output_depth, output_response, _, inference_failed = forward_with_fallback(
                    model=rcnet_model,
                    image=image,
                    radar_points=radar_points,
                    bounding_boxes_list=bounding_boxes_list,
                    response_thr=response_thr,
                    device=device)

                output_depth = np.squeeze(output_depth.cpu().numpy())
                output_response = np.squeeze(output_response.cpu().numpy())

                if inference_failed:
                    tqdm.write(
                        'ERROR: RCNet inference failed for {}: '
                        'no positive response at threshold 0.0'.format(
                            os.path.basename(radar_paths[sample_idx])))

            '''
            Save outputs
            '''
            data_utils.save_depth(output_depth, depth_predicted_paths[sample_idx])
            data_utils.save_color_depth(output_depth, depth_predicted_color_paths[sample_idx])
            data_utils.save_response(output_response, response_predicted_paths[sample_idx])
def forward_with_fallback(model,
                          image,
                          radar_points,
                          bounding_boxes_list,
                          response_thr=0.5,
                          device=torch.device('cuda')):
    '''Run inference, lowering the response threshold only when necessary.'''
    output_depth, output_response = forward(
        model=model,
        image=image,
        radar_points=radar_points,
        bounding_boxes_list=bounding_boxes_list,
        response_thr=response_thr,
        device=device)

    threshold = float(response_thr)
    while not torch.any(output_depth > 0).item() and threshold > 0.0:
        threshold = max(0.0, threshold - 0.05)
        output_depth, output_response = forward(
            model=model,
            image=image,
            radar_points=radar_points,
            bounding_boxes_list=bounding_boxes_list,
            response_thr=threshold,
            device=device)

    inference_failed = not torch.any(output_depth > 0).item()
    return output_depth, output_response, threshold, inference_failed


def forward(model, image, radar_points, bounding_boxes_list, response_thr=0.5, device=torch.device('cuda')):
    # Determine crop size for possible radar correspondence
    patch_size = model.input_patch_size_image
    pad_size = patch_size[1] // 2

    image = torchvision.transforms.functional.pad(
        image,
        (pad_size, 0, pad_size, 0),
        padding_mode='edge')
    start_y = image.shape[-2] - patch_size[0]

    output_tiles = []
    if radar_points.dim() == 3:
        # Convert to 1 x N x 3 to N x 3
        radar_points = torch.squeeze(radar_points, dim=0)

    x_shifts = radar_points[:, 0].clone()

    height = image.shape[-2]
    crop_height = height - patch_size[0]

    output_crops = model.forward(
        image=image,
        point=radar_points,
        bounding_boxes=bounding_boxes_list,
        return_logits=False)
    for output_crop, x in zip(output_crops, x_shifts):
        output = torch.zeros([1, image.shape[-2], image.shape[-1]], device=device)

        output_crop = torch.where(output_crop < response_thr, torch.zeros_like(output_crop), output_crop)
        # Add crop to output
        output[:, crop_height:, int(x) - pad_size:int(x) + pad_size] = output_crop
        output_tiles.append(output)

    output_tiles = torch.cat(output_tiles, dim=0)
    output_tiles = output_tiles[:, :, pad_size:-pad_size]

    # Find the max response over all tiles
    output_response, output_indices = torch.max(output_tiles, dim=0, keepdim=True)
    # Fill a floating-point depth map based on the selected radar point.
    output_depth = torch.zeros_like(output_response)
    for point_idx in range(radar_points.shape[0]):
        output_depth = torch.where(
            output_indices == point_idx,
            torch.full_like(
                output_response,
                fill_value=float(radar_points[point_idx, 2]),
            ),
            output_depth)

    # Leave as 0s if we did not predict
    output_depth = torch.where(
        output_response == 0,
        torch.zeros_like(output_depth),
        output_depth)

    return output_depth, output_response


def log_network_settings(log_path,
                         # Network settings
                         encoder_type,
                         n_filters_encoder_image,
                         n_neurons_encoder_depth,
                         decoder_type,
                         n_filters_decoder,
                         # Weight settings
                         weight_initializer,
                         activation_func,
                         parameters_model=[]):
    # Computer number of parameters
    n_parameter = sum(p.numel() for p in parameters_model)

    n_parameter_text = 'n_parameter={}'.format(n_parameter)
    n_parameter_vars = []

    log('Network settings:', log_path)
    log('encoder_type={}'.format(encoder_type),
        log_path)
    log('n_filters_encoder_image={}'.format(n_filters_encoder_image),
        log_path)
    log('n_neurons_encoder_depth={}'.format(n_neurons_encoder_depth),
        log_path)
    log('decoder_type={}'.format(decoder_type),
        log_path)
    log('n_filters_decoder={}'.format(
        n_filters_decoder),
        log_path)
    log('', log_path)

    log('Weight settings:', log_path)
    log(n_parameter_text.format(*n_parameter_vars),
        log_path)
    log('weight_initializer={}  activation_func={}'.format(
        weight_initializer, activation_func),
        log_path)
    log('', log_path)
