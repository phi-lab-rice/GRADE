"""SML evaluation: dense metric depth from a trained Scale Map Learner.

The validate() and log_evaluation_results() functions below are copied
verbatim from RadarCam-Depth/SML/sml_main.py (the baseline's train() half is
replaced by sml_train_rice.py and is not vendored, which also drops the
tensorboard dependency).
"""

import os
import time

import numpy as np
import torch
import torch.utils.data

import data.data_utils as data_utils
import data.SML_dataset as UTV
import modules.midas.transforms as transforms
import modules.midas.utils as utils
import utils.eval_utils as eval_utils
from utils.log_utils import log


def validate(
        image_paths,
        radar_paths,
        gt_paths,
        sparse_gt_paths,
        rcnet_paths,

        best_results,
        ScaleMapLearner,
        step,
        min_radar_valid_depth,
        max_radar_valid_depth,
        min_eval_depth,
        max_eval_depth,
        output_path,

        mono_pred_paths = None,
        mono_ga_paths = None,

        save_output = False,
        random_sample = False,
        random_sample_size = 1000,
        log_path = None,
        depth_predictor = 'dpt_hybrid',
        ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if random_sample:
        random_sample_idx = np.random.choice(len(image_paths), random_sample_size, replace=False)
        image_paths = [image_paths[idx] for idx in random_sample_idx]
        radar_paths = [radar_paths[idx] for idx in random_sample_idx]
        gt_paths = [gt_paths[idx] for idx in random_sample_idx]
        sparse_gt_paths = [sparse_gt_paths[idx] for idx in random_sample_idx]
        rcnet_paths = [rcnet_paths[idx] for idx in random_sample_idx]
        if mono_ga_paths is not None:
            mono_pred_paths = [mono_pred_paths[idx] for idx in random_sample_idx]
        if mono_ga_paths is not None:
            mono_ga_paths = [mono_ga_paths[idx] for idx in random_sample_idx]

    val_dataloader = torch.utils.data.DataLoader(
        UTV.SML_dataset(
            image_paths = image_paths,
            radar_paths = radar_paths,
            gt_paths = gt_paths,
            sparse_gt_paths = sparse_gt_paths,
            rcnet_paths = rcnet_paths,
            mono_pred_paths = mono_pred_paths,
            mono_ga_paths = mono_ga_paths,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=1)

    n_sample = len(val_dataloader)
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)
    abs_rel = np.zeros(n_sample)
    sq_rel = np.zeros(n_sample)
    delta1 = np.zeros(n_sample)

    save_file_name = os.path.join(output_path, 'RadarCam-Depth')
    if save_output:
        os.makedirs(save_file_name, exist_ok=True)
        os.makedirs(os.path.join(save_file_name, 'sml_depth'), exist_ok=True)
        os.makedirs(os.path.join(save_file_name, 'sml_depth_color'), exist_ok=True)

    time_start = time.time()

    for idx, inputs in enumerate(val_dataloader):
        inputs = [in_.to(device) for in_ in inputs]

        image, _, _, _, sparse_gt, rcnet, mono_ga_pos = inputs
        input_height, input_width = image.shape[1:3]

        # transform
        ScaleMapLearner_transform = transforms.get_transforms(depth_predictor, 'void', '150')

        rcnet_valid = (rcnet < max_radar_valid_depth) * (rcnet > min_radar_valid_depth)
        rcnet_valid = rcnet_valid.bool()
        rcnet[~rcnet_valid] = np.inf  # set invalid depth
        rcnet = 1.0 / rcnet
        mono_ga = 1.0 / mono_ga_pos

        rcnet = rcnet.squeeze().cpu().numpy()
        rcnet_valid = rcnet_valid.squeeze().cpu().numpy()
        mono_ga_pos = mono_ga_pos.squeeze().cpu().numpy()
        int_depth = mono_ga.squeeze().cpu().numpy()

        int_scales = np.ones_like(int_depth)
        int_scales[rcnet_valid] = rcnet[rcnet_valid] / int_depth[rcnet_valid]
        int_scales = utils.normalize_unit_range(int_scales.astype(np.float32))

        # transforms
        sample = {'image': image.squeeze().cpu().numpy(),
                  'int_depth': int_depth,
                  'int_scales': int_scales,
                  'int_depth_no_tf': int_depth}

        sample = ScaleMapLearner_transform(sample)

        x = torch.cat([sample['int_depth'], sample['int_scales']], 0)
        x = x.to(device)
        d = sample['int_depth_no_tf'].to(device)

        with torch.no_grad():
            sml_pred, sml_scales = ScaleMapLearner.forward(x.unsqueeze(0), d.unsqueeze(0))
            sml_pred = (
                torch.nn.functional.interpolate(
                    1.0 / sml_pred,
                    size=(input_height, input_width),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        sparse_gt = np.squeeze(sparse_gt.cpu().numpy())
        validity_map = np.where(sparse_gt > 0, 1, 0)

        # Select valid regions to evaluate
        validity_mask = np.where(validity_map > 0, 1, 0)
        min_max_mask = np.logical_and(
            sparse_gt > min_eval_depth,
            sparse_gt < max_eval_depth)
        mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)
        output_depth = sml_pred[mask]
        sparse_gt = sparse_gt[mask]

        # Compute validation metrics
        mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * sparse_gt)
        rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * sparse_gt)
        imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * sparse_gt)
        irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * sparse_gt)
        abs_rel[idx] = eval_utils.mean_abs_rel_err(1000.0 * output_depth, 1000.0 * sparse_gt)
        sq_rel[idx] = eval_utils.mean_sq_rel_err(1000.0 * output_depth, 1000.0 * sparse_gt)
        delta1[idx] = eval_utils.thr_acc(output_depth, sparse_gt)
        print(mae[idx], rmse[idx], imae[idx], irmse[idx], abs_rel[idx], sq_rel[idx], delta1[idx])

        if save_output:
            basename = os.path.basename(image_paths[idx]).split('.')[0] + '.png'
            print('Saving output {}'.format(basename))
            sky_mask = mono_ga_pos >= 200
            sml_pred[sky_mask] = mono_ga_pos[sky_mask]
            data_utils.save_depth(sml_pred, os.path.join(save_file_name, 'sml_depth', basename))
            data_utils.save_color_depth(sml_pred, os.path.join(save_file_name, 'sml_depth_color', basename))

    time_end = time.time()
    print('Time taken: {:.4f} seconds'.format(time_end - time_start))
    print('average time per sample: {:.4f} seconds'.format((time_end - time_start) / len(image_paths)))

    # Compute mean metrics
    mae = np.mean(mae)
    rmse = np.mean(rmse)
    imae = np.mean(imae)
    irmse = np.mean(irmse)
    abs_rel = np.mean(abs_rel)
    sq_rel = np.mean(sq_rel)
    delta1 = np.mean(delta1)

    # Print validation results to console
    log_evaluation_results(
        title='Validation results',
        mae=mae,
        rmse=rmse,
        imae=imae,
        irmse=irmse,
        abs_rel=abs_rel,
        sq_rel=sq_rel,
        delta1=delta1,
        step=step,
        log_path=log_path)

    n_improve = 0
    if np.round(mae, 4) < np.round(best_results['mae'], 4):
        n_improve = n_improve + 1
    if np.round(rmse, 4) < np.round(best_results['rmse'], 4):
        n_improve = n_improve + 1
    if np.round(imae, 4) < np.round(best_results['imae'], 4):
        n_improve = n_improve + 1
    if np.round(irmse, 4) < np.round(best_results['irmse'], 4):
        n_improve = n_improve + 1
    if np.round(abs_rel, 4) < np.round(best_results['abs_rel'], 4):
        n_improve = n_improve + 1
    if np.round(sq_rel, 4) < np.round(best_results['sq_rel'], 4):
        n_improve = n_improve + 1
    if np.round(delta1, 4) > np.round(best_results['delta1'], 4):
        n_improve = n_improve + 1

    if n_improve > 3:
        best_results['step'] = step
        best_results['mae'] = mae
        best_results['rmse'] = rmse
        best_results['imae'] = imae
        best_results['irmse'] = irmse
        best_results['abs_rel'] = abs_rel
        best_results['sq_rel'] = sq_rel
        best_results['delta1'] = delta1

    log_evaluation_results(
        title='Best results',
        mae=best_results['mae'],
        rmse=best_results['rmse'],
        imae=best_results['imae'],
        irmse=best_results['irmse'],
        step=best_results['step'],
        abs_rel=best_results['abs_rel'],
        sq_rel=best_results['sq_rel'],
        delta1=best_results['delta1'],
        log_path=log_path)

    return best_results


def log_evaluation_results(title,
                           mae,
                           rmse,
                           imae,
                           irmse,
                           abs_rel=None,
                           sq_rel=None,
                           delta1=None,
                           step=-1,
                           log_path=None):

    log(title + ':', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE', 'Abs_Rel', 'Sq_Rel', 'Delta1'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        step,
        mae,
        rmse,
        imae,
        irmse,
        abs_rel,
        sq_rel,
        delta1),
        log_path)
