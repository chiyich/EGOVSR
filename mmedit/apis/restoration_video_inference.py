# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp

import mmcv
import numpy as np
import torch

from mmedit.datasets.pipelines import Compose

VIDEO_EXTENSIONS = ('.mp4', '.mov')


def pad_sequence(data, window_size):
    padding = window_size // 2

    data = torch.cat([
        data[:, 1 + padding:1 + 2 * padding].flip(1), data,
        data[:, -1 - 2 * padding:-1 - padding].flip(1)
    ],
                     dim=1)

    return data


def restoration_video_inference(model,
                                img_dir,
                                window_size,
                                start_idx,
                                filename_tmpl,
                                max_seq_len=None):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img_dir (str): Directory of the input video.
        window_size (int): The window size used in sliding-window framework.
            This value should be set according to the settings of the network.
            A value smaller than 0 means using recurrent framework.
        start_idx (int): The index corresponds to the first frame in the
            sequence.
        filename_tmpl (str): Template for file name.
        max_seq_len (int | None): The maximum sequence length that the model
            processes. If the sequence length is larger than this number,
            the sequence is split into multiple segments. If it is None,
            the entire sequence is processed at once.

    Returns:
        Tensor: The predicted restoration result.
    """

    device = next(model.parameters()).device  # model device

    # build the data pipeline
    if model.cfg.get('demo_pipeline', None):
        demo_pipeline = model.cfg.demo_pipeline
    elif model.cfg.get('demo_pipeline', None):
        demo_pipeline = model.cfg.demo_pipeline
    else:
        demo_pipeline = model.cfg.val_pipeline

    # check if the input is a video
    file_extension = osp.splitext(img_dir)[1]
    if file_extension in VIDEO_EXTENSIONS:
        video_reader = mmcv.VideoReader(img_dir)
        # load the images
        data = dict(lq=[], lq_path=None, key=img_dir)
        for i,frame in enumerate(video_reader):
            data['lq'].append(np.flip(frame, axis=2))
            if i==399:
                break

        # remove the data loading pipeline
        tmp_pipeline = []
        for pipeline in demo_pipeline:
            if pipeline['type'] not in [
                    'GenerateSegmentIndices', 'LoadImageFromFileList'
            ]:
                tmp_pipeline.append(pipeline)
        demo_pipeline = tmp_pipeline
    else:
        # the first element in the pipeline must be 'GenerateSegmentIndices'
        if demo_pipeline[0]['type'] != 'GenerateSegmentIndices':
            raise TypeError('The first element in the pipeline must be '
                            f'"GenerateSegmentIndices", but got '
                            f'"{demo_pipeline[0]["type"]}".')

        # specify start_idx and filename_tmpl
        demo_pipeline[0]['start_idx'] = start_idx
        demo_pipeline[0]['filename_tmpl'] = filename_tmpl

        # prepare data
        sequence_length = len(glob.glob(osp.join(img_dir, '*')))
        lq_folder = osp.dirname(img_dir)
        key = osp.basename(img_dir)
        data = dict(
            lq_path=lq_folder,
            gt_path='',
            key=key,
            sequence_length=sequence_length)

    # compose the pipeline
    demo_pipeline = Compose(demo_pipeline)
    data = demo_pipeline(data)
    data = data['lq'].unsqueeze(0)  # in cpu
    print(data.shape)
    # forward the model
    with torch.no_grad():
        if window_size > 0:  # sliding window framework
            data = pad_sequence(data, window_size)
            result = []
            for i in range(window_size//2, data.size(1) - 2 * (window_size // 2)*3):
                data_i = data[:, i-window_size//2:i + window_size//2*3].to(device)
                
                result.append(model(lq=data_i, test_mode=True)['output'].cpu()[:,window_size//2:window_size//2*3])
            result = torch.stack(result, dim=1)
        else:  # recurrent framework
            if max_seq_len is None:
                result = model(
                    lq=data.to(device), test_mode=True)['output'].cpu()
            else:
                result = []
                for i in range(0, data.size(1), max_seq_len):
                    result.append(
                        model(
                            lq=torch.cat((data[:, i:i + max_seq_len],torch.flip(data[:, i:i + max_seq_len],dims=[1])),dim=1).to(device),
                            test_mode=True)['output'].cpu()[:,:max_seq_len])
                # for i in range((max_seq_len // 2), data.size(1)- (max_seq_len // 2)*3+1, max_seq_len):
                #     result.append(
                #         model(
                #             lq=torch.cat((data[:, i-(max_seq_len // 2):i +  (max_seq_len // 2)*3],torch.flip(data[:, i-(max_seq_len // 2):i +  (max_seq_len // 2)*3],dims=[1])),dim=1).to(device),
                #             test_mode=True)['output'].cpu()[:,(max_seq_len // 2):(max_seq_len // 2)*3])
                result = torch.cat(result, dim=1)
    return result
