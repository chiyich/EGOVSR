
import torch.nn.functional as F
from mmcv.parallel import is_module_wrapper

from ..builder import build_loss
from ..common import set_requires_grad
from ..registry import MODELS
from .real_esrgan import RealESRGAN
from mmedit.core import tensor2img
import numbers
import os.path as osp
import mmcv
import torch
import cv2
import numpy as np

@MODELS.register_module()
class RealEGOVSR(RealESRGAN):
    """EGOVSR model for real-world egocentric video super-resolution.


    Args:
        generator (dict): Config for the generator.
        discriminator (dict, optional): Config for the discriminator.
            Default: None.
        gan_loss (dict, optional): Config for the gan loss.
            Note that the loss weight in gan loss is only for the generator.
        pixel_loss (dict, optional): Config for the pixel loss. Default: None.
        cleaning_loss (dict, optional): Config for the image cleaning loss.
            Default: None.
        perceptual_loss (dict, optional): Config for the perceptual loss.
            Default: None.
        is_use_sharpened_gt_in_pixel (bool, optional): Whether to use the image
            sharpened by unsharp masking as the GT for pixel loss.
            Default: False.
        is_use_sharpened_gt_in_percep (bool, optional): Whether to use the
            image sharpened by unsharp masking as the GT for perceptual loss.
            Default: False.
        is_use_sharpened_gt_in_gan (bool, optional): Whether to use the
            image sharpened by unsharp masking as the GT for adversarial loss.
            Default: False.
        is_use_ema (bool, optional): When to apply exponential moving average
            on the network weights. Default: True.
        train_cfg (dict): Config for training. Default: None.
            You may change the training of gan by setting:
            `disc_steps`: how many discriminator updates after one generate
            update;
            `disc_init_steps`: how many discriminator updates at the start of
            the training.
            These two keys are useful when training with WGAN.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 discriminator=None,
                 gan_loss=None,
                 pixel_loss=None,
                 cleaning_loss=None,
                 mask_loss=None,
                 perceptual_loss=None,
                 is_use_sharpened_gt_in_pixel=False,
                 is_use_sharpened_gt_in_percep=False,
                 is_use_sharpened_gt_in_gan=False,
                 is_use_ema=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super().__init__(generator, discriminator, gan_loss, pixel_loss,
                         perceptual_loss, is_use_sharpened_gt_in_pixel,
                         is_use_sharpened_gt_in_percep,
                         is_use_sharpened_gt_in_gan, is_use_ema, train_cfg,
                         test_cfg, pretrained)

        self.cleaning_loss = build_loss(
            cleaning_loss) if cleaning_loss else None
        self.mask_loss = build_loss(
            mask_loss) if mask_loss else None

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """

        # during initialization, load weights from the ema model
        if (self.step_counter == self.start_iter
                and self.generator_ema is not None):
            if is_module_wrapper(self.generator):
                self.generator.module.load_state_dict(
                    self.generator_ema.module.state_dict())
            else:
                self.generator.load_state_dict(self.generator_ema.state_dict())

        # data
        lq = data_batch['lq']
        gt = data_batch['gt']
        lq_clean = data_batch['lq_clean']

        gt_pixel, gt_percep, gt_gan = gt.clone(), gt.clone(), gt.clone()
        if self.is_use_sharpened_gt_in_pixel:
            gt_pixel = data_batch['gt_unsharp']
        if self.is_use_sharpened_gt_in_percep:
            gt_percep = data_batch['gt_unsharp']
        if self.is_use_sharpened_gt_in_gan:
            gt_gan = data_batch['gt_unsharp']

        if self.cleaning_loss:
            n, t, c, h, w = gt.size()
            gt_clean = gt_pixel.view(-1, c, h, w)
            gt_clean = F.interpolate(gt_clean, scale_factor=0.25, mode='area')
            gt_clean = gt_clean.view(n, t, c, h // 4, w // 4)
            lq_clean = lq_clean.view(-1, c, h, w)
            lq_clean = F.interpolate(lq_clean, scale_factor=0.25, mode='area')
            lq_clean = lq_clean.view(n, t, c, h // 4, w // 4)
            mask = torch.cat([gt_clean, lq_clean],dim=1)
            mask = mask.view(n* t*2, c, h // 4, w // 4)           

        else:
            mask = None

        # generator
        (fake_g_output, mask_clean), fake_g_lq = self.generator(lq, return_lqs=True, gt_clean=mask)
        mask_clean = mask_clean.reshape(n, t*2, 1, h // 4, w // 4)
        mask_clean_gt = mask_clean[:,:t,:,:,:].detach()
        mask_clean_lq = mask_clean[:,t:,:,:,:]
        losses = dict()
        log_vars = dict()

        # reshape: (n, t, c, h, w) -> (n*t, c, h, w)
        c, h, w = gt.shape[2:]
        gt_pixel = gt_pixel.view(-1, c, h, w)
        gt_percep = gt_percep.view(-1, c, h, w)
        gt_gan = gt_gan.view(-1, c, h, w)
        fake_g_output = fake_g_output.view(-1, c, h, w)

        # no updates to discriminator parameters
        if self.gan_loss:
            set_requires_grad(self.discriminator, False)

        if (self.step_counter % self.disc_steps == 0
                and self.step_counter >= self.disc_init_steps):
            if self.pixel_loss:
                losses['loss_pix'] = self.pixel_loss(fake_g_output, gt_pixel)
            if self.cleaning_loss:
                losses['loss_clean'] = self.cleaning_loss(fake_g_lq, lq_clean)
            if self.mask_loss:
                losses['loss_mask'] = self.mask_loss(mask_clean_lq, gt_clean, lq_clean)
            if self.perceptual_loss:
                loss_percep, loss_style = self.perceptual_loss(
                    fake_g_output, gt_percep)
                if loss_percep is not None:
                    losses['loss_perceptual'] = loss_percep
                if loss_style is not None:
                    losses['loss_style'] = loss_style

            # gan loss for generator
            if self.gan_loss:
                fake_g_pred = self.discriminator(fake_g_output)
                losses['loss_gan'] = self.gan_loss(
                    fake_g_pred, target_is_real=True, is_disc=False)

            # parse loss
            loss_g, log_vars_g = self.parse_losses(losses)
            log_vars.update(log_vars_g)

            # optimize
            optimizer['generator'].zero_grad()
            loss_g.backward()
            optimizer['generator'].step()

        # discriminator
        if self.gan_loss:
            set_requires_grad(self.discriminator, True)
            # real
            real_d_pred = self.discriminator(gt_gan)
            loss_d_real = self.gan_loss(
                real_d_pred, target_is_real=True, is_disc=True)
            loss_d, log_vars_d = self.parse_losses(
                dict(loss_d_real=loss_d_real))
            optimizer['discriminator'].zero_grad()
            loss_d.backward()
            log_vars.update(log_vars_d)

            # fake
            fake_d_pred = self.discriminator(fake_g_output.detach())
            loss_d_fake = self.gan_loss(
                fake_d_pred, target_is_real=False, is_disc=True)
            loss_d, log_vars_d = self.parse_losses(
                dict(loss_d_fake=loss_d_fake))
            loss_d.backward()
            log_vars.update(log_vars_d)

            optimizer['discriminator'].step()

        self.step_counter += 1

        log_vars.pop('loss')  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=fake_g_output.cpu()))

        return outputs
    
    def evaluate(self, output, gt):
        """Evaluation function.

        If the output contains multiple frames, we compute the metric
        one by one and take an average.

        Args:
            output (Tensor): Model output with shape (n, t, c, h, w).
            gt (Tensor): GT Tensor with shape (n, t, c, h, w).

        Returns:
            dict: Evaluation results.
        """
        crop_border = self.test_cfg.crop_border
        convert_to = self.test_cfg.get('convert_to', None)

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            if output.ndim == 5:  # a sequence: (n, t, c, h, w)
                avg = []
                for i in range(0, output.size(1)):
                    output_i = tensor2img(output[:, i, :, :, :])
                    gt_i = tensor2img(gt[:, i, :, :, :])
                    if metric != 'NIQE':
                        avg.append(self.allowed_metrics[metric](
                            output_i, gt_i, crop_border=crop_border, convert_to=convert_to))
                    else:
                        avg.append(self.allowed_metrics[metric](
                            output_i, crop_border=crop_border, convert_to=convert_to))              
                eval_result[metric] = np.mean(avg)
            elif output.ndim == 4:  # an image: (n, c, t, w), for Vimeo-90K-T
                output_img = tensor2img(output)
                gt_img = tensor2img(gt)
                if metric != 'NIQE':
                    value = self.allowed_metrics[metric](
                        output_img, gt_img, crop_border=crop_border, convert_to=convert_to)
                else:
                    value = self.allowed_metrics[metric](
                        output_img, crop_border=crop_border, convert_to=convert_to)
                eval_result[metric] = value

        return eval_result
    

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                is_mirror_extended = True

        return is_mirror_extended
    
    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        _model = self.generator_ema if self.is_use_ema else self.generator
        #h, w = gt.size()[-2:]
        n, t, c, _, _ = lq.size()

        if self.check_if_mirror_extended(lq):
            mask_in = t//4
        else:
            mask_in = t//2
        output, mask = _model(lq, gt_clean=lq[:,mask_in])
        #mask = (mask-mask.min())/(mask.max()-mask.min())
        if gt is not None and gt.ndim == 4:
            t = output.size(1)
            if self.check_if_mirror_extended(lq):  # with mirror extension
                output = 0.5 * (output[:, t // 4] + output[:, -1 - t // 4])
            else:  # without mirror extension
                output = output[:, t // 2]

        if self.test_cfg is not None and self.test_cfg.get(
                'metrics', None) and gt is not None:
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())

        # save image
        if save_image:
            if output.ndim == 4:  # an image, key = 000001/0000 (Vimeo-90K)
                img_name = meta[0]['key'].replace('/', '_')
                if isinstance(iteration, numbers.Number):
                    save_path_mask = osp.join(
                        osp.join(save_path,'mask'), f'{img_name}-{iteration + 1:06d}mask.png')
                    save_path = osp.join(
                        save_path, f'{img_name}-{iteration + 1:06d}.png')
                elif iteration is None:
                    save_path_mask = osp.join(
                        osp.join(save_path,'mask'), f'{img_name}mask.png')
                    save_path = osp.join(save_path, f'{img_name}.png')
                else:
                    raise ValueError('iteration should be number or None, '
                                     f'but got {type(iteration)}')
                mmcv.imwrite(tensor2img(output), save_path)
                mask_heat=cv2.applyColorMap(tensor2img(mask), cv2.COLORMAP_JET)
                mmcv.imwrite(mask_heat, save_path_mask)
            elif output.ndim == 5:  # a sequence, key = 000
                folder_name = meta[0]['key'].split('/')[0]
                for i in range(0, output.size(1)):
                    if isinstance(iteration, numbers.Number):
                        save_path_i = osp.join(
                            save_path, folder_name,
                            f'{i:08d}-{iteration + 1:06d}.png')
                    elif iteration is None:
                        save_path_i = osp.join(save_path, folder_name,
                                               f'{i:08d}.png')
                    else:
                        raise ValueError('iteration should be number or None, '
                                         f'but got {type(iteration)}')
                    mmcv.imwrite(
                        tensor2img(output[:, i, :, :, :]), save_path_i)

        return results
