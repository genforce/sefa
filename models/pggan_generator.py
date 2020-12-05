# python3.7
"""Contains the implementation of generator described in PGGAN.

Paper: https://arxiv.org/pdf/1710.10196.pdf

Official TensorFlow implementation:
https://github.com/tkarras/progressive_growing_of_gans
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['PGGANGenerator']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Initial resolution.
_INIT_RES = 4

# Default gain factor for weight scaling.
_WSCALE_GAIN = np.sqrt(2.0)


class PGGANGenerator(nn.Module):
    """Defines the generator network in PGGAN.

    NOTE: The synthesized images are with `RGB` channel order and pixel range
    [-1, 1].

    Settings for the network:

    (1) resolution: The resolution of the output image.
    (2) z_space_dim: The dimension of the latent space, Z. (default: 512)
    (3) image_channels: Number of channels of the output image. (default: 3)
    (4) final_tanh: Whether to use `tanh` to control the final pixel range.
        (default: False)
    (5) label_size: Size of the additional label for conditional generation.
        (default: 0)
    (6) fused_scale: Whether to fused `upsample` and `conv2d` together,
        resulting in `conv2d_transpose`. (default: False)
    (7) use_wscale: Whether to use weight scaling. (default: True)
    (8) fmaps_base: Factor to control number of feature maps for each layer.
        (default: 16 << 10)
    (9) fmaps_max: Maximum number of feature maps in each layer. (default: 512)
    """

    def __init__(self,
                 resolution,
                 z_space_dim=512,
                 image_channels=3,
                 final_tanh=False,
                 label_size=0,
                 fused_scale=False,
                 use_wscale=True,
                 fmaps_base=16 << 10,
                 fmaps_max=512):
        """Initializes with basic settings.

        Raises:
            ValueError: If the `resolution` is not supported.
        """
        super().__init__()

        if resolution not in _RESOLUTIONS_ALLOWED:
            raise ValueError(f'Invalid resolution: `{resolution}`!\n'
                             f'Resolutions allowed: {_RESOLUTIONS_ALLOWED}.')

        self.init_res = _INIT_RES
        self.init_res_log2 = int(np.log2(self.init_res))
        self.resolution = resolution
        self.final_res_log2 = int(np.log2(self.resolution))
        self.z_space_dim = z_space_dim
        self.image_channels = image_channels
        self.final_tanh = final_tanh
        self.label_size = label_size
        self.fused_scale = fused_scale
        self.use_wscale = use_wscale
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max

        # Number of convolutional layers.
        self.num_layers = (self.final_res_log2 - self.init_res_log2 + 1) * 2

        # Level of detail (used for progressive training).
        self.register_buffer('lod', torch.zeros(()))
        self.pth_to_tf_var_mapping = {'lod': 'lod'}

        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            res = 2 ** res_log2
            block_idx = res_log2 - self.init_res_log2

            # First convolution layer for each resolution.
            if res == self.init_res:
                self.add_module(
                    f'layer{2 * block_idx}',
                    ConvBlock(in_channels=self.z_space_dim + self.label_size,
                              out_channels=self.get_nf(res),
                              kernel_size=self.init_res,
                              padding=self.init_res - 1,
                              use_wscale=self.use_wscale))
                tf_layer_name = 'Dense'
            else:
                self.add_module(
                    f'layer{2 * block_idx}',
                    ConvBlock(in_channels=self.get_nf(res // 2),
                              out_channels=self.get_nf(res),
                              upsample=True,
                              fused_scale=self.fused_scale,
                              use_wscale=self.use_wscale))
                tf_layer_name = 'Conv0_up' if self.fused_scale else 'Conv0'
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.weight'] = (
                f'{res}x{res}/{tf_layer_name}/weight')
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.bias'] = (
                f'{res}x{res}/{tf_layer_name}/bias')

            # Second convolution layer for each resolution.
            self.add_module(
                f'layer{2 * block_idx + 1}',
                ConvBlock(in_channels=self.get_nf(res),
                          out_channels=self.get_nf(res),
                          use_wscale=self.use_wscale))
            tf_layer_name = 'Conv' if res == self.init_res else 'Conv1'
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.weight'] = (
                f'{res}x{res}/{tf_layer_name}/weight')
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.bias'] = (
                f'{res}x{res}/{tf_layer_name}/bias')

            # Output convolution layer for each resolution.
            self.add_module(
                f'output{block_idx}',
                ConvBlock(in_channels=self.get_nf(res),
                          out_channels=self.image_channels,
                          kernel_size=1,
                          padding=0,
                          use_wscale=self.use_wscale,
                          wscale_gain=1.0,
                          activation_type='linear'))
            self.pth_to_tf_var_mapping[f'output{block_idx}.weight'] = (
                f'ToRGB_lod{self.final_res_log2 - res_log2}/weight')
            self.pth_to_tf_var_mapping[f'output{block_idx}.bias'] = (
                f'ToRGB_lod{self.final_res_log2 - res_log2}/bias')

        self.upsample = UpsamplingLayer()
        self.final_activate = nn.Tanh() if self.final_tanh else nn.Identity()

    def get_nf(self, res):
        """Gets number of feature maps according to current resolution."""
        return min(self.fmaps_base // res, self.fmaps_max)

    def forward(self, z, label=None, lod=None, **_unused_kwargs):
        if z.ndim != 2 or z.shape[1] != self.z_space_dim:
            raise ValueError(f'Input latent code should be with shape '
                             f'[batch_size, latent_dim], where '
                             f'`latent_dim` equals to {self.z_space_dim}!\n'
                             f'But `{z.shape}` is received!')
        z = self.layer0.pixel_norm(z)
        if self.label_size:
            if label is None:
                raise ValueError(f'Model requires an additional label '
                                 f'(with size {self.label_size}) as input, '
                                 f'but no label is received!')
            if label.ndim != 2 or label.shape != (z.shape[0], self.label_size):
                raise ValueError(f'Input label should be with shape '
                                 f'[batch_size, label_size], where '
                                 f'`batch_size` equals to that of '
                                 f'latent codes ({z.shape[0]}) and '
                                 f'`label_size` equals to {self.label_size}!\n'
                                 f'But `{label.shape}` is received!')
            z = torch.cat((z, label), dim=1)

        lod = self.lod.cpu().tolist() if lod is None else lod
        if lod + self.init_res_log2 > self.final_res_log2:
            raise ValueError(f'Maximum level-of-detail (lod) is '
                             f'{self.final_res_log2 - self.init_res_log2}, '
                             f'but `{lod}` is received!')

        x = z.view(z.shape[0], self.z_space_dim + self.label_size, 1, 1)
        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            current_lod = self.final_res_log2 - res_log2
            if lod < current_lod + 1:
                block_idx = res_log2 - self.init_res_log2
                x = self.__getattr__(f'layer{2 * block_idx}')(x)
                x = self.__getattr__(f'layer{2 * block_idx + 1}')(x)
            if current_lod - 1 < lod <= current_lod:
                image = self.__getattr__(f'output{block_idx}')(x)
            elif current_lod < lod < current_lod + 1:
                alpha = np.ceil(lod) - lod
                image = (self.__getattr__(f'output{block_idx}')(x) * alpha +
                         self.upsample(image) * (1 - alpha))
            elif lod >= current_lod + 1:
                image = self.upsample(image)
        image = self.final_activate(image)

        results = {
            'z': z,
            'label': label,
            'image': image,
        }
        return results


class PixelNormLayer(nn.Module):
    """Implements pixel-wise feature vector normalization layer."""

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.eps = epsilon

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
        return x / norm


class UpsamplingLayer(nn.Module):
    """Implements the upsampling layer.

    Basically, this layer can be used to upsample feature maps with nearest
    neighbor interpolation.
    """

    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor <= 1:
            return x
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


class ConvBlock(nn.Module):
    """Implements the convolutional block.

    Basically, this block executes pixel-wise normalization layer, upsampling
    layer (if needed), convolutional layer, and activation layer in sequence.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 add_bias=True,
                 upsample=False,
                 fused_scale=False,
                 use_wscale=True,
                 wscale_gain=_WSCALE_GAIN,
                 activation_type='lrelu'):
        """Initializes with block settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            kernel_size: Size of the convolutional kernels. (default: 3)
            stride: Stride parameter for convolution operation. (default: 1)
            padding: Padding parameter for convolution operation. (default: 1)
            add_bias: Whether to add bias onto the convolutional result.
                (default: True)
            upsample: Whether to upsample the input tensor before convolution.
                (default: False)
            fused_scale: Whether to fused `upsample` and `conv2d` together,
                resulting in `conv2d_transpose`. (default: False)
            use_wscale: Whether to use weight scaling. (default: True)
            wscale_gain: Gain factor for weight scaling. (default: _WSCALE_GAIN)
            activation_type: Type of activation. Support `linear` and `lrelu`.
                (default: `lrelu`)

        Raises:
            NotImplementedError: If the `activation_type` is not supported.
        """
        super().__init__()

        self.pixel_norm = PixelNormLayer()

        if upsample and not fused_scale:
            self.upsample = UpsamplingLayer()
        else:
            self.upsample = nn.Identity()

        if upsample and fused_scale:
            self.use_conv2d_transpose = True
            weight_shape = (in_channels, out_channels, kernel_size, kernel_size)
            self.stride = 2
            self.padding = 1
        else:
            self.use_conv2d_transpose = False
            weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
            self.stride = stride
            self.padding = padding

        fan_in = kernel_size * kernel_size * in_channels
        wscale = wscale_gain / np.sqrt(fan_in)
        if use_wscale:
            self.weight = nn.Parameter(torch.randn(*weight_shape))
            self.wscale = wscale
        else:
            self.weight = nn.Parameter(torch.randn(*weight_shape) * wscale)
            self.wscale = 1.0

        if add_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        if activation_type == 'linear':
            self.activate = nn.Identity()
        elif activation_type == 'lrelu':
            self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            raise NotImplementedError(f'Not implemented activation function: '
                                      f'`{activation_type}`!')

    def forward(self, x):
        x = self.pixel_norm(x)
        x = self.upsample(x)
        weight = self.weight * self.wscale
        if self.use_conv2d_transpose:
            weight = F.pad(weight, (1, 1, 1, 1, 0, 0, 0, 0), 'constant', 0.0)
            weight = (weight[:, :, 1:, 1:] + weight[:, :, :-1, 1:] +
                      weight[:, :, 1:, :-1] + weight[:, :, :-1, :-1])
            x = F.conv_transpose2d(x,
                                   weight=weight,
                                   bias=self.bias,
                                   stride=self.stride,
                                   padding=self.padding)
        else:
            x = F.conv2d(x,
                         weight=weight,
                         bias=self.bias,
                         stride=self.stride,
                         padding=self.padding)
        x = self.activate(x)
        return x
