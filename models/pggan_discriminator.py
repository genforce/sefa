# python3.7
"""Contains the implementation of discriminator described in PGGAN.

Paper: https://arxiv.org/pdf/1710.10196.pdf

Official TensorFlow implementation:
https://github.com/tkarras/progressive_growing_of_gans
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['PGGANDiscriminator']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Initial resolution.
_INIT_RES = 4

# Default gain factor for weight scaling.
_WSCALE_GAIN = np.sqrt(2.0)


class PGGANDiscriminator(nn.Module):
    """Defines the discriminator network in PGGAN.

    NOTE: The discriminator takes images with `RGB` channel order and pixel
    range [-1, 1] as inputs.

    Settings for the network:

    (1) resolution: The resolution of the input image.
    (2) image_channels: Number of channels of the input image. (default: 3)
    (3) label_size: Size of the additional label for conditional generation.
        (default: 0)
    (4) fused_scale: Whether to fused `conv2d` and `downsample` together,
        resulting in `conv2d` with strides. (default: False)
    (5) use_wscale: Whether to use weight scaling. (default: True)
    (6) minibatch_std_group_size: Group size for the minibatch standard
        deviation layer. 0 means disable. (default: 16)
    (7) fmaps_base: Factor to control number of feature maps for each layer.
        (default: 16 << 10)
    (8) fmaps_max: Maximum number of feature maps in each layer. (default: 512)
    """

    def __init__(self,
                 resolution,
                 image_channels=3,
                 label_size=0,
                 fused_scale=False,
                 use_wscale=True,
                 minibatch_std_group_size=16,
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
        self.image_channels = image_channels
        self.label_size = label_size
        self.fused_scale = fused_scale
        self.use_wscale = use_wscale
        self.minibatch_std_group_size = minibatch_std_group_size
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max

        # Level of detail (used for progressive training).
        self.register_buffer('lod', torch.zeros(()))
        self.pth_to_tf_var_mapping = {'lod': 'lod'}

        for res_log2 in range(self.final_res_log2, self.init_res_log2 - 1, -1):
            res = 2 ** res_log2
            block_idx = self.final_res_log2 - res_log2

            # Input convolution layer for each resolution.
            self.add_module(
                f'input{block_idx}',
                ConvBlock(in_channels=self.image_channels,
                          out_channels=self.get_nf(res),
                          kernel_size=1,
                          padding=0,
                          use_wscale=self.use_wscale))
            self.pth_to_tf_var_mapping[f'input{block_idx}.weight'] = (
                f'FromRGB_lod{block_idx}/weight')
            self.pth_to_tf_var_mapping[f'input{block_idx}.bias'] = (
                f'FromRGB_lod{block_idx}/bias')

            # Convolution block for each resolution (except the last one).
            if res != self.init_res:
                self.add_module(
                    f'layer{2 * block_idx}',
                    ConvBlock(in_channels=self.get_nf(res),
                              out_channels=self.get_nf(res),
                              use_wscale=self.use_wscale))
                tf_layer0_name = 'Conv0'
                self.add_module(
                    f'layer{2 * block_idx + 1}',
                    ConvBlock(in_channels=self.get_nf(res),
                              out_channels=self.get_nf(res // 2),
                              downsample=True,
                              fused_scale=self.fused_scale,
                              use_wscale=self.use_wscale))
                tf_layer1_name = 'Conv1_down' if self.fused_scale else 'Conv1'

            # Convolution block for last resolution.
            else:
                self.add_module(
                    f'layer{2 * block_idx}',
                    ConvBlock(
                        in_channels=self.get_nf(res),
                        out_channels=self.get_nf(res),
                        use_wscale=self.use_wscale,
                        minibatch_std_group_size=self.minibatch_std_group_size))
                tf_layer0_name = 'Conv'
                self.add_module(
                    f'layer{2 * block_idx + 1}',
                    DenseBlock(in_channels=self.get_nf(res) * res * res,
                               out_channels=self.get_nf(res // 2),
                               use_wscale=self.use_wscale))
                tf_layer1_name = 'Dense0'

            self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.weight'] = (
                f'{res}x{res}/{tf_layer0_name}/weight')
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.bias'] = (
                f'{res}x{res}/{tf_layer0_name}/bias')
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.weight'] = (
                f'{res}x{res}/{tf_layer1_name}/weight')
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.bias'] = (
                f'{res}x{res}/{tf_layer1_name}/bias')

        # Final dense block.
        self.add_module(
            f'layer{2 * block_idx + 2}',
            DenseBlock(in_channels=self.get_nf(res // 2),
                       out_channels=1 + self.label_size,
                       use_wscale=self.use_wscale,
                       wscale_gain=1.0,
                       activation_type='linear'))
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 2}.weight'] = (
            f'{res}x{res}/Dense1/weight')
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 2}.bias'] = (
            f'{res}x{res}/Dense1/bias')

        self.downsample = DownsamplingLayer()

    def get_nf(self, res):
        """Gets number of feature maps according to current resolution."""
        return min(self.fmaps_base // res, self.fmaps_max)

    def forward(self, image, lod=None, **_unused_kwargs):
        expected_shape = (self.image_channels, self.resolution, self.resolution)
        if image.ndim != 4 or image.shape[1:] != expected_shape:
            raise ValueError(f'The input tensor should be with shape '
                             f'[batch_size, channel, height, width], where '
                             f'`channel` equals to {self.image_channels}, '
                             f'`height`, `width` equal to {self.resolution}!\n'
                             f'But `{image.shape}` is received!')

        lod = self.lod.cpu().tolist() if lod is None else lod
        if lod + self.init_res_log2 > self.final_res_log2:
            raise ValueError(f'Maximum level-of-detail (lod) is '
                             f'{self.final_res_log2 - self.init_res_log2}, '
                             f'but `{lod}` is received!')

        lod = self.lod.cpu().tolist()
        for res_log2 in range(self.final_res_log2, self.init_res_log2 - 1, -1):
            block_idx = current_lod = self.final_res_log2 - res_log2
            if current_lod <= lod < current_lod + 1:
                x = self.__getattr__(f'input{block_idx}')(image)
            elif current_lod - 1 < lod < current_lod:
                alpha = lod - np.floor(lod)
                x = (self.__getattr__(f'input{block_idx}')(image) * alpha +
                     x * (1 - alpha))
            if lod < current_lod + 1:
                x = self.__getattr__(f'layer{2 * block_idx}')(x)
                x = self.__getattr__(f'layer{2 * block_idx + 1}')(x)
            if lod > current_lod:
                image = self.downsample(image)
        x = self.__getattr__(f'layer{2 * block_idx + 2}')(x)
        return x


class MiniBatchSTDLayer(nn.Module):
    """Implements the minibatch standard deviation layer."""

    def __init__(self, group_size=16, epsilon=1e-8):
        super().__init__()
        self.group_size = group_size
        self.epsilon = epsilon

    def forward(self, x):
        if self.group_size <= 1:
            return x
        group_size = min(self.group_size, x.shape[0])                  # [NCHW]
        y = x.view(group_size, -1, x.shape[1], x.shape[2], x.shape[3]) # [GMCHW]
        y = y - torch.mean(y, dim=0, keepdim=True)                     # [GMCHW]
        y = torch.mean(y ** 2, dim=0)                                  # [MCHW]
        y = torch.sqrt(y + self.epsilon)                               # [MCHW]
        y = torch.mean(y, dim=[1, 2, 3], keepdim=True)                 # [M111]
        y = y.repeat(group_size, 1, x.shape[2], x.shape[3])            # [N1HW]
        return torch.cat([x, y], dim=1)


class DownsamplingLayer(nn.Module):
    """Implements the downsampling layer.

    Basically, this layer can be used to downsample feature maps with average
    pooling.
    """

    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor <= 1:
            return x
        return F.avg_pool2d(x,
                            kernel_size=self.scale_factor,
                            stride=self.scale_factor,
                            padding=0)


class ConvBlock(nn.Module):
    """Implements the convolutional block.

    Basically, this block executes minibatch standard deviation layer (if
    needed), convolutional layer, activation layer, and downsampling layer (
    if needed) in sequence.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 add_bias=True,
                 downsample=False,
                 fused_scale=False,
                 use_wscale=True,
                 wscale_gain=_WSCALE_GAIN,
                 activation_type='lrelu',
                 minibatch_std_group_size=0):
        """Initializes with block settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            kernel_size: Size of the convolutional kernels. (default: 3)
            stride: Stride parameter for convolution operation. (default: 1)
            padding: Padding parameter for convolution operation. (default: 1)
            add_bias: Whether to add bias onto the convolutional result.
                (default: True)
            downsample: Whether to downsample the result after convolution.
                (default: False)
            fused_scale: Whether to fused `conv2d` and `downsample` together,
                resulting in `conv2d` with strides. (default: False)
            use_wscale: Whether to use weight scaling. (default: True)
            wscale_gain: Gain factor for weight scaling. (default: _WSCALE_GAIN)
            activation_type: Type of activation. Support `linear` and `lrelu`.
                (default: `lrelu`)
            minibatch_std_group_size: Group size for the minibatch standard
                deviation layer. 0 means disable. (default: 0)

        Raises:
            NotImplementedError: If the `activation_type` is not supported.
        """
        super().__init__()

        if minibatch_std_group_size > 1:
            in_channels = in_channels + 1
            self.mbstd = MiniBatchSTDLayer(group_size=minibatch_std_group_size)
        else:
            self.mbstd = nn.Identity()

        if downsample and not fused_scale:
            self.downsample = DownsamplingLayer()
        else:
            self.downsample = nn.Identity()

        if downsample and fused_scale:
            self.use_stride = True
            self.stride = 2
            self.padding = 1
        else:
            self.use_stride = False
            self.stride = stride
            self.padding = padding

        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
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
        x = self.mbstd(x)
        weight = self.weight * self.wscale
        if self.use_stride:
            weight = F.pad(weight, (1, 1, 1, 1, 0, 0, 0, 0), 'constant', 0.0)
            weight = (weight[:, :, 1:, 1:] + weight[:, :, :-1, 1:] +
                      weight[:, :, 1:, :-1] + weight[:, :, :-1, :-1]) * 0.25
        x = F.conv2d(x,
                     weight=weight,
                     bias=self.bias,
                     stride=self.stride,
                     padding=self.padding)
        x = self.activate(x)
        x = self.downsample(x)
        return x


class DenseBlock(nn.Module):
    """Implements the dense block.

    Basically, this block executes fully-connected layer, and activation layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 add_bias=True,
                 use_wscale=True,
                 wscale_gain=_WSCALE_GAIN,
                 activation_type='lrelu'):
        """Initializes with block settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            add_bias: Whether to add bias onto the fully-connected result.
                (default: True)
            use_wscale: Whether to use weight scaling. (default: True)
            wscale_gain: Gain factor for weight scaling. (default: _WSCALE_GAIN)
            activation_type: Type of activation. Support `linear` and `lrelu`.
                (default: `lrelu`)

        Raises:
            NotImplementedError: If the `activation_type` is not supported.
        """
        super().__init__()
        weight_shape = (out_channels, in_channels)
        wscale = wscale_gain / np.sqrt(in_channels)
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
        if x.ndim != 2:
            x = x.view(x.shape[0], -1)
        x = F.linear(x, weight=self.weight * self.wscale, bias=self.bias)
        x = self.activate(x)
        return x
