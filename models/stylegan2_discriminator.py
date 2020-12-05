# python3.7
"""Contains the implementation of discriminator described in StyleGAN2.

Compared to that of StyleGAN, the discriminator in StyleGAN2 mainly adds skip
connections, increases model size and disables progressive growth. This script
ONLY supports config F in the original paper.

Paper: https://arxiv.org/pdf/1912.04958.pdf

Official TensorFlow implementation: https://github.com/NVlabs/stylegan2
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['StyleGAN2Discriminator']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Initial resolution.
_INIT_RES = 4

# Architectures allowed.
_ARCHITECTURES_ALLOWED = ['resnet', 'skip', 'origin']

# Default gain factor for weight scaling.
_WSCALE_GAIN = 1.0


class StyleGAN2Discriminator(nn.Module):
    """Defines the discriminator network in StyleGAN2.

    NOTE: The discriminator takes images with `RGB` channel order and pixel
    range [-1, 1] as inputs.

    Settings for the network:

    (1) resolution: The resolution of the input image.
    (2) image_channels: Number of channels of the input image. (default: 3)
    (3) label_size: Size of the additional label for conditional generation.
        (default: 0)
    (4) architecture: Type of architecture. Support `origin`, `skip`, and
        `resnet`. (default: `resnet`)
    (5) use_wscale: Whether to use weight scaling. (default: True)
    (6) minibatch_std_group_size: Group size for the minibatch standard
        deviation layer. 0 means disable. (default: 4)
    (7) minibatch_std_channels: Number of new channels after the minibatch
        standard deviation layer. (default: 1)
    (8) fmaps_base: Factor to control number of feature maps for each layer.
        (default: 32 << 10)
    (9) fmaps_max: Maximum number of feature maps in each layer. (default: 512)
    """

    def __init__(self,
                 resolution,
                 image_channels=3,
                 label_size=0,
                 architecture='resnet',
                 use_wscale=True,
                 minibatch_std_group_size=4,
                 minibatch_std_channels=1,
                 fmaps_base=32 << 10,
                 fmaps_max=512):
        """Initializes with basic settings.

        Raises:
            ValueError: If the `resolution` is not supported, or `architecture`
                is not supported.
        """
        super().__init__()

        if resolution not in _RESOLUTIONS_ALLOWED:
            raise ValueError(f'Invalid resolution: `{resolution}`!\n'
                             f'Resolutions allowed: {_RESOLUTIONS_ALLOWED}.')
        if architecture not in _ARCHITECTURES_ALLOWED:
            raise ValueError(f'Invalid architecture: `{architecture}`!\n'
                             f'Architectures allowed: '
                             f'{_ARCHITECTURES_ALLOWED}.')

        self.init_res = _INIT_RES
        self.init_res_log2 = int(np.log2(self.init_res))
        self.resolution = resolution
        self.final_res_log2 = int(np.log2(self.resolution))
        self.image_channels = image_channels
        self.label_size = label_size
        self.architecture = architecture
        self.use_wscale = use_wscale
        self.minibatch_std_group_size = minibatch_std_group_size
        self.minibatch_std_channels = minibatch_std_channels
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max

        self.pth_to_tf_var_mapping = {}
        for res_log2 in range(self.final_res_log2, self.init_res_log2 - 1, -1):
            res = 2 ** res_log2
            block_idx = self.final_res_log2 - res_log2

            # Input convolution layer for each resolution (if needed).
            if res_log2 == self.final_res_log2 or self.architecture == 'skip':
                self.add_module(
                    f'input{block_idx}',
                    ConvBlock(in_channels=self.image_channels,
                              out_channels=self.get_nf(res),
                              kernel_size=1,
                              use_wscale=self.use_wscale))
                self.pth_to_tf_var_mapping[f'input{block_idx}.weight'] = (
                    f'{res}x{res}/FromRGB/weight')
                self.pth_to_tf_var_mapping[f'input{block_idx}.bias'] = (
                    f'{res}x{res}/FromRGB/bias')

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
                              scale_factor=2,
                              use_wscale=self.use_wscale))
                tf_layer1_name = 'Conv1_down'

                if self.architecture == 'resnet':
                    layer_name = f'skip_layer{block_idx}'
                    self.add_module(
                        layer_name,
                        ConvBlock(in_channels=self.get_nf(res),
                                  out_channels=self.get_nf(res // 2),
                                  kernel_size=1,
                                  add_bias=False,
                                  scale_factor=2,
                                  use_wscale=self.use_wscale,
                                  activation_type='linear'))
                    self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                        f'{res}x{res}/Skip/weight')

            # Convolution block for last resolution.
            else:
                self.add_module(
                    f'layer{2 * block_idx}',
                    ConvBlock(in_channels=self.get_nf(res),
                              out_channels=self.get_nf(res),
                              use_wscale=self.use_wscale,
                              minibatch_std_group_size=minibatch_std_group_size,
                              minibatch_std_channels=minibatch_std_channels))
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
                           out_channels=max(self.label_size, 1),
                           use_wscale=self.use_wscale,
                           activation_type='linear'))
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 2}.weight'] = (
                f'Output/weight')
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 2}.bias'] = (
                f'Output/bias')

        if self.architecture == 'skip':
            self.downsample = DownsamplingLayer()

    def get_nf(self, res):
        """Gets number of feature maps according to current resolution."""
        return min(self.fmaps_base // res, self.fmaps_max)

    def forward(self, image, label=None, **_unused_kwargs):
        expected_shape = (self.image_channels, self.resolution, self.resolution)
        if image.ndim != 4 or image.shape[1:] != expected_shape:
            raise ValueError(f'The input tensor should be with shape '
                             f'[batch_size, channel, height, width], where '
                             f'`channel` equals to {self.image_channels}, '
                             f'`height`, `width` equal to {self.resolution}!\n'
                             f'But `{image.shape}` is received!')
        if self.label_size:
            if label is None:
                raise ValueError(f'Model requires an additional label '
                                 f'(with size {self.label_size}) as inputs, '
                                 f'but no label is received!')
            batch_size = image.shape[0]
            if label.ndim != 2 or label.shape != (batch_size, self.label_size):
                raise ValueError(f'Input label should be with shape '
                                 f'[batch_size, label_size], where '
                                 f'`batch_size` equals to that of '
                                 f'images ({image.shape[0]}) and '
                                 f'`label_size` equals to {self.label_size}!\n'
                                 f'But `{label.shape}` is received!')

        x = self.input0(image)
        for res_log2 in range(self.final_res_log2, self.init_res_log2 - 1, -1):
            block_idx = self.final_res_log2 - res_log2
            if self.architecture == 'skip' and block_idx > 0:
                image = self.downsample(image)
                x = x + self.__getattr__(f'input{block_idx}')(image)
            if self.architecture == 'resnet' and res_log2 != self.init_res_log2:
                residual = self.__getattr__(f'skip_layer{block_idx}')(x)
            x = self.__getattr__(f'layer{2 * block_idx}')(x)
            x = self.__getattr__(f'layer{2 * block_idx + 1}')(x)
            if self.architecture == 'resnet' and res_log2 != self.init_res_log2:
                x = (x + residual) / np.sqrt(2.0)
        x = self.__getattr__(f'layer{2 * block_idx + 2}')(x)

        if self.label_size:
            x = torch.sum(x * label, dim=1, keepdim=True)
        return x


class MiniBatchSTDLayer(nn.Module):
    """Implements the minibatch standard deviation layer."""

    def __init__(self, group_size=4, new_channels=1, epsilon=1e-8):
        super().__init__()
        self.group_size = group_size
        self.new_channels = new_channels
        self.epsilon = epsilon

    def forward(self, x):
        if self.group_size <= 1:
            return x
        ng = min(self.group_size, x.shape[0])
        nc = self.new_channels
        temp_c = x.shape[1] // nc                               # [NCHW]
        y = x.view(ng, -1, nc, temp_c, x.shape[2], x.shape[3])  # [GMncHW]
        y = y - torch.mean(y, dim=0, keepdim=True)              # [GMncHW]
        y = torch.mean(y ** 2, dim=0)                           # [MncHW]
        y = torch.sqrt(y + self.epsilon)                        # [MncHW]
        y = torch.mean(y, dim=[2, 3, 4], keepdim=True)          # [Mn111]
        y = torch.mean(y, dim=2)                                # [Mn11]
        y = y.repeat(ng, 1, x.shape[2], x.shape[3])             # [NnHW]
        return torch.cat([x, y], dim=1)


class DownsamplingLayer(nn.Module):
    """Implements the downsampling layer.

    This layer can also be used as filtering by setting `scale_factor` as 1.
    """

    def __init__(self, scale_factor=2, kernel=(1, 3, 3, 1), extra_padding=0):
        super().__init__()
        assert scale_factor >= 1
        self.scale_factor = scale_factor

        if extra_padding != 0:
            assert scale_factor == 1

        if kernel is None:
            kernel = np.ones((scale_factor), dtype=np.float32)
        else:
            kernel = np.array(kernel, dtype=np.float32)
        assert kernel.ndim == 1
        kernel = np.outer(kernel, kernel)
        kernel = kernel / np.sum(kernel)
        assert kernel.ndim == 2
        assert kernel.shape[0] == kernel.shape[1]
        kernel = kernel[np.newaxis, np.newaxis]
        self.register_buffer('kernel', torch.from_numpy(kernel))
        self.kernel = self.kernel.flip(0, 1)
        padding = kernel.shape[2] - scale_factor + extra_padding
        self.padding = ((padding + 1) // 2, padding // 2,
                        (padding + 1) // 2, padding // 2)

    def forward(self, x):
        assert x.ndim == 4
        channels = x.shape[1]
        x = x.view(-1, 1, x.shape[2], x.shape[3])
        x = F.pad(x, self.padding, mode='constant', value=0)
        x = F.conv2d(x, self.kernel, stride=self.scale_factor)
        x = x.view(-1, channels, x.shape[2], x.shape[3])
        return x


class ConvBlock(nn.Module):
    """Implements the convolutional block.

    Basically, this block executes minibatch standard deviation layer (if
    needed), filtering layer (if needed), convolutional layer, and activation
    layer in sequence.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 add_bias=True,
                 scale_factor=1,
                 filtering_kernel=(1, 3, 3, 1),
                 use_wscale=True,
                 wscale_gain=_WSCALE_GAIN,
                 lr_mul=1.0,
                 activation_type='lrelu',
                 minibatch_std_group_size=0,
                 minibatch_std_channels=1):
        """Initializes with block settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            kernel_size: Size of the convolutional kernels. (default: 3)
            add_bias: Whether to add bias onto the convolutional result.
                (default: True)
            scale_factor: Scale factor for downsampling. `1` means skip
                downsampling. (default: 1)
            filtering_kernel: Kernel used for filtering before downsampling.
                (default: (1, 3, 3, 1))
            use_wscale: Whether to use weight scaling. (default: True)
            wscale_gain: Gain factor for weight scaling. (default: _WSCALE_GAIN)
            lr_mul: Learning multiplier for both weight and bias. (default: 1.0)
            activation_type: Type of activation. Support `linear` and `lrelu`.
                (default: `lrelu`)
            minibatch_std_group_size: Group size for the minibatch standard
                deviation layer. 0 means disable. (default: 0)
            minibatch_std_channels: Number of new channels after the minibatch
                standard deviation layer. (default: 1)

        Raises:
            NotImplementedError: If the `activation_type` is not supported.
        """
        super().__init__()

        if minibatch_std_group_size > 1:
            in_channels = in_channels + minibatch_std_channels
            self.mbstd = MiniBatchSTDLayer(group_size=minibatch_std_group_size,
                                           new_channels=minibatch_std_channels)
        else:
            self.mbstd = nn.Identity()

        if scale_factor > 1:
            extra_padding = kernel_size - scale_factor
            self.filter = DownsamplingLayer(scale_factor=1,
                                            kernel=filtering_kernel,
                                            extra_padding=extra_padding)
            self.stride = scale_factor
            self.padding = 0  # Padding is done in `DownsamplingLayer`.
        else:
            self.filter = nn.Identity()
            assert kernel_size % 2 == 1
            self.stride = 1
            self.padding = kernel_size // 2

        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        fan_in = kernel_size * kernel_size * in_channels
        wscale = wscale_gain / np.sqrt(fan_in)
        if use_wscale:
            self.weight = nn.Parameter(torch.randn(*weight_shape) / lr_mul)
            self.wscale = wscale * lr_mul
        else:
            self.weight = nn.Parameter(
                torch.randn(*weight_shape) * wscale / lr_mul)
            self.wscale = lr_mul

        if add_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.bscale = lr_mul

        if activation_type == 'linear':
            self.activate = nn.Identity()
            self.activate_scale = 1.0
        elif activation_type == 'lrelu':
            self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.activate_scale = np.sqrt(2.0)
        else:
            raise NotImplementedError(f'Not implemented activation function: '
                                      f'`{activation_type}`!')

    def forward(self, x):
        x = self.mbstd(x)
        x = self.filter(x)
        weight = self.weight * self.wscale
        bias = self.bias * self.bscale if self.bias is not None else None
        x = F.conv2d(x,
                     weight=weight,
                     bias=bias,
                     stride=self.stride,
                     padding=self.padding)
        x = self.activate(x) * self.activate_scale
        return x


class DenseBlock(nn.Module):
    """Implements the dense block.

    Basically, this block executes fully-connected layer and activation layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 add_bias=True,
                 use_wscale=True,
                 wscale_gain=_WSCALE_GAIN,
                 lr_mul=1.0,
                 activation_type='lrelu'):
        """Initializes with block settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            add_bias: Whether to add bias onto the fully-connected result.
                (default: True)
            use_wscale: Whether to use weight scaling. (default: True)
            wscale_gain: Gain factor for weight scaling. (default: _WSCALE_GAIN)
            lr_mul: Learning multiplier for both weight and bias. (default: 1.0)
            activation_type: Type of activation. Support `linear` and `lrelu`.
                (default: `lrelu`)

        Raises:
            NotImplementedError: If the `activation_type` is not supported.
        """
        super().__init__()
        weight_shape = (out_channels, in_channels)
        wscale = wscale_gain / np.sqrt(in_channels)
        if use_wscale:
            self.weight = nn.Parameter(torch.randn(*weight_shape) / lr_mul)
            self.wscale = wscale * lr_mul
        else:
            self.weight = nn.Parameter(
                torch.randn(*weight_shape) * wscale / lr_mul)
            self.wscale = lr_mul

        if add_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.bscale = lr_mul

        if activation_type == 'linear':
            self.activate = nn.Identity()
            self.activate_scale = 1.0
        elif activation_type == 'lrelu':
            self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.activate_scale = np.sqrt(2.0)
        else:
            raise NotImplementedError(f'Not implemented activation function: '
                                      f'`{activation_type}`!')

    def forward(self, x):
        if x.ndim != 2:
            x = x.view(x.shape[0], -1)
        bias = self.bias * self.bscale if self.bias is not None else None
        x = F.linear(x, weight=self.weight * self.wscale, bias=bias)
        x = self.activate(x) * self.activate_scale
        return x
