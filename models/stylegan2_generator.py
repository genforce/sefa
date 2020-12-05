# python3.7
"""Contains the implementation of generator described in StyleGAN2.

Compared to that of StyleGAN, the generator in StyleGAN2 mainly introduces style
demodulation, adds skip connections, increases model size, and disables
progressive growth. This script ONLY supports config F in the original paper.

Paper: https://arxiv.org/pdf/1912.04958.pdf

Official TensorFlow implementation: https://github.com/NVlabs/stylegan2
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sync_op import all_gather

__all__ = ['StyleGAN2Generator']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Initial resolution.
_INIT_RES = 4

# Architectures allowed.
_ARCHITECTURES_ALLOWED = ['resnet', 'skip', 'origin']

# Default gain factor for weight scaling.
_WSCALE_GAIN = 1.0


class StyleGAN2Generator(nn.Module):
    """Defines the generator network in StyleGAN2.

    NOTE: The synthesized images are with `RGB` channel order and pixel range
    [-1, 1].

    Settings for the mapping network:

    (1) z_space_dim: Dimension of the input latent space, Z. (default: 512)
    (2) w_space_dim: Dimension of the outout latent space, W. (default: 512)
    (3) label_size: Size of the additional label for conditional generation.
        (default: 0)
    (4）mapping_layers: Number of layers of the mapping network. (default: 8)
    (5) mapping_fmaps: Number of hidden channels of the mapping network.
        (default: 512)
    (6) mapping_lr_mul: Learning rate multiplier for the mapping network.
        (default: 0.01)
    (7) repeat_w: Repeat w-code for different layers.

    Settings for the synthesis network:

    (1) resolution: The resolution of the output image.
    (2) image_channels: Number of channels of the output image. (default: 3)
    (3) final_tanh: Whether to use `tanh` to control the final pixel range.
        (default: False)
    (4) const_input: Whether to use a constant in the first convolutional layer.
        (default: True)
    (5) architecture: Type of architecture. Support `origin`, `skip`, and
        `resnet`. (default: `resnet`)
    (6) fused_modulate: Whether to fuse `style_modulate` and `conv2d` together.
        (default: True)
    (7) demodulate: Whether to perform style demodulation. (default: True)
    (8) use_wscale: Whether to use weight scaling. (default: True)
    (9) fmaps_base: Factor to control number of feature maps for each layer.
        (default: 16 << 10)
    (10) fmaps_max: Maximum number of feature maps in each layer. (default: 512)
    """

    def __init__(self,
                 resolution,
                 z_space_dim=512,
                 w_space_dim=512,
                 label_size=0,
                 mapping_layers=8,
                 mapping_fmaps=512,
                 mapping_lr_mul=0.01,
                 repeat_w=True,
                 image_channels=3,
                 final_tanh=False,
                 const_input=True,
                 architecture='skip',
                 fused_modulate=True,
                 demodulate=True,
                 use_wscale=True,
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
        self.resolution = resolution
        self.z_space_dim = z_space_dim
        self.w_space_dim = w_space_dim
        self.label_size = label_size
        self.mapping_layers = mapping_layers
        self.mapping_fmaps = mapping_fmaps
        self.mapping_lr_mul = mapping_lr_mul
        self.repeat_w = repeat_w
        self.image_channels = image_channels
        self.final_tanh = final_tanh
        self.const_input = const_input
        self.architecture = architecture
        self.fused_modulate = fused_modulate
        self.demodulate = demodulate
        self.use_wscale = use_wscale
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max

        self.num_layers = int(np.log2(self.resolution // self.init_res * 2)) * 2

        if self.repeat_w:
            self.mapping_space_dim = self.w_space_dim
        else:
            self.mapping_space_dim = self.w_space_dim * self.num_layers
        self.mapping = MappingModule(input_space_dim=self.z_space_dim,
                                     hidden_space_dim=self.mapping_fmaps,
                                     final_space_dim=self.mapping_space_dim,
                                     label_size=self.label_size,
                                     num_layers=self.mapping_layers,
                                     use_wscale=self.use_wscale,
                                     lr_mul=self.mapping_lr_mul)

        self.truncation = TruncationModule(w_space_dim=self.w_space_dim,
                                           num_layers=self.num_layers,
                                           repeat_w=self.repeat_w)

        self.synthesis = SynthesisModule(resolution=self.resolution,
                                         init_resolution=self.init_res,
                                         w_space_dim=self.w_space_dim,
                                         image_channels=self.image_channels,
                                         final_tanh=self.final_tanh,
                                         const_input=self.const_input,
                                         architecture=self.architecture,
                                         fused_modulate=self.fused_modulate,
                                         demodulate=self.demodulate,
                                         use_wscale=self.use_wscale,
                                         fmaps_base=self.fmaps_base,
                                         fmaps_max=self.fmaps_max)

        self.pth_to_tf_var_mapping = {}
        for key, val in self.mapping.pth_to_tf_var_mapping.items():
            self.pth_to_tf_var_mapping[f'mapping.{key}'] = val
        for key, val in self.truncation.pth_to_tf_var_mapping.items():
            self.pth_to_tf_var_mapping[f'truncation.{key}'] = val
        for key, val in self.synthesis.pth_to_tf_var_mapping.items():
            self.pth_to_tf_var_mapping[f'synthesis.{key}'] = val

    def forward(self,
                z,
                label=None,
                w_moving_decay=0.995,
                style_mixing_prob=0.9,
                trunc_psi=None,
                trunc_layers=None,
                randomize_noise=False,
                **_unused_kwargs):
        mapping_results = self.mapping(z, label)
        w = mapping_results['w']

        if self.training and w_moving_decay < 1:
            batch_w_avg = all_gather(w).mean(dim=0)
            self.truncation.w_avg.copy_(
                self.truncation.w_avg * w_moving_decay +
                batch_w_avg * (1 - w_moving_decay))

        if self.training and style_mixing_prob > 0:
            new_z = torch.randn_like(z)
            new_w = self.mapping(new_z, label)['w']
            if np.random.uniform() < style_mixing_prob:
                mixing_cutoff = np.random.randint(1, self.num_layers)
                w = self.truncation(w)
                new_w = self.truncation(new_w)
                w[:, :mixing_cutoff] = new_w[:, :mixing_cutoff]

        wp = self.truncation(w, trunc_psi, trunc_layers)
        synthesis_results = self.synthesis(wp, randomize_noise)

        return {**mapping_results, **synthesis_results}


class MappingModule(nn.Module):
    """Implements the latent space mapping module.

    Basically, this module executes several dense layers in sequence.
    """

    def __init__(self,
                 input_space_dim=512,
                 hidden_space_dim=512,
                 final_space_dim=512,
                 label_size=0,
                 num_layers=8,
                 normalize_input=True,
                 use_wscale=True,
                 lr_mul=0.01):
        super().__init__()

        self.input_space_dim = input_space_dim
        self.hidden_space_dim = hidden_space_dim
        self.final_space_dim = final_space_dim
        self.label_size = label_size
        self.num_layers = num_layers
        self.normalize_input = normalize_input
        self.use_wscale = use_wscale
        self.lr_mul = lr_mul

        self.norm = PixelNormLayer() if self.normalize_input else nn.Identity()

        self.pth_to_tf_var_mapping = {}
        for i in range(num_layers):
            dim_mul = 2 if label_size else 1
            in_channels = (input_space_dim * dim_mul if i == 0 else
                           hidden_space_dim)
            out_channels = (final_space_dim if i == (num_layers - 1) else
                            hidden_space_dim)
            self.add_module(f'dense{i}',
                            DenseBlock(in_channels=in_channels,
                                       out_channels=out_channels,
                                       use_wscale=self.use_wscale,
                                       lr_mul=self.lr_mul))
            self.pth_to_tf_var_mapping[f'dense{i}.weight'] = f'Dense{i}/weight'
            self.pth_to_tf_var_mapping[f'dense{i}.bias'] = f'Dense{i}/bias'
        if label_size:
            self.label_weight = nn.Parameter(
                torch.randn(label_size, input_space_dim))
            self.pth_to_tf_var_mapping[f'label_weight'] = f'LabelConcat/weight'

    def forward(self, z, label=None):
        if z.ndim != 2 or z.shape[1] != self.input_space_dim:
            raise ValueError(f'Input latent code should be with shape '
                             f'[batch_size, input_dim], where '
                             f'`input_dim` equals to {self.input_space_dim}!\n'
                             f'But `{z.shape}` is received!')
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
            embedding = torch.matmul(label, self.label_weight)
            z = torch.cat((z, embedding), dim=1)

        z = self.norm(z)
        w = z
        for i in range(self.num_layers):
            w = self.__getattr__(f'dense{i}')(w)
        results = {
            'z': z,
            'label': label,
            'w': w,
        }
        if self.label_size:
            results['embedding'] = embedding
        return results


class TruncationModule(nn.Module):
    """Implements the truncation module.

    Truncation is executed as follows:

    For layers in range [0, truncation_layers), the truncated w-code is computed
    as

    w_new = w_avg + (w - w_avg) * truncation_psi

    To disable truncation, please set
    (1) truncation_psi = 1.0 (None) OR
    (2) truncation_layers = 0 (None)

    NOTE: The returned tensor is layer-wise style codes.
    """

    def __init__(self, w_space_dim, num_layers, repeat_w=True):
        super().__init__()

        self.num_layers = num_layers
        self.w_space_dim = w_space_dim
        self.repeat_w = repeat_w

        if self.repeat_w:
            self.register_buffer('w_avg', torch.zeros(w_space_dim))
        else:
            self.register_buffer('w_avg', torch.zeros(num_layers * w_space_dim))
        self.pth_to_tf_var_mapping = {'w_avg': 'dlatent_avg'}

    def forward(self, w, trunc_psi=None, trunc_layers=None):
        if w.ndim == 2:
            if self.repeat_w and w.shape[1] == self.w_space_dim:
                w = w.view(-1, 1, self.w_space_dim)
                wp = w.repeat(1, self.num_layers, 1)
            else:
                assert w.shape[1] == self.w_space_dim * self.num_layers
                wp = w.view(-1, self.num_layers, self.w_space_dim)
        else:
            wp = w
        assert wp.ndim == 3
        assert wp.shape[1:] == (self.num_layers, self.w_space_dim)

        trunc_psi = 1.0 if trunc_psi is None else trunc_psi
        trunc_layers = 0 if trunc_layers is None else trunc_layers
        if trunc_psi < 1.0 and trunc_layers > 0:
            layer_idx = np.arange(self.num_layers).reshape(1, -1, 1)
            coefs = np.ones_like(layer_idx, dtype=np.float32)
            coefs[layer_idx < trunc_layers] *= trunc_psi
            coefs = torch.from_numpy(coefs).to(wp)
            w_avg = self.w_avg.view(1, -1, self.w_space_dim)
            wp = w_avg + (wp - w_avg) * coefs
        return wp


class SynthesisModule(nn.Module):
    """Implements the image synthesis module.

    Basically, this module executes several convolutional layers in sequence.
    """

    def __init__(self,
                 resolution=1024,
                 init_resolution=4,
                 w_space_dim=512,
                 image_channels=3,
                 final_tanh=False,
                 const_input=True,
                 architecture='skip',
                 fused_modulate=True,
                 demodulate=True,
                 use_wscale=True,
                 fmaps_base=32 << 10,
                 fmaps_max=512):
        super().__init__()

        self.init_res = init_resolution
        self.init_res_log2 = int(np.log2(self.init_res))
        self.resolution = resolution
        self.final_res_log2 = int(np.log2(self.resolution))
        self.w_space_dim = w_space_dim
        self.image_channels = image_channels
        self.final_tanh = final_tanh
        self.const_input = const_input
        self.architecture = architecture
        self.fused_modulate = fused_modulate
        self.demodulate = demodulate
        self.use_wscale = use_wscale
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max

        self.num_layers = (self.final_res_log2 - self.init_res_log2 + 1) * 2

        self.pth_to_tf_var_mapping = {}
        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            res = 2 ** res_log2
            block_idx = res_log2 - self.init_res_log2

            # First convolution layer for each resolution.
            if res == self.init_res:
                if self.const_input:
                    self.add_module(f'early_layer',
                                    InputBlock(init_resolution=self.init_res,
                                               channels=self.get_nf(res)))
                    self.pth_to_tf_var_mapping[f'early_layer.const'] = (
                        f'{res}x{res}/Const/const')
                else:
                    self.add_module(f'early_layer',
                                    DenseBlock(in_channels=self.w_space_dim,
                                               out_channels=self.get_nf(res),
                                               use_wscale=self.use_wscale))
                    self.pth_to_tf_var_mapping[f'early_layer.weight'] = (
                        f'{res}x{res}/Dense/weight')
                    self.pth_to_tf_var_mapping[f'early_layer.bias'] = (
                        f'{res}x{res}/Dense/bias')
            else:
                layer_name = f'layer{2 * block_idx - 1}'
                self.add_module(
                    layer_name,
                    ModulateConvBlock(in_channels=self.get_nf(res // 2),
                                      out_channels=self.get_nf(res),
                                      resolution=res,
                                      w_space_dim=self.w_space_dim,
                                      scale_factor=2,
                                      fused_modulate=self.fused_modulate,
                                      demodulate=self.demodulate,
                                      use_wscale=self.use_wscale))
                self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                    f'{res}x{res}/Conv0_up/weight')
                self.pth_to_tf_var_mapping[f'{layer_name}.bias'] = (
                    f'{res}x{res}/Conv0_up/bias')
                self.pth_to_tf_var_mapping[f'{layer_name}.style.weight'] = (
                    f'{res}x{res}/Conv0_up/mod_weight')
                self.pth_to_tf_var_mapping[f'{layer_name}.style.bias'] = (
                    f'{res}x{res}/Conv0_up/mod_bias')
                self.pth_to_tf_var_mapping[f'{layer_name}.noise_strength'] = (
                    f'{res}x{res}/Conv0_up/noise_strength')
                self.pth_to_tf_var_mapping[f'{layer_name}.noise'] = (
                    f'noise{2 * block_idx - 1}')

                if self.architecture == 'resnet':
                    layer_name = f'layer{2 * block_idx - 1}'
                    self.add_module(
                        layer_name,
                        ConvBlock(in_channels=self.get_nf(res // 2),
                                  out_channels=self.get_nf(res),
                                  kernel_size=1,
                                  add_bias=False,
                                  scale_factor=2,
                                  use_wscale=self.use_wscale,
                                  activation_type='linear'))
                    self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                        f'{res}x{res}/Skip/weight')

            # Second convolution layer for each resolution.
            layer_name = f'layer{2 * block_idx}'
            self.add_module(
                layer_name,
                ModulateConvBlock(in_channels=self.get_nf(res),
                                  out_channels=self.get_nf(res),
                                  resolution=res,
                                  w_space_dim=self.w_space_dim,
                                  fused_modulate=self.fused_modulate,
                                  demodulate=self.demodulate,
                                  use_wscale=self.use_wscale))
            tf_layer_name = 'Conv' if res == self.init_res else 'Conv1'
            self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                f'{res}x{res}/{tf_layer_name}/weight')
            self.pth_to_tf_var_mapping[f'{layer_name}.bias'] = (
                f'{res}x{res}/{tf_layer_name}/bias')
            self.pth_to_tf_var_mapping[f'{layer_name}.style.weight'] = (
                f'{res}x{res}/{tf_layer_name}/mod_weight')
            self.pth_to_tf_var_mapping[f'{layer_name}.style.bias'] = (
                f'{res}x{res}/{tf_layer_name}/mod_bias')
            self.pth_to_tf_var_mapping[f'{layer_name}.noise_strength'] = (
                f'{res}x{res}/{tf_layer_name}/noise_strength')
            self.pth_to_tf_var_mapping[f'{layer_name}.noise'] = (
                f'noise{2 * block_idx}')

            # Output convolution layer for each resolution (if needed).
            if res_log2 == self.final_res_log2 or self.architecture == 'skip':
                layer_name = f'output{block_idx}'
                self.add_module(
                    layer_name,
                    ModulateConvBlock(in_channels=self.get_nf(res),
                                      out_channels=image_channels,
                                      resolution=res,
                                      w_space_dim=self.w_space_dim,
                                      kernel_size=1,
                                      fused_modulate=self.fused_modulate,
                                      demodulate=False,
                                      use_wscale=self.use_wscale,
                                      add_noise=False,
                                      activation_type='linear'))
                self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                    f'{res}x{res}/ToRGB/weight')
                self.pth_to_tf_var_mapping[f'{layer_name}.bias'] = (
                    f'{res}x{res}/ToRGB/bias')
                self.pth_to_tf_var_mapping[f'{layer_name}.style.weight'] = (
                    f'{res}x{res}/ToRGB/mod_weight')
                self.pth_to_tf_var_mapping[f'{layer_name}.style.bias'] = (
                    f'{res}x{res}/ToRGB/mod_bias')

        if self.architecture == 'skip':
            self.upsample = UpsamplingLayer()
        self.final_activate = nn.Tanh() if final_tanh else nn.Identity()

    def get_nf(self, res):
        """Gets number of feature maps according to current resolution."""
        return min(self.fmaps_base // res, self.fmaps_max)

    def forward(self, wp, randomize_noise=False):
        if wp.ndim != 3 or wp.shape[1:] != (self.num_layers, self.w_space_dim):
            raise ValueError(f'Input tensor should be with shape '
                             f'[batch_size, num_layers, w_space_dim], where '
                             f'`num_layers` equals to {self.num_layers}, and '
                             f'`w_space_dim` equals to {self.w_space_dim}!\n'
                             f'But `{wp.shape}` is received!')

        results = {'wp': wp}
        x = self.early_layer(wp[:, 0])
        if self.architecture == 'origin':
            for layer_idx in range(self.num_layers - 1):
                x, style = self.__getattr__(f'layer{layer_idx}')(
                    x, wp[:, layer_idx], randomize_noise)
                results[f'style{layer_idx:02d}'] = style
            image, style = self.__getattr__(f'output{layer_idx // 2}')(
                x, wp[:, layer_idx + 1])
            results[f'output_style{layer_idx // 2}'] = style
        elif self.architecture == 'skip':
            for layer_idx in range(self.num_layers - 1):
                x, style = self.__getattr__(f'layer{layer_idx}')(
                    x, wp[:, layer_idx], randomize_noise)
                results[f'style{layer_idx:02d}'] = style
                if layer_idx % 2 == 0:
                    temp, style = self.__getattr__(f'output{layer_idx // 2}')(
                        x, wp[:, layer_idx + 1])
                    results[f'output_style{layer_idx // 2}'] = style
                    if layer_idx == 0:
                        image = temp
                    else:
                        image = temp + self.upsample(image)
        elif self.architecture == 'resnet':
            x, style = self.layer0(x)
            results[f'style00'] = style
            for layer_idx in range(1, self.num_layers - 1, 2):
                residual = self.__getattr__(f'skip_layer{layer_idx // 2}')(x)
                x, style = self.__getattr__(f'layer{layer_idx}')(
                    x, wp[:, layer_idx], randomize_noise)
                results[f'style{layer_idx:02d}'] = style
                x, style = self.__getattr__(f'layer{layer_idx + 1}')(
                    x, wp[:, layer_idx + 1], randomize_noise)
                results[f'style{layer_idx + 1:02d}'] = style
                x = (x + residual) / np.sqrt(2.0)
            image, style = self.__getattr__(f'output{layer_idx // 2 + 1}')(
                x, wp[:, layer_idx + 2])
            results[f'output_style{layer_idx // 2}'] = style
        results['image'] = self.final_activate(image)
        return results


class PixelNormLayer(nn.Module):
    """Implements pixel-wise feature vector normalization layer."""

    def __init__(self, dim=1, epsilon=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = epsilon

    def forward(self, x):
        norm = torch.sqrt(
            torch.mean(x ** 2, dim=self.dim, keepdim=True) + self.eps)
        return x / norm


class UpsamplingLayer(nn.Module):
    """Implements the upsampling layer.

    This layer can also be used as filtering by setting `scale_factor` as 1.
    """

    def __init__(self,
                 scale_factor=2,
                 kernel=(1, 3, 3, 1),
                 extra_padding=0,
                 kernel_gain=None):
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
        if kernel_gain is None:
            kernel = kernel * (scale_factor ** 2)
        else:
            assert kernel_gain > 0
            kernel = kernel * (kernel_gain ** 2)
        assert kernel.ndim == 2
        assert kernel.shape[0] == kernel.shape[1]
        kernel = kernel[np.newaxis, np.newaxis]
        self.register_buffer('kernel', torch.from_numpy(kernel))
        self.kernel = self.kernel.flip(0, 1)

        self.upsample_padding = (0, scale_factor - 1,  # Width padding.
                                 0, 0,                 # Width.
                                 0, scale_factor - 1,  # Height padding.
                                 0, 0,                 # Height.
                                 0, 0,                 # Channel.
                                 0, 0)                 # Batch size.

        padding = kernel.shape[2] - scale_factor + extra_padding
        self.padding = ((padding + 1) // 2 + scale_factor - 1, padding // 2,
                        (padding + 1) // 2 + scale_factor - 1, padding // 2)

    def forward(self, x):
        assert x.ndim == 4
        channels = x.shape[1]
        if self.scale_factor > 1:
            x = x.view(-1, channels, x.shape[2], 1, x.shape[3], 1)
            x = F.pad(x, self.upsample_padding, mode='constant', value=0)
            x = x.view(-1, channels, x.shape[2] * self.scale_factor,
                       x.shape[4] * self.scale_factor)
        x = x.view(-1, 1, x.shape[2], x.shape[3])
        x = F.pad(x, self.padding, mode='constant', value=0)
        x = F.conv2d(x, self.kernel, stride=1)
        x = x.view(-1, channels, x.shape[2], x.shape[3])
        return x


class InputBlock(nn.Module):
    """Implements the input block.

    Basically, this block starts from a const input, which is with shape
    `(channels, init_resolution, init_resolution)`.
    """

    def __init__(self, init_resolution, channels):
        super().__init__()
        self.const = nn.Parameter(
            torch.randn(1, channels, init_resolution, init_resolution))

    def forward(self, w):
        x = self.const.repeat(w.shape[0], 1, 1, 1)
        return x


class ConvBlock(nn.Module):
    """Implements the convolutional block (no style modulation).

    Basically, this block executes, convolutional layer, filtering layer (if
    needed), and activation layer in sequence.

    NOTE: This block is particularly used for skip-connection branch in the
    `resnet` structure.
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
                 activation_type='lrelu'):
        """Initializes with block settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            kernel_size: Size of the convolutional kernels. (default: 3)
            add_bias: Whether to add bias onto the convolutional result.
                (default: True)
            scale_factor: Scale factor for upsampling. `1` means skip
                upsampling. (default: 1)
            filtering_kernel: Kernel used for filtering after upsampling.
                (default: (1, 3, 3, 1))
            use_wscale: Whether to use weight scaling. (default: True)
            wscale_gain: Gain factor for weight scaling. (default: _WSCALE_GAIN)
            lr_mul: Learning multiplier for both weight and bias. (default: 1.0)
            activation_type: Type of activation. Support `linear` and `lrelu`.
                (default: `lrelu`)

        Raises:
            NotImplementedError: If the `activation_type` is not supported.
        """
        super().__init__()

        if scale_factor > 1:
            self.use_conv2d_transpose = True
            extra_padding = scale_factor - kernel_size
            self.filter = UpsamplingLayer(scale_factor=1,
                                          kernel=filtering_kernel,
                                          extra_padding=extra_padding,
                                          kernel_gain=scale_factor)
            self.stride = scale_factor
            self.padding = 0  # Padding is done in `UpsamplingLayer`.
        else:
            self.use_conv2d_transpose = False
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
        weight = self.weight * self.wscale
        bias = self.bias * self.bscale if self.bias is not None else None
        if self.use_conv2d_transpose:
            weight = weight.permute(1, 0, 2, 3).flip(2, 3)
            x = F.conv_transpose2d(x,
                                   weight=weight,
                                   bias=bias,
                                   stride=self.scale_factor,
                                   padding=self.padding)
            x = self.filter(x)
        else:
            x = F.conv2d(x,
                         weight=weight,
                         bias=bias,
                         stride=self.stride,
                         padding=self.padding)
        x = self.activate(x) * self.activate_scale
        return x


class ModulateConvBlock(nn.Module):
    """Implements the convolutional block with style modulation."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 resolution,
                 w_space_dim,
                 kernel_size=3,
                 add_bias=True,
                 scale_factor=1,
                 filtering_kernel=(1, 3, 3, 1),
                 fused_modulate=True,
                 demodulate=True,
                 use_wscale=True,
                 wscale_gain=_WSCALE_GAIN,
                 lr_mul=1.0,
                 add_noise=True,
                 activation_type='lrelu',
                 epsilon=1e-8):
        """Initializes with block settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            resolution: Resolution of the output tensor.
            w_space_dim: Dimension of W space for style modulation.
            kernel_size: Size of the convolutional kernels. (default: 3)
            add_bias: Whether to add bias onto the convolutional result.
                (default: True)
            scale_factor: Scale factor for upsampling. `1` means skip
                upsampling. (default: 1)
            filtering_kernel: Kernel used for filtering after upsampling.
                (default: (1, 3, 3, 1))
            fused_modulate: Whether to fuse `style_modulate` and `conv2d`
                together. (default: True)
            demodulate: Whether to perform style demodulation. (default: True)
            use_wscale: Whether to use weight scaling. (default: True)
            wscale_gain: Gain factor for weight scaling. (default: _WSCALE_GAIN)
            lr_mul: Learning multiplier for both weight and bias. (default: 1.0)
            add_noise: Whether to add noise onto the output tensor. (default:
                True)
            activation_type: Type of activation. Support `linear` and `lrelu`.
                (default: `lrelu`)
            epsilon: Small number to avoid `divide by zero`. (default: 1e-8)

        Raises:
            NotImplementedError: If the `activation_type` is not supported.
        """
        super().__init__()

        self.res = resolution
        self.in_c = in_channels
        self.out_c = out_channels
        self.ksize = kernel_size
        self.eps = epsilon

        if scale_factor > 1:
            self.use_conv2d_transpose = True
            extra_padding = scale_factor - kernel_size
            self.filter = UpsamplingLayer(scale_factor=1,
                                          kernel=filtering_kernel,
                                          extra_padding=extra_padding,
                                          kernel_gain=scale_factor)
            self.stride = scale_factor
            self.padding = 0  # Padding is done in `UpsamplingLayer`.
        else:
            self.use_conv2d_transpose = False
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

        self.style = DenseBlock(in_channels=w_space_dim,
                                out_channels=in_channels,
                                additional_bias=1.0,
                                use_wscale=use_wscale,
                                activation_type='linear')

        self.fused_modulate = fused_modulate
        self.demodulate = demodulate

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

        self.add_noise = add_noise
        if self.add_noise:
            self.register_buffer('noise', torch.randn(1, 1, self.res, self.res))
            self.noise_strength = nn.Parameter(torch.zeros(()))

    def forward(self, x, w, randomize_noise=False):
        batch = x.shape[0]

        weight = self.weight * self.wscale
        weight = weight.permute(2, 3, 1, 0)

        # Style modulation.
        style = self.style(w)
        _weight = weight.view(1, self.ksize, self.ksize, self.in_c, self.out_c)
        _weight = _weight * style.view(batch, 1, 1, self.in_c, 1)

        # Style demodulation.
        if self.demodulate:
            _weight_norm = torch.sqrt(
                torch.sum(_weight ** 2, dim=[1, 2, 3]) + self.eps)
            _weight = _weight / _weight_norm.view(batch, 1, 1, 1, self.out_c)

        if self.fused_modulate:
            x = x.view(1, batch * self.in_c, x.shape[2], x.shape[3])
            weight = _weight.permute(1, 2, 3, 0, 4).reshape(
                self.ksize, self.ksize, self.in_c, batch * self.out_c)
        else:
            x = x * style.view(batch, self.in_c, 1, 1)

        if self.use_conv2d_transpose:
            weight = weight.flip(0, 1)
            if self.fused_modulate:
                weight = weight.view(
                    self.ksize, self.ksize, self.in_c, batch, self.out_c)
                weight = weight.permute(0, 1, 4, 3, 2)
                weight = weight.reshape(
                    self.ksize, self.ksize, self.out_c, batch * self.in_c)
                weight = weight.permute(3, 2, 0, 1)
            else:
                weight = weight.permute(2, 3, 0, 1)
            x = F.conv_transpose2d(x,
                                   weight=weight,
                                   bias=None,
                                   stride=self.stride,
                                   padding=self.padding,
                                   groups=(batch if self.fused_modulate else 1))
            x = self.filter(x)
        else:
            weight = weight.permute(3, 2, 0, 1)
            x = F.conv2d(x,
                         weight=weight,
                         bias=None,
                         stride=self.stride,
                         padding=self.padding,
                         groups=(batch if self.fused_modulate else 1))

        if self.fused_modulate:
            x = x.view(batch, self.out_c, self.res, self.res)
        elif self.demodulate:
            x = x / _weight_norm.view(batch, self.out_c, 1, 1)

        if self.add_noise:
            if randomize_noise:
                noise = torch.randn(x.shape[0], 1, self.res, self.res).to(x)
            else:
                noise = self.noise
            x = x + noise * self.noise_strength.view(1, 1, 1, 1)

        bias = self.bias * self.bscale if self.bias is not None else None
        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
        x = self.activate(x) * self.activate_scale
        return x, style


class DenseBlock(nn.Module):
    """Implements the dense block.

    Basically, this block executes fully-connected layer and activation layer.

    NOTE: This layer supports adding an additional bias beyond the trainable
    bias parameter. This is specially used for the mapping from the w code to
    the style code.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 add_bias=True,
                 additional_bias=0,
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
            additional_bias: The additional bias, which is independent from the
                bias parameter. (default: 0.0)
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
        self.additional_bias = additional_bias

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
        x = self.activate(x + self.additional_bias) * self.activate_scale
        return x
