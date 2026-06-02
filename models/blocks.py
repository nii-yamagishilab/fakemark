import math
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


class Mish(nn.Module):
    """Mish activation function.

    Applies the Mish nonlinearity element-wise:
        Mish(x) = x * tanh(softplus(x))
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Mish activation function.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor of the same shape with Mish applied element-wise.
        """
        return x * torch.tanh(F.softplus(x))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer-based models.

    Adds fixed sinusoidal positional embeddings to the input tensor to provide
    information about token positions in the sequence.

    Args:
        d_model: Dimensionality of the model embeddings.
        dropout: Dropout probability applied after adding positional encoding.
        max_len: Maximum sequence length supported by the encoding.
    """

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the input tensor.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Tensor with positional encoding added and dropout applied.
        """
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class FCBlock(nn.Module):
    """Fully connected block with optional activation and dropout.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        activation: Optional activation module to apply after the linear layer.
        bias: Whether to include a bias term in the linear layer.
        dropout: Dropout probability. ``None`` disables dropout.
        spectral_norm: Whether to apply spectral normalization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Optional[nn.Module] = None,
        bias: bool = False,
        dropout: Optional[float] = None,
        spectral_norm: bool = False,
    ) -> None:
        super(FCBlock, self).__init__()
        self.fc_layer = nn.Sequential()
        self.fc_layer.add_module(
            "fc_layer",
            LinearNorm(
                in_features,
                out_features,
                bias,
                spectral_norm,
            ),
        )
        if activation is not None:
            self.fc_layer.add_module("activ", activation)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the fully connected block.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        x = self.fc_layer(x)
        if self.dropout is not None:
            x = F.dropout(x, self.dropout, self.training)
        return x


class LinearNorm(nn.Module):
    """Linear projection with Xavier initialization and optional spectral normalization.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: Whether to include a bias term.
        spectral_norm: Whether to apply spectral normalization to the layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        spectral_norm: bool = False,
    ) -> None:
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)
        if spectral_norm:
            self.linear = nn.utils.spectral_norm(self.linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear projection.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        x = self.linear(x)
        return x


class Conv1DBlock(nn.Module):
    """1D convolutional block with optional activation, dropout, and masking.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        activation: Optional activation module to apply after convolution.
        dropout: Dropout probability. ``None`` disables dropout.
        spectral_norm: Whether to apply spectral normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: Optional[nn.Module] = None,
        dropout: Optional[float] = None,
        spectral_norm: bool = False,
    ) -> None:
        super(Conv1DBlock, self).__init__()

        self.conv_layer = nn.Sequential()
        self.conv_layer.add_module(
            "conv_layer",
            ConvNorm(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="tanh",
                spectral_norm=spectral_norm,
            ),
        )
        if activation is not None:
            self.conv_layer.add_module("activ", activation)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply the 1D convolutional block.

        Args:
            x: Input tensor of shape ``(batch, channels, length)``.
            mask: Optional boolean mask of shape ``(batch, length)``. Positions
                where the mask is ``True`` are zeroed out in the output.

        Returns:
            Output tensor of shape ``(batch, out_channels, length)``.
        """
        x = self.conv_layer(x)

        if self.dropout is not None:
            x = F.dropout(x, self.dropout, self.training)

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)

        return x


class ConvNorm(nn.Module):
    """1D convolution with optional spectral normalization.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Zero-padding added to both sides of the input. If ``None``,
            padding is computed automatically to preserve the input length
            (requires an odd kernel size).
        dilation: Spacing between kernel elements.
        bias: Whether to include a bias term.
        w_init_gain: Gain value for Xavier initialization (unused, kept for
            interface compatibility).
        spectral_norm: Whether to apply spectral normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        bias: bool = True,
        w_init_gain: str = "linear",
        spectral_norm: bool = False,
    ) -> None:
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply the 1D convolution.

        Args:
            signal: Input tensor of shape ``(batch, in_channels, length)``.

        Returns:
            Output tensor of shape ``(batch, out_channels, new_length)``.
        """
        conv_signal = self.conv(signal)
        return conv_signal


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism.

    Performs scaled dot-product attention with multiple parallel attention heads.

    Args:
        n_head: Number of attention heads.
        d_model: Dimensionality of the input and output.
        d_k: Dimensionality of the key and query vectors per head.
        d_v: Dimensionality of the value vectors per head.
        dropout: Dropout probability.
        layer_norm: Whether to apply layer normalization to the output.
        spectral_norm: Whether to apply spectral normalization to linear
            projections.
    """

    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1,
        layer_norm: bool = False,
        spectral_norm: bool = False,
    ) -> None:
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = LinearNorm(d_model, n_head * d_k, spectral_norm=spectral_norm)
        self.w_ks = LinearNorm(d_model, n_head * d_k, spectral_norm=spectral_norm)
        self.w_vs = LinearNorm(d_model, n_head * d_v, spectral_norm=spectral_norm)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model) if layer_norm else None

        self.fc = LinearNorm(n_head * d_v, d_model, spectral_norm=spectral_norm)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head attention.

        Args:
            q: Query tensor of shape ``(batch, len_q, d_model)``.
            k: Key tensor of shape ``(batch, len_k, d_model)``.
            v: Value tensor of shape ``(batch, len_v, d_model)``.
            mask: Optional attention mask tensor. Will be repeated across heads.

        Returns:
            A tuple of:
                - Output tensor of shape ``(batch, len_q, d_model)``.
                - Attention weights tensor.
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = output + residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism.

    Computes attention weights from queries and keys, then applies them to
    values. Optionally masks out positions before the softmax.

    Args:
        temperature: Scaling factor for the dot-product (typically sqrt(d_k)).
    """

    def __init__(self, temperature: float) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention.

        Args:
            q: Query tensor of shape ``(batch, len_q, d_k)``.
            k: Key tensor of shape ``(batch, len_k, d_k)``.
            v: Value tensor of shape ``(batch, len_v, d_v)``.
            mask: Optional boolean mask. Positions where the mask is ``True``
                are filled with negative infinity before softmax.

        Returns:
            A tuple of:
                - Output tensor of shape ``(batch, len_q, d_v)``.
                - Attention weights of shape ``(batch, len_q, len_k)``.
        """
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn


class SkipGatedBlock(nn.Module):
    """Gated 2D convolutional block with an optional skip connection.

    Applies a convolution gated by a sigmoid activation. When the number of
    input and output channels match, a residual skip connection is added.

    Args:
        c_in: Number of input channels.
        c_out: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Zero-padding added to both sides of the input.
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int, stride: int, padding: int) -> None:
        super(SkipGatedBlock, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.gate = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.skip_connection = c_in == c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated convolution with optional skip connection.

        Args:
            x: Input tensor of shape ``(batch, c_in, height, width)``.

        Returns:
            Output tensor of shape ``(batch, c_out, height', width')``.
        """
        conv_output = self.conv(x)
        gated_output = torch.sigmoid(self.gate(x))
        output = conv_output * gated_output
        if self.skip_connection:
            output += x
        return output


class ReluBlock(nn.Module):
    """2D convolutional block with instance normalization and LeakyReLU.

    Args:
        c_in: Number of input channels.
        c_out: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Zero-padding added to both sides of the input.
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int, stride: int, padding: int) -> None:
        super(ReluBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.InstanceNorm2d(c_out),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, instance normalization, and LeakyReLU.

        Args:
            x: Input tensor of shape ``(batch, c_in, height, width)``.

        Returns:
            Output tensor of shape ``(batch, c_out, height', width')``.
        """
        return self.conv(x)


class Conv2Encoder(nn.Module):
    """Stacked 2D convolutional encoder.

    Builds a sequence of 2D convolutional blocks (either gated or ReLU-based)
    that maintain spatial dimensions through same-padding.

    Args:
        input_channel: Number of input channels.
        hidden_dim: Number of hidden channels used throughout the encoder.
        block: Block type to use. Either ``'skip'`` for
            :class:`SkipGatedBlock` or ``'relu'`` for :class:`ReluBlock`.
        n_layers: Total number of convolutional layers.

    Raises:
        ValueError: If ``block`` is not ``'skip'`` or ``'relu'``.
    """

    def __init__(
        self,
        input_channel: int = 1,
        hidden_dim: int = 64,
        block: str = 'skip',
        n_layers: int = 3,
    ) -> None:
        super(Conv2Encoder, self).__init__()
        if block == 'skip':
            core: Type[nn.Module] = SkipGatedBlock
        elif block == 'relu':
            core = ReluBlock
        else:
            raise ValueError(f"Invalid block type: {block}")

        layers = [core(c_in=input_channel, c_out=hidden_dim, kernel_size=3, stride=1, padding=1)]

        for i in range(n_layers - 1):
            layers.append(core(c_in=hidden_dim, c_out=hidden_dim, kernel_size=3, stride=1, padding=1))

        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input through stacked 2D convolutional blocks.

        Args:
            x: Input tensor of shape ``(batch, input_channel, height, width)``.

        Returns:
            Encoded tensor of shape ``(batch, hidden_dim, height, width)``.
        """
        return self.main(x)


class WatermarkEmbedder(nn.Module):
    """Embeds a watermark into an input representation.

    Uses a stack of 2D convolutional blocks to transform the input and produce
    a single-channel output suitable for watermark embedding.

    Args:
        input_channel: Number of input channels.
        hidden_dim: Number of hidden channels used in intermediate layers.
        block: Block type to use. Either ``'skip'`` for
            :class:`SkipGatedBlock` or ``'relu'`` for :class:`ReluBlock`.
        n_layers: Total number of convolutional layers.

    Raises:
        ValueError: If ``block`` is not ``'skip'`` or ``'relu'``.
    """

    def __init__(
        self,
        input_channel: int = 1,
        hidden_dim: int = 64,
        block: str = 'skip',
        n_layers: int = 4,
    ) -> None:
        super(WatermarkEmbedder, self).__init__()
        if block == 'skip':
            core: Type[nn.Module] = SkipGatedBlock
        elif block == 'relu':
            core = ReluBlock
        else:
            raise ValueError(f"Invalid block type: {block}")

        layers = [core(c_in=input_channel, c_out=hidden_dim, kernel_size=3, stride=1, padding=1)]

        for i in range(n_layers - 2):
            layers.append(core(c_in=hidden_dim, c_out=hidden_dim, kernel_size=3, stride=1, padding=1))

        layers.append(core(c_in=hidden_dim, c_out=1, kernel_size=1, stride=1, padding=0))

        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a watermark into the input.

        Args:
            x: Input tensor of shape ``(batch, input_channel, height, width)``.

        Returns:
            Single-channel output tensor of shape ``(batch, 1, height, width)``.
        """
        return self.main(x)


class WatermarkExtracter(nn.Module):
    """Extracts a watermark from a watermarked representation.

    Uses a stack of 2D convolutional blocks to recover a single-channel
    watermark signal from the input.

    Args:
        input_channel: Number of input channels.
        hidden_dim: Number of hidden channels used in intermediate layers.
        block: Block type to use. Either ``'skip'`` for
            :class:`SkipGatedBlock` or ``'relu'`` for :class:`ReluBlock`.
        n_layers: Total number of convolutional layers.

    Raises:
        ValueError: If ``block`` is not ``'skip'`` or ``'relu'``.
    """

    def __init__(
        self,
        input_channel: int = 1,
        hidden_dim: int = 64,
        block: str = 'skip',
        n_layers: int = 6,
    ) -> None:
        super(WatermarkExtracter, self).__init__()
        if block == 'skip':
            core: Type[nn.Module] = SkipGatedBlock
        elif block == 'relu':
            core = ReluBlock
        else:
            raise ValueError(f"Invalid block type: {block}")
        layers = [core(c_in=input_channel, c_out=hidden_dim, kernel_size=3, stride=1, padding=1)]

        for i in range(n_layers - 2):
            layers.append(core(c_in=hidden_dim, c_out=hidden_dim, kernel_size=3, stride=1, padding=1))

        layers.append(core(c_in=hidden_dim, c_out=1, kernel_size=3, stride=1, padding=1))

        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract a watermark from the input.

        Args:
            x: Input tensor of shape ``(batch, input_channel, height, width)``.

        Returns:
            Single-channel output tensor of shape ``(batch, 1, height, width)``.
        """
        return self.main(x)
