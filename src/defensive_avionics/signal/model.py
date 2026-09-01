"""Residual 1D-CNN for communication-modulation classification."""

from __future__ import annotations


def build_iq_cnn(num_classes: int):
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Install the signal dependency group first.") from exc

    class ResidualBlock1D(nn.Module):
        def __init__(
            self,
            input_channels: int,
            output_channels: int,
            dropout: float,
        ) -> None:
            super().__init__()

            self.conv1 = nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=5,
                padding=2,
                bias=False,
            )
            self.batch_norm1 = nn.BatchNorm1d(output_channels)

            self.conv2 = nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size=5,
                padding=2,
                bias=False,
            )
            self.batch_norm2 = nn.BatchNorm1d(output_channels)

            if input_channels != output_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(
                        input_channels,
                        output_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm1d(output_channels),
                )
            else:
                self.shortcut = nn.Identity()

            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.pool = nn.MaxPool1d(kernel_size=2)

        def forward(self, inputs):
            residual = self.shortcut(inputs)

            features = self.conv1(inputs)
            features = self.batch_norm1(features)
            features = self.activation(features)

            features = self.conv2(features)
            features = self.batch_norm2(features)
            features = self.dropout(features)

            features = self.activation(features + residual)
            return self.pool(features)

    class IQConvNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.stem = nn.Sequential(
                nn.Conv1d(
                    2,
                    64,
                    kernel_size=7,
                    padding=3,
                    bias=False,
                ),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )

            self.features = nn.Sequential(
                ResidualBlock1D(64, 64, 0.10),
                ResidualBlock1D(64, 128, 0.15),
                ResidualBlock1D(128, 256, 0.20),
            )

            self.average_pool = nn.AdaptiveAvgPool1d(1)
            self.maximum_pool = nn.AdaptiveMaxPool1d(1)

            self.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.30),
                nn.Linear(256, num_classes),
            )

        def forward(self, inputs):
            features = self.stem(inputs)
            features = self.features(features)

            average_features = self.average_pool(features).squeeze(-1)
            maximum_features = self.maximum_pool(features).squeeze(-1)

            combined_features = torch.cat(
                (average_features, maximum_features),
                dim=1,
            )

            return self.classifier(combined_features)

    return IQConvNet()
