import torch
import torch.nn as nn


class AtencioPerFase(nn.Module):
    def __init__(self, canals_ocults=8):
        super().__init__()

        self.branca_magnitud = nn.Sequential(
            nn.Conv2d(1, canals_ocults, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(canals_ocults, canals_ocults, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.branca_fase = nn.Sequential(
            nn.Conv2d(1, canals_ocults, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(canals_ocults, canals_ocults, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.fusio = nn.Sequential(
            nn.Conv2d(2 * canals_ocults, canals_ocults, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(canals_ocults, 5, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x0, magnitud, fase):
        feat_magnitud = self.branca_magnitud(magnitud)
        feat_fase = self.branca_fase(fase)

        fusio = torch.cat([feat_magnitud, feat_fase], dim=1)
        a0 = self.fusio(fusio)

        x_tilde = x0 * a0
        return x_tilde, a0


class AdaptadorCanals(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1x1 = nn.Conv2d(5, 3, kernel_size=1)

    def forward(self, x):
        return self.conv_1x1(x)


class BackboneCNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.bloc1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.bloc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.bloc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.pool_global = nn.AdaptiveAvgPool2d((1, 1))

        self.classificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.bloc1(x)
        x = self.bloc2(x)
        x = self.bloc3(x)

        x = self.pool_global(x)
        logits = self.classificador(x)

        return logits


class ModelEspectreFase(nn.Module):
    def __init__(self, canals_ocults_atencio=8, num_classes=1):
        super().__init__()

        self.atencio = AtencioPerFase(canals_ocults=canals_ocults_atencio)
        self.adaptador = AdaptadorCanals()
        self.backbone = BackboneCNN(num_classes=num_classes)

    def forward(self, x0, magnitud, fase, retorna_atencio=False):
        x_tilde, a0 = self.atencio(x0, magnitud, fase)
        x_adaptada = self.adaptador(x_tilde)
        logits = self.backbone(x_adaptada)

        if retorna_atencio:
            return logits, a0

        return logits