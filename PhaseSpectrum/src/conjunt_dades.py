import torch
from torch.utils.data import Dataset

from caracteristiques import obre_imatgeRGB, construeix_x0


class ConjuntDadesDFFD(Dataset):
    def __init__(self, manifest, mida=(299, 299), transformacio=None):
        self.manifest = manifest.reset_index(drop=True).copy()
        self.mida = mida
        # transformacio guarda posibles canvis lleugers a la imtage
        self.transformacio = transformacio
 
    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        fila = self.manifest.iloc[idx]

        ruta = fila["path"]
        etiqueta = torch.tensor(float(fila["label"]), dtype=torch.float32)

        img_rgb = obre_imatgeRGB(ruta, mida=self.mida)

        if self.transformacio is not None:
            img_rgb = self.transformacio(img_rgb)

        x0, magnitud_log, fase_norm, lbp_tensor = construeix_x0(img_rgb)

        return {
            "x0": x0,
            "magnitud": magnitud_log,
            "fase": fase_norm,
            "etiqueta": etiqueta,
            "ruta": ruta,
            "source": fila["source"]
        }