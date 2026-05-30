from pathlib import Path
import random
import pandas as pd

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

METODE_REAL = "ffhq"
METODES_FAKE = ["pggan_v1", "pggan_v2", "stargan", "stylegan_celeba", "stylegan_ffhq"]


def es_imatge(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in EXTENSIONS


def llista_imatges_recursiu(carpeta: Path):
    if not carpeta.exists():
        raise FileNotFoundError(f"No existeix la carpeta: {carpeta}")

    return [p for p in carpeta.rglob("*") if es_imatge(p)]


def mostra_rutes(rutes, n=None, llavor=42):
    """
        Nota: aquest metode retorna una seleccio d'imatges aleatories donada una ruta, y una quantitat
    """
    rutes = list(rutes)
    rng = random.Random(llavor)

    if n is None or n >= len(rutes):
        rng.shuffle(rutes)
        return rutes

    return rng.sample(rutes, n)


def manifest1vs1(arrel, metode_fake, llavor=42):

    carpeta_reals = arrel / "Real" / METODE_REAL
    carpeta_fakes = arrel / "Fake" / metode_fake

    # agafam les rutes de les imatges falses i reals
    rutes_reals = llista_imatges_recursiu(carpeta_reals)
    rutes_fakes = llista_imatges_recursiu(carpeta_fakes)

    if len(rutes_reals) == 0:
        raise ValueError(f"No s'han trobat imatges reals a: {carpeta_reals}")

    if len(rutes_fakes) == 0:
        raise ValueError(f"No s'han trobat imatges fake a: {carpeta_fakes}")

    # si elegim l'experiment stargan, limitam la quantitat d'imatges a 20k
    if metode_fake == "stargan":
        rutes_fakes = mostra_rutes(rutes_fakes, 20000, llavor=llavor)

    #aqui s'equipara la cuantitat d'imatges reals i falses, per comptar amb les mateixes
    rutes_reals = mostra_rutes(rutes_reals, len(rutes_fakes), llavor=llavor)

    files = []

    # etiquetam cada ruta amb el seu metode correponent i el valor 0,1
    for p in rutes_reals:
        files.append({"path": str(p), "label": 0, "source": METODE_REAL})

    for p in rutes_fakes:
        files.append({"path": str(p), "label": 1, "source": metode_fake})

    # construïm el nou "dataset" y el mesclam
    df = pd.DataFrame(files, columns=["path", "label", "source"])
    return df.sample(frac=1, random_state=llavor).reset_index(drop=True)


def manifestMescla(arrel, metodes_fake=None, fake_per_metode=4000, llavor=42):

    if metodes_fake is None:
        metodes_fake = METODES_FAKE

    if isinstance(metodes_fake, str):
        metodes_fake = [metodes_fake]

    carpeta_reals = arrel / "Real" / METODE_REAL
    rutes_reals = llista_imatges_recursiu(carpeta_reals)

    if len(rutes_reals) == 0:
        raise ValueError(f"No s'han trobat imatges reals a: {carpeta_reals}")

    files_fake = []

    for metode in metodes_fake:

        carpeta_fake = arrel / "Fake" / metode
        rutes_fake = llista_imatges_recursiu(carpeta_fake)

        if len(rutes_fake) == 0:
            raise ValueError(f"No s'han trobat imatges fake a: {carpeta_fake}")

        rutes_fake = mostra_rutes(rutes_fake, fake_per_metode, llavor=llavor)

        for p in rutes_fake:
            files_fake.append({"path": str(p), "label": 1, "source": metode})

    total_fake = len(files_fake)

    if total_fake == 0:
        raise ValueError("No s'ha pogut construir cap mostra fake per a la mescla")

    rutes_reals = mostra_rutes(rutes_reals, total_fake, llavor=llavor)

    files_reals = []
    for p in rutes_reals:
        files_reals.append({"path": str(p), "label": 0, "source": METODE_REAL})

    df = pd.DataFrame(files_reals + files_fake, columns=["path", "label", "source"])
    return df.sample(frac=1, random_state=llavor).reset_index(drop=True)

def construeix_manifest_faceapp(carpeta_reals, carpeta_fakes, source_fake="faceapp", llavor=42):
    """
    Construeix un manifest per a un split concret ja definit pel dataset.
    En aquest cas:
    - deixam totes les fake de faceapp
    - retallam les reals de ffhq per igualar la quantitat
    """
    rutes_reals = llista_imatges_recursiu(carpeta_reals)
    rutes_fakes = llista_imatges_recursiu(carpeta_fakes)

    if len(rutes_reals) == 0:
        raise ValueError(f"No s'han trobat imatges reals a: {carpeta_reals}")

    if len(rutes_fakes) == 0:
        raise ValueError(f"No s'han trobat imatges fake a: {carpeta_fakes}")

    if len(rutes_reals) < len(rutes_fakes):
        raise ValueError(
            f"No hi ha prou imatges reals a {carpeta_reals}. "
            f"Reals: {len(rutes_reals)} | Fakes: {len(rutes_fakes)}"
        )

    # mantenim totes les fake i retallam les reals
    rutes_reals = mostra_rutes(rutes_reals, len(rutes_fakes), llavor=llavor)

    files = []

    for p in rutes_reals:
        files.append({"path": str(p), "label": 0, "source": "ffhq"})

    for p in rutes_fakes:
        files.append({"path": str(p), "label": 1, "source": source_fake})

    df = pd.DataFrame(files, columns=["path", "label", "source"])
    return df.sample(frac=1, random_state=llavor).reset_index(drop=True)


def manifestFaceAppFFHQ(arrel, llavor=42):
    """
    Construeix directament els manifests train, val i test per a:
    - ffhq com a real
    - faceapp com a fake
    """
    carpeta_ffhq = arrel / "Real" / "ffhq"
    carpeta_faceapp = arrel / "Fake" / "faceapp"

    df_train = construeix_manifest_faceapp(
        carpeta_ffhq / "train",
        carpeta_faceapp / "train",
        source_fake="faceapp",
        llavor=llavor
    )

    df_val = construeix_manifest_faceapp(
        carpeta_ffhq / "validation",
        carpeta_faceapp / "validation",
        source_fake="faceapp",
        llavor=llavor
    )

    df_test = construeix_manifest_faceapp(
        carpeta_ffhq / "test",
        carpeta_faceapp / "test",
        source_fake="faceapp",
        llavor=llavor
    )

    return df_train, df_val, df_test