import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc


def entrena_una_epoca(model, dataloader, optimizer, criterion, device):
    model.train()

    perdua_total = 0.0

    for batch in dataloader:
        x0 = batch["x0"].to(device)
        magnitud = batch["magnitud"].to(device)
        fase = batch["fase"].to(device)
        # afegim una dimensio mes perque perque la loss pugui comparar correctament
        # logits i etiquetes
        etiqueta = batch["etiqueta"].to(device).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(x0, magnitud, fase)
        perdua = criterion(logits, etiqueta)

        perdua.backward()
        optimizer.step()

        perdua_total += perdua.item()

    return perdua_total / len(dataloader)


@torch.no_grad()
def avalua(model, dataloader, criterion, device):
    model.eval()

    perdua_total = 0.0
    y_true = []
    y_prob = []
    
    for batch in dataloader:
        x0 = batch["x0"].to(device)
        magnitud = batch["magnitud"].to(device)
        fase = batch["fase"].to(device)
        etiqueta = batch["etiqueta"].to(device).unsqueeze(1)

        logits = model(x0, magnitud, fase)
        perdua = criterion(logits, etiqueta)

        probs = torch.sigmoid(logits)

        perdua_total += perdua.item()
        y_true.extend(etiqueta.cpu().numpy().ravel().tolist())
        y_prob.extend(probs.cpu().numpy().ravel().tolist())

    metriques = calcula_metriques(y_true, y_prob)

    return perdua_total / len(dataloader), metriques


def calcula_metriques(y_true, y_prob, llindar=0.5):
    y_pred = [1 if p >= llindar else 0 for p in y_prob]

    resultat = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    try:
        resultat["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        resultat["auc"] = None

    return resultat

@torch.no_grad()
def recull_prediccions(model, dataloader, device, llindar=0.5):
    model.eval()

    y_true = []
    y_prob = []

    for batch in dataloader:
        x0 = batch["x0"].to(device)
        magnitud = batch["magnitud"].to(device)
        fase = batch["fase"].to(device)
        etiqueta = batch["etiqueta"].to(device).unsqueeze(1)

        logits = model(x0, magnitud, fase)
        probs = torch.sigmoid(logits)

        y_true.extend(etiqueta.cpu().numpy().ravel().tolist())
        y_prob.extend(probs.cpu().numpy().ravel().tolist())

    y_pred = [1 if p >= llindar else 0 for p in y_prob]

    return y_true, y_prob, y_pred


def dibuixa_historic(historic, titol="Historic d'entrenament",ruta=None):
    epoques = range(1, len(historic["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epoques, historic["train_loss"], label="Train loss")
    plt.plot(epoques, historic["val_loss"], label="Val loss")
    plt.xlabel("Epoca")
    plt.ylabel("Perdua")
    plt.title(f"{titol} - Perdua")
    plt.legend()
    plt.grid(True)
    if ruta is not None:
        ruta_perdua = ruta / "perdua.png"
        plt.savefig(ruta_perdua)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epoques, historic["val_accuracy"], label="Val accuracy")
    plt.plot(epoques, historic["val_f1"], label="Val F1")
    plt.plot(epoques, historic["val_auc"], label="Val AUC")
    plt.xlabel("Epoca")
    plt.ylabel("Valor")
    plt.title(f"{titol} - Metriques")
    plt.legend()
    plt.grid(True)
    if ruta is not None:
        ruta_metrica = ruta / "metriques.png"
        plt.savefig(ruta_metrica)
    plt.show()


def dibuixa_matriu_confusio(y_true, y_pred, titol="Matriu de confusio",ruta=None):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 5))
    plt.imshow(cm)
    plt.title(titol)
    plt.xlabel("Prediccio")
    plt.ylabel("Etiqueta real")
    plt.xticks([0, 1], ["Real", "Fake"])
    plt.yticks([0, 1], ["Real", "Fake"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    if ruta is not None:
        ruta_matriu = ruta / "Matriu de confusio.png"
        plt.savefig(ruta_matriu)
    plt.show()


def dibuixa_corba_roc(y_true, y_prob, titol="Corba ROC",ruta=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(titol)
    plt.legend()
    plt.grid(True)
    if ruta is not None:
        ruta_corba = ruta / "Corba ROC.png"
        plt.savefig(ruta_corba)
    plt.show()


def dibuixa_comparacio_metodes(df_resultats, metrica="f1", titol=None, ruta=None):
    df_ordenat = df_resultats.sort_values(by=metrica, ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(df_ordenat["metode"], df_ordenat[metrica])
    plt.xlabel("Metode")
    plt.ylabel(metrica.upper())
    if titol is None:
        titol = f"Comparacio de metodes segons {metrica}"
    plt.title(titol)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if ruta is not None:
        plt.savefig(ruta)
    plt.show()