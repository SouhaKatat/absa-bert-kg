import os
import json
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def evaluate_model(model, dataloader, device, save_dir="results"):
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, node_embeddings, y = [
                b.to(device) for b in batch
            ]

            outputs = model(input_ids, attention_mask, node_embeddings)
            _, predicted = torch.max(outputs, dim=1)

            preds.extend(predicted.cpu().numpy())
            labels.extend(y.cpu().numpy())

    os.makedirs(save_dir, exist_ok=True)

    report_dict = classification_report(labels, preds, output_dict=True)
    report_text = classification_report(labels, preds)
    print(report_text)

    with open(f"{save_dir}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(f"{save_dir}/confusion_matrix.png", bbox_inches="tight")
    plt.close()

    return report_dict
