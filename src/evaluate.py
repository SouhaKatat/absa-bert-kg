import os
import json
import torch
from sklearn.metrics import classification_report


def evaluate_model(model, dataloader, device, save_path="results/metrics.json"):
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, node_embeddings, y = [
                b.to(device) for b in batch
            ]

            outputs = model(
                input_ids,
                attention_mask,
                node_embeddings
            )

            _, predicted = torch.max(outputs, dim=1)

            preds.extend(predicted.cpu().numpy())
            labels.extend(y.cpu().numpy())

    report_dict = classification_report(labels, preds, output_dict=True)
    report_text = classification_report(labels, preds)

    print(report_text)

    os.makedirs("results", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    return report_dict
