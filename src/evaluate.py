
import torch
from sklearn.metrics import classification_report


def evaluate_model(model, dataloader, device):

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

    print(classification_report(labels, preds))
