import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, BertTokenizerFast

from src.data_loader import (
    load_and_preprocess_data,
    preprocess_and_tokenize_data,
    create_data_loader,
)
from src.graph_builder import create_graph_kg2
from src.node2vec_embeddings import generate_node2vec_embeddings
from src.model import BERT_ABSA
from src.train import train_model
from src.evaluate import evaluate_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_path = "data/semeval 2014 - train.csv"
    test_path = "data/semeval 2014 - test.csv"

    train_df, test_df = load_and_preprocess_data(train_path, test_path)
    print("Train rows:", len(train_df))
    print("Test rows:", len(test_df))

    full_df = pd.concat([train_df, test_df], ignore_index=True)

    graph = create_graph_kg2(full_df)
    print("Graph nodes:", len(graph.nodes()))
    print("Graph edges:", len(graph.edges()))

    node_embeddings = generate_node2vec_embeddings(graph)
    print("Node embeddings generated:", len(node_embeddings))

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased")

    train_data = preprocess_and_tokenize_data(train_df, tokenizer, node_embeddings)
    test_data = preprocess_and_tokenize_data(test_df, tokenizer, node_embeddings)

    print("Processed train samples:", len(train_data))
    print("Processed test samples:", len(test_data))

    train_loader = create_data_loader(train_data, batch_size=32)
    test_loader = create_data_loader(test_data, batch_size=32)

    model = BERT_ABSA(bert_model=bert_model, num_classes=3).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    train_model(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=25,
    )

    evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/bert_absa_model.pt")
    print("Model saved to results/bert_absa_model.pt")


if __name__ == "__main__":
    main()
