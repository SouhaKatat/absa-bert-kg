
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


def main():
    # -----------------------------
    # 1. Device
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # 2. File paths
    # Replace these with your real dataset filenames
    # -----------------------------
    train_path = "data/semeval 2014 - train.csv"
    test_path = "data/semeval 2014 - test.csv"

    # -----------------------------
    # 3. Load data
    # -----------------------------
    train_df, test_df = load_and_preprocess_data(train_path, test_path)
    print("Train rows:", len(train_df))
    print("Test rows:", len(test_df))

    # -----------------------------
    # 4. Build graph
    # Using KG2 because your Colab code used it for Node2Vec training
    # -----------------------------

    import pandas as pd
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    graph = create_graph_kg2(full_df)
    print("Graph nodes:", len(graph.nodes()))
    print("Graph edges:", len(graph.edges()))

    # -----------------------------
    # 5. Generate Node2Vec embeddings
    # -----------------------------
    node_embeddings = generate_node2vec_embeddings(graph)
    print("Node embeddings generated:", len(node_embeddings))

    # -----------------------------
    # 6. Load tokenizer and BERT
    # -----------------------------
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased")

    # -----------------------------
    # 7. Preprocess dataset
    # -----------------------------
    train_data = preprocess_and_tokenize_data(train_df, tokenizer, node_embeddings)
    test_data = preprocess_and_tokenize_data(test_df, tokenizer, node_embeddings)

    print("Processed train samples:", len(train_data))
    print("Processed test samples:", len(test_data))

    # -----------------------------
    # 8. Create dataloaders
    # -----------------------------
    batch_size = 32

    train_loader = create_data_loader(train_data, batch_size=batch_size)
    test_loader = create_data_loader(test_data, batch_size=batch_size)

    # -----------------------------
    # 9. Build model
    # -----------------------------
    model = BERT_ABSA(bert_model=bert_model, num_classes=3).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # -----------------------------
    # 10. Train
    # -----------------------------
    epochs = 25
    train_model(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=epochs,
    )

    # -----------------------------
    # 11. Evaluate
    # -----------------------------
    evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    # -----------------------------
    # 12. Save trained model weights
    # -----------------------------
    torch.save(model.state_dict(), "results/bert_absa_model.pt")
    print("Model saved to results/bert_absa_model.pt")


if __name__ == "__main__":
    main()
