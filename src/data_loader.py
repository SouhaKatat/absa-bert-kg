
import pandas as pd
import torch
import ast
import numpy as np

from torch.utils.data import TensorDataset, DataLoader, RandomSampler


def load_and_preprocess_data(train_path, test_path):

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df['aspectTerms'] = train_df['aspectTerms'].apply(ast.literal_eval)
    test_df['aspectTerms'] = test_df['aspectTerms'].apply(ast.literal_eval)

    return train_df, test_df


def preprocess_and_tokenize_data(df, tokenizer, node_embeddings, max_length=128):

    processed_data = []

    for _, row in df.iterrows():

        raw_text = row['raw_text']
        aspect_terms = row['aspectTerms']

        for aspect in aspect_terms:

            tokenized_text = tokenizer(
                raw_text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            aspect_term = aspect['term']

            node2vec_embedding = torch.tensor(
                node_embeddings.get(aspect_term, np.zeros(300)),
                dtype=torch.float32
            )

            processed_data.append(
                (
                    tokenized_text['input_ids'][0],
                    tokenized_text['attention_mask'][0],
                    node2vec_embedding,
                    aspect['polarity']
                )
            )

    return processed_data


def create_data_loader(processed_data, batch_size):

    input_ids = torch.stack([x[0] for x in processed_data])
    attention_masks = torch.stack([x[1] for x in processed_data])
    node_embeddings = torch.stack([x[2] for x in processed_data])

    labels = torch.tensor(
        [0 if x[3] == "negative" else 1 if x[3] == "positive" else 2 for x in processed_data]
    )

    dataset = TensorDataset(input_ids, attention_masks, node_embeddings, labels)

    sampler = RandomSampler(dataset)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    return dataloader
