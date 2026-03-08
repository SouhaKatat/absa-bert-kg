def train_model(model, dataloader, optimizer, criterion, device, epochs):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            input_ids, attention_mask, node_embeddings, labels = [
                b.to(device) for b in batch
            ]

            optimizer.zero_grad()

            outputs = model(
                input_ids,
                attention_mask,
                node_embeddings
            )

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
