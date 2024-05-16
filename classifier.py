import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, num_classes=2):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Assuming you have instantiated the VAE model
vae = VariationalAutoEncoder(input_dim=50176, z_dim=2)

# Extract the encoder part of the VAE model
encoder = vae.encode

# Assuming you have defined your training data loader and optimizer
# train_loader = ...
# optimizer = ...

# Define the classifier model on top of the encoder
classifier = Classifier(input_dim=2)  # Assuming the encoder output dimension is 200

# Define your loss function
criterion = nn.CrossEntropyLoss()

# Train the classifier
epochs = 5
for epoch in range(epochs):
    classifier.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        mu, _ = encoder(data.view(data.size(0), -1))  # Flatten input data
        output = classifier(mu)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


