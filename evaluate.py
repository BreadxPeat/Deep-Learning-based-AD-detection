# Evaluate the classifier
classifier.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        mu, _ = encoder(data.view(data.size(0), -1))  # Flatten input data
        outputs = classifier(mu)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")