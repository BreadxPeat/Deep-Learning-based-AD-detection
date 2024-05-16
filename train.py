# Define train function
import model
def train(num_epochs, model, optimizer, loss_fn):
    # Start training
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader))
        for i, (x, y) in loop:
            # Forward pass
            x = x.to(device).view(-1, INPUT_DIM)
            x_reconst, mu, sigma = model(x)

            # loss, formulas from https://www.youtube.com/watch?v=igP03FXZqgo&t=2182s
            reconst_loss = loss_fn(x_reconst, x)
            kl_div = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            # Backprop and optimize
            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
            
            
# Initialize model, optimizer, loss
model = VariationalAutoEncoder(784, 20).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.BCELoss(reduction="sum")

# Run training
train(NUM_EPOCHS, model, optimizer, loss_fn)