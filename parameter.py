device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = 784
Z_DIM = 20
H_DIM = 200
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR_RATE = 3e-4
