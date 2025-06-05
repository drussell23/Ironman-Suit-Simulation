"""
neural_controller.py

Feed-forward neural network autopilot for the Iron Man suit.
Includes inference, training utilities, and model persistence.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralController(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list = [128, 128, 64],
        output_size: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        last = input_size
        for h in hidden_sizes:
            layers += [
                nn.Linear(last, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            last = h
        layers.append(nn.Linear(last, output_size))
        self.network = nn.Sequential(*layers).to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.to(DEVICE))

    def predict(self, state: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            out = self.network(x)
        return out.cpu().numpy()

    def save(self, path: str):
        torch.save(self.network.state_dict(), path)
        logger.info(f"Saved model to {path}")

    def load(self, path: str):
        state = torch.load(path, map_location=DEVICE)
        self.network.load_state_dict(state)
        logger.info(f"Loaded model from {path}")

class ControllerDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ControllerTrainer:
    def __init__(
        self,
        model: NeuralController,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        step_size: int = 50,
        gamma: float = 0.5,
    ):
        self.model = model
        self.optimizer = optim.Adam(
            self.model.network.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma
        )

    def fit(
        self,
        train_dataset: ControllerDataset,
        val_dataset: ControllerDataset = None,
        epochs: int = 100,
        batch_size: int = 32,
    ):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

        for epoch in range(1, epochs+1):
            self.model.train()
            train_loss = 0.0
            
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                self.optimizer.zero_grad()
                preds = self.model.network(Xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            avg_train = train_loss / len(train_loader)
            logger.info(f"Epoch {epoch}/{epochs} - Train loss: {avg_train:.6f}")
            
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for Xv, yv in val_loader:
                        Xv, yv = Xv.to(DEVICE), yv.to(DEVICE)
                        val_loss += self.criterion(self.model.network(Xv), yv).item()
                avg_val = val_loss / len(val_loader)
                logger.info(f"Epoch {epoch}/{epochs} - Val loss: {avg_val:.6f}")
            self.scheduler.step()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Using device: {DEVICE}")