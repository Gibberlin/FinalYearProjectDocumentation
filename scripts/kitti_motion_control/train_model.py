"""
AV Control Model — Training Script (Balanced Multi-Target Loss)
Trains a neural network to accurately predict steering, throttle, and brake
using target normalization to prevent steering loss from dominating pedal controls.

Usage: python train_model.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "processed_data"
PLOT_DIR = BASE / "visualizations"
MODEL_DIR = BASE / "model"

MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# Which columns the AI sees (inputs)
INPUT_FEATURES = [
    "speed_kmh",
    "forward_speed_mps",
    "heading_deg",
    "yaw_rate_rad_s",
    "forward_acceleration_mps2",
    "lateral_acceleration_mps2",
    "curvature_m_inv",
    "persons_detected",
    "bicycles_detected",
    "cars_detected",
    "trucks_detected",
    "buses_detected",
    "traffic_lights_detected",
    "total_detections",
]

# What the AI must predict (outputs)
OUTPUT_TARGETS = [
    "steering_angle_deg",
    "throttle_command",
    "brake_command",
]

# Training hyperparameters
EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 0.003
HIDDEN_SIZES = [64, 32, 16]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================
# 2. LOAD AND PREPARE DATA
# ==================================================

print("=" * 55)
print("STEP 1: Loading data...")
print("=" * 55)

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "val.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

print(f"  Train: {len(train_df)} frames")
print(f"  Val:   {len(val_df)} frames")
print(f"  Test:  {len(test_df)} frames")

# Extract input features and output targets
X_train = train_df[INPUT_FEATURES].values.astype(np.float32)
y_train = train_df[OUTPUT_TARGETS].values.astype(np.float32)

X_val = val_df[INPUT_FEATURES].values.astype(np.float32)
y_val = val_df[OUTPUT_TARGETS].values.astype(np.float32)

X_test = test_df[INPUT_FEATURES].values.astype(np.float32)
y_test = test_df[OUTPUT_TARGETS].values.astype(np.float32)


# ==================================================
# 3. NORMALISE FEATURES AND TARGETS
# ==================================================

print("\nSTEP 2: Normalising features and target scales...")

# Input feature normalization (zero-mean, unit-variance)
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
train_std[train_std < 1e-6] = 1.0

X_train = (X_train - train_mean) / train_std
X_val = (X_val - train_mean) / train_std
X_test = (X_test - train_mean) / train_std

# Target scale normalization:
# Prevents large steering numbers (+/- 30 deg) from drowning out
# small throttle/brake numbers (0.0 to 1.0) in the MSE loss function!
target_scales = y_train.std(axis=0)
target_scales[target_scales < 1e-6] = 1.0

print(f"  Target scale factors (std):")
for name, scale in zip(OUTPUT_TARGETS, target_scales):
    print(f"    {name:25s}: {scale:.4f}")

y_train_norm = y_train / target_scales
y_val_norm = y_val / target_scales

# Save normalization parameters
norm_params = pd.DataFrame({
    "feature": INPUT_FEATURES,
    "mean": train_mean,
    "std": train_std,
})
norm_path = MODEL_DIR / "normalisation_params.csv"
norm_params.to_csv(norm_path, index=False)
print(f"  Saved input normalisation params: {norm_path}")

target_df = pd.DataFrame({
    "target": OUTPUT_TARGETS,
    "scale": target_scales,
})
target_df.to_csv(MODEL_DIR / "target_scales.csv", index=False)


# ==================================================
# 4. PREPARE DATALOADERS
# ==================================================

print("\nSTEP 3: Preparing PyTorch DataLoaders...")

train_dataset = TensorDataset(
    torch.tensor(X_train),
    torch.tensor(y_train_norm),
)
val_dataset = TensorDataset(
    torch.tensor(X_val),
    torch.tensor(y_val_norm),
)
test_dataset = TensorDataset(
    torch.tensor(X_test),
    torch.tensor(y_test),  # test keeps physical units
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ==================================================
# 5. DEFINE THE NEURAL NETWORK
# ==================================================

class AVControlModel(nn.Module):
    """
    Lightweight Multi-Target Vehicle Control Model.
    Automatically multiplies internal outputs by target_scales so that
    inference directly outputs calibrated physical units (deg, [0,1]).
    """
    def __init__(self, input_size, hidden_sizes, output_size, target_scales=None):
        super().__init__()
        layers = []
        prev_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.05))
            prev_size = h_size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

        if target_scales is not None:
            self.register_buffer("target_scales", torch.tensor(target_scales, dtype=torch.float32))
        else:
            self.register_buffer("target_scales", torch.ones(output_size, dtype=torch.float32))

    def forward_norm(self, x):
        """Returns normalized predictions (used during training)."""
        return self.network(x)

    def forward(self, x):
        """Returns physical control predictions (used during inference)."""
        return self.network(x) * self.target_scales


model = AVControlModel(
    input_size=len(INPUT_FEATURES),
    hidden_sizes=HIDDEN_SIZES,
    output_size=len(OUTPUT_TARGETS),
    target_scales=target_scales,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n  Model architecture: {len(INPUT_FEATURES)} → "
      f"{' → '.join(str(h) for h in HIDDEN_SIZES)} → {len(OUTPUT_TARGETS)}")
print(f"  Total parameters: {total_params:,}")


# ==================================================
# 6. LOSS AND OPTIMIZER
# ==================================================

criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", patience=15, factor=0.5)


# ==================================================
# 7. TRAINING LOOP
# ==================================================

print("\n" + "=" * 55)
print("STEP 4: Training the model...")
print("=" * 55)

train_losses = []
val_losses = []
best_val_loss = float("inf")
best_epoch = 0

for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        predictions_norm = model.forward_norm(X_batch)
        loss = criterion(predictions_norm, y_batch)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        epoch_train_loss += loss.item() * len(X_batch)

    epoch_train_loss /= len(train_dataset)
    train_losses.append(epoch_train_loss)

    # Validation
    model.eval()
    epoch_val_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            preds_norm = model.forward_norm(X_batch)
            loss = criterion(preds_norm, y_batch)
            epoch_val_loss += loss.item() * len(X_batch)

    epoch_val_loss /= len(val_dataset)
    val_losses.append(epoch_val_loss)
    scheduler.step(epoch_val_loss)

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")

    if (epoch + 1) % 15 == 0 or epoch == 0:
        lr = optimiser.param_groups[0]["lr"]
        print(f"  Epoch {epoch + 1:3d}/{EPOCHS} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | LR: {lr:.6f}")

print(f"\n  Best validation loss: {best_val_loss:.6f} at epoch {best_epoch + 1}")


# ==================================================
# 8. EVALUATE ON TEST SET
# ==================================================

print("\n" + "=" * 55)
print("STEP 5: Evaluating on test set...")
print("=" * 55)

model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", weights_only=True))
model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        # Direct call to model(X_batch) outputs calibrated physical values!
        preds = model(X_batch).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y_batch.numpy())

predictions = np.concatenate(all_preds)
targets = np.concatenate(all_targets)

print("\n  Per-output Performance on Unseen Test Sequences:")
for i, name in enumerate(OUTPUT_TARGETS):
    mae = np.mean(np.abs(predictions[:, i] - targets[:, i]))
    rmse = np.sqrt(np.mean((predictions[:, i] - targets[:, i]) ** 2))
    ss_res = np.sum((targets[:, i] - predictions[:, i]) ** 2)
    ss_tot = np.sum((targets[:, i] - targets[:, i].mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    unit = "°" if "deg" in name else ""
    print(f"    {name:25s} | MAE: {mae:.4f}{unit}  RMSE: {rmse:.4f}{unit}  R²: {r2:.4f}")


# ==================================================
# 9. SAVE TRAINING CURVES PLOT
# ==================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, label="Train Loss", linewidth=1.2)
axes[0].plot(val_losses, label="Validation Loss", linewidth=1.2)
axes[0].axvline(best_epoch, color="red", linestyle="--", alpha=0.5, label=f"Best Epoch ({best_epoch + 1})")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Normalized MSE Loss")
axes[0].set_title("Training & Validation Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale("log")

axes[1].scatter(targets[:, 1], np.clip(predictions[:, 1], 0, 1), alpha=0.4, color="tab:green", label="Throttle")
axes[1].scatter(targets[:, 2], np.clip(predictions[:, 2], 0, 1), alpha=0.4, color="tab:red", label="Brake")
axes[1].plot([0, 1], [0, 1], "k--", label="Perfect Fit")
axes[1].set_xlabel("Actual Command [0, 1]")
axes[1].set_ylabel("Predicted Command [0, 1]")
axes[1].set_title("Pedal Prediction Accuracy (Test Set)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = PLOT_DIR / "training_results.png"
plt.savefig(plot_path, dpi=200)
plt.close()
print(f"\n  Training plot saved: {plot_path}")


# ==================================================
# 10. EXPORT FOR EDGE DEPLOYMENT
# ==================================================

print("\n" + "=" * 55)
print("STEP 6: Exporting model for deployment...")
print("=" * 55)

model.eval()
dummy_input = torch.randn(1, len(INPUT_FEATURES)).to(DEVICE)

# TorchScript export
scripted_model = torch.jit.trace(model, dummy_input)
script_path = MODEL_DIR / "av_controller.pt"
scripted_model.save(str(script_path))
print(f"  TorchScript model: {script_path}")

# ONNX export
onnx_path = MODEL_DIR / "av_controller.onnx"
try:
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["sensor_inputs"],
        output_names=["control_outputs"],
        dynamic_axes={"sensor_inputs": {0: "batch_size"}, "control_outputs": {0: "batch_size"}},
        opset_version=18,
    )
    print(f"  ONNX model:        {onnx_path}")
except Exception as e:
    print(f"  ONNX export notice: {e}")

print("\nRetraining and export completed successfully!")
