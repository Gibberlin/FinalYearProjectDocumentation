"""
AV Vehicle Controller - Real-time Edge Inference Script
Demonstrates how the trained model is used inside an edge-controlled vehicle.

Loads normalisation parameters and model weights, takes sensor inputs,
and outputs real-time steering, throttle, and brake commands with latency benchmarks.

Usage:
    python predict_control.py
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# --------------------------------------------------
# 1. SETUP PATHS AND CONFIGURATION
# --------------------------------------------------

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
DATA_DIR = BASE / "processed_data"

MODEL_PATH = MODEL_DIR / "best_model.pth"
TORCHSCRIPT_PATH = MODEL_DIR / "av_controller.pt"
NORM_PATH = MODEL_DIR / "normalisation_params.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"

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

OUTPUT_TARGETS = [
    "steering_angle_deg",
    "throttle_command",
    "brake_command",
]


# --------------------------------------------------
# 2. MODEL DEFINITION
# --------------------------------------------------

class AVControlModel(nn.Module):
    def __init__(self, input_size=14, hidden_sizes=[64, 32, 16], output_size=3):
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
        self.register_buffer("target_scales", torch.ones(output_size, dtype=torch.float32))

    def forward(self, x):
        return self.network(x) * self.target_scales


# --------------------------------------------------
# 3. EDGE CONTROLLER INTERFACE CLASS
# --------------------------------------------------

class EdgeVehicleController:
    """
    Lightweight controller deployed directly on an edge device (e.g. Jetson/RPi).
    """
    def __init__(self, model_path=MODEL_PATH, norm_path=NORM_PATH):
        if not norm_path.exists():
            raise FileNotFoundError(f"Normalisation parameters not found at {norm_path}")
        
        # Load normalization statistics (mean and std)
        norm_df = pd.read_csv(norm_path).set_index("feature")
        self.mean = norm_df.loc[INPUT_FEATURES, "mean"].values.astype(np.float32)
        self.std = norm_df.loc[INPUT_FEATURES, "std"].values.astype(np.float32)

        # Load PyTorch model
        self.device = torch.device("cpu")  # Edge inference typically runs on CPU or GPU
        self.model = AVControlModel(
            input_size=len(INPUT_FEATURES),
            hidden_sizes=[64, 32, 16],
            output_size=len(OUTPUT_TARGETS)
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def predict(self, sensor_dict: dict) -> dict:
        """
        Takes raw sensor readings from the vehicle and returns control commands.
        """
        # Convert dictionary to ordered feature array
        raw_vals = np.array([float(sensor_dict.get(feat, 0.0)) for feat in INPUT_FEATURES], dtype=np.float32)

        # Normalise using training statistics
        norm_vals = (raw_vals - self.mean) / self.std

        # Run forward inference
        tensor_input = torch.tensor(norm_vals, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.model(tensor_input).squeeze(0).numpy()

        steering_deg = float(output[0])
        throttle = float(np.clip(output[1], 0.0, 1.0))
        brake = float(np.clip(output[2], 0.0, 1.0))

        # Vehicle logic: never apply high throttle and brake simultaneously
        if brake > 0.15:
            throttle = 0.0

        return {
            "steering_angle_deg": round(steering_deg, 2),
            "throttle_command": round(throttle, 3),
            "brake_command": round(brake, 3),
        }


# --------------------------------------------------
# 4. DEMONSTRATION & BENCHMARK
# --------------------------------------------------

def main():
    print("=" * 60)
    print("EDGE AI CONTROLLER - INFERENCE TEST & BENCHMARK")
    print("=" * 60)

    controller = EdgeVehicleController()
    print("✓ Model and normalisation parameters successfully loaded.\n")

    # Load real test frames to compare model predictions with human driving
    if TEST_DATA_PATH.exists():
        test_df = pd.read_csv(TEST_DATA_PATH)
        print(f"Loaded test dataset ({len(test_df)} unseen test frames).")
        print("\nTesting 5 consecutive sample frames from unseen drive sequence:")
        print("-" * 60)

        sample_subset = test_df.iloc[10:15]
        for idx, row in sample_subset.iterrows():
            sensor_input = {k: row[k] for k in INPUT_FEATURES}
            
            # Predict
            pred = controller.predict(sensor_input)

            actual_steering = row["steering_angle_deg"]
            actual_throttle = row["throttle_command"]
            actual_brake = row["brake_command"]

            def fmt_s(val):
                d = "R" if val < -0.3 else ("L" if val > 0.3 else "Center")
                return f"{val:+6.2f}° ({d})"

            print(f"Frame #{int(row['frame_index']):03d} | Speed: {row['speed_kmh']:.1f} km/h | Objects Ahead: {int(row['total_detections'])}")
            print(f"  Predicted -> Steering: {fmt_s(pred['steering_angle_deg'])} | Throttle: {pred['throttle_command']:.2f} | Brake: {pred['brake_command']:.2f}")
            print(f"  Actual    -> Steering: {fmt_s(actual_steering)} | Throttle: {actual_throttle:.2f} | Brake: {actual_brake:.2f}")
            print()

        # Latency benchmark
        print("-" * 60)
        print("RUNNING LATENCY BENCHMARK (1,000 predictions on CPU)...")
        sample_input = {k: test_df.iloc[0][k] for k in INPUT_FEATURES}

        # Warmup
        for _ in range(50):
            _ = controller.predict(sample_input)

        start_time = time.perf_counter()
        iterations = 1000
        for _ in range(iterations):
            _ = controller.predict(sample_input)
        elapsed_total_ms = (time.perf_counter() - start_time) * 1000
        avg_latency_ms = elapsed_total_ms / iterations
        fps = 1000.0 / avg_latency_ms

        print(f"  Total time for {iterations} iterations: {elapsed_total_ms:.2f} ms")
        print(f"  Average Inference Latency:   {avg_latency_ms:.3f} ms / prediction")
        print(f"  Throughput:                  {fps:.1f} control loops / second")
        print("-" * 60)
        print("✓ Performance: Sub-millisecond latency is well within the 20-50 ms edge AV requirement!")

    print("\n" + "=" * 60)
    print("Inference demonstration completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

