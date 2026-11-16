"""Quick script to export the trained model to ONNX format."""
import torch
import torch.nn as nn
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "model"

INPUT_FEATURES = [
    "speed_kmh", "forward_speed_mps", "heading_deg", "yaw_rate_rad_s",
    "forward_acceleration_mps2", "lateral_acceleration_mps2", "curvature_m_inv",
    "persons_detected", "bicycles_detected", "cars_detected",
    "trucks_detected", "buses_detected", "traffic_lights_detected", "total_detections",
]
OUTPUT_TARGETS = ["steering_angle_deg", "throttle_command", "brake_command"]
HIDDEN_SIZES = [64, 32, 16]

class AVControlModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
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

model = AVControlModel(len(INPUT_FEATURES), HIDDEN_SIZES, len(OUTPUT_TARGETS))
model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", weights_only=True))
model.eval()

dummy = torch.randn(1, len(INPUT_FEATURES))
onnx_path = MODEL_DIR / "av_controller.onnx"

torch.onnx.export(
    model, dummy, str(onnx_path),
    input_names=["sensor_inputs"],
    output_names=["control_outputs"],
    dynamic_axes={"sensor_inputs": {0: "batch_size"}, "control_outputs": {0: "batch_size"}},
    opset_version=18,
)

pt_path = MODEL_DIR / "av_controller.pt"
print(f"ONNX exported:    {onnx_path}")
print(f"ONNX size:        {onnx_path.stat().st_size / 1024:.1f} KB")
print(f"TorchScript size: {pt_path.stat().st_size / 1024:.1f} KB")
