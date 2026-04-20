import torch
import torch.nn as nn
from app import LSTMModel

model = LSTMModel(input_size=14, hidden=128, layers=2)
model.load_state_dict(torch.load('outputs/models/lstm_best.pt', map_location='cpu'))
model.eval()

# Dummy input corresponding to [batch_size, sequence_length, num_features]
dummy_input = torch.randn(1, 168, 14)

# EXPORT
torch.onnx.export(
    model, 
    dummy_input,
    'pwa/lstm_model.onnx',
    export_params=True,
    opset_version=14,  # Downgraded to 14 to avoid external weights
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("Exported successfully to pwa/lstm_model.onnx")
