import onnx

model = onnx.load('pwa/lstm_model.onnx')
onnx.save_model(model, 'pwa/lstm_model.onnx', save_as_external_data=False, all_tensors_to_one_file=True)
print("Saved unified model!")
