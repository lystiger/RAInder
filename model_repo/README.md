# model_repo

Triton model repository for RAInder.

## Structure

- `real_esrgan_x2/config.pbtxt`
- `real_esrgan_x2/1/model.onnx`
- `real_esrgan_x4/config.pbtxt`
- `real_esrgan_x4/1/model.onnx`

## Notes

- Current setup exposes two model names in Triton:
  - `real_esrgan_x2`
  - `real_esrgan_x4`
- Keep one model artifact per model directory/version for Triton compatibility.
