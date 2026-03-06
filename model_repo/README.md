# model_repo

Triton model repository for RAInder.

## Structure

- `real_esrgan_x2/config.pbtxt`
- `real_esrgan_x2/1/model.onnx`
- `real_esrgan_x4/config.pbtxt`
- `real_esrgan_x4/1/model.onnx`
- `anime_gan_hayao/config.pbtxt`
- `anime_gan_hayao/1/model.onnx` (symlink to `../AnimeGANv2_Hayao.onnx`)

## Notes

- Current setup exposes two model names in Triton:
 - Current setup exposes model names in Triton:
  - `real_esrgan_x2`
  - `real_esrgan_x4`
  - `anime_gan_hayao`
- Keep one model artifact per model directory/version for Triton compatibility.
