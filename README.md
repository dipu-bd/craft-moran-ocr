# CRAFT-MORAN-OCR

An OCR system using CRAFT for text detection and MORAN for recognition

## Inspiration

- [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch):
  Official Pytorch implementation of CRAFT text detector
  | [Paper](https://arxiv.org/abs/1904.01941)
  | [Pretrained Model](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ)
  | [Supplementary](https://youtu.be/HI8MzpY8KMI)

- [MORAN_v2](https://github.com/Canjie-Luo/MORAN_v2): MORAN is a network with rectification mechanism for general scene text recognition.
  The paper (accepted to appear in Pattern Recognition, 2019) in [arXiv](https://arxiv.org/abs/1901.03003),
  [final](https://www.sciencedirect.com/science/article/pii/S0031320319300263) version is available now.

## Testing

- Run `sh init.sh` to get started
- Run `python3 . scan/1.jpg` or `python3 . <your-image-path>`
- OCR results are stored in `output/` folder. You can change it in `__main__.py`
- Modify `train_MORAN.sh` to generated trained weights, and replace `moran_v2_demo.pth` with it
