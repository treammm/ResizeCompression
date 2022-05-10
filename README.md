# ResizeCompression

This repository is the implementation of paper: Li-Heng Chen, Christos G. Bampis, Zhi Li, Lukáš Krasula and Alan C. Bovik, "[Estimating the Resize Parameter in End-to-end Learned Image Compression](https://arxiv.org/abs/2204.12022)"

## Environment
- python3.7.x
- Tensorflow==1.15.0
### Dependencies
1. [Tensorflow Compression](https://github.com/tensorflow/compression)
```
pip install tensorflow-compression==1.3
```
Or just do
```
pip install -r requirements.txt
```

## Colab demo
You may also load the notebook `xxx.ipynp` to Google Colab from this GitHub repository: 
1. Navigate to http://colab.research.google.com/github.
2. Enter the path `TBD` in the "Enter a GitHub URL or search by organization or user" field
3. Start Colabing!

## Work in progress
1. ~Network and warping utilities~
2. Training code of bls2017 +Resize-Compress
3. Test code (encode/decode) of bls2017 +Resize-Compress
4. Colab demo
5. ~Subjective study data~
