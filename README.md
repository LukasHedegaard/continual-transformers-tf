# Continual Transformers TensorFlow
<div align="left">
  <!-- <a href="https://pypi.org/project/continual-transformers-tf/">
    <img src="https://img.shields.io/pypi/pyversions/continual-transformers-tf" height="20" >
  </a> -->
  <a href="http://arxiv.org/abs/2201.06268">
    <img src="http://img.shields.io/badge/paper-arxiv.2201.06268-B31B1B.svg" height="20" >
  </a>
  <!-- <a href="https://badge.fury.io/py/continual-transformers-tf">
    <img src="https://badge.fury.io/py/continual-transformers-tf.svg" height="20" >
  </a> -->
  <!-- <a href="https://pepy.tech/project/continual-transformers-tf">
    <img src="https://pepy.tech/badge/continual-transformers-tf/month" height="20">
  </a> -->
  <a href="https://codecov.io/gh/LukasHedegaard/continual-transformers-tf">
    <img src="https://codecov.io/gh/LukasHedegaard/continual-transformers-tf/branch/main/graph/badge.svg?token=Z01P5G5DWW"/>
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" height="20">
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" height="20">
  </a>
</div>

TensorFlow implementation of Continual Transformer building blocks, which augment regular transformer layers with the ability to compute the attention output _per token step_.

The layers are modelled on the `tf.keras.layers.MultiHeadAttention` and should work as drop-in replacements in most cases.

## Setup
Continual Transformers and its modules can be installed in in your project using:
```bash
pip install git+https://github.com/LukasHedegaard/continual-transformers-tf.git
```

## Layers
### [Continual Single-output Multi Head Attention](tests/test_co_si_mha.py)
```python
from continual_transformers_tf import CoSiMultiHeadAttention

layer = CoSiMultiHeadAttention(seq_len=10, num_heads=2, key_dim=4)
```

<div align="center">
  <img src="figures/CoSiDotProductAttention.png" width="500">
  <br>
  <div align="left">
  Fig. 1: Continual Single-Output Dot-Product Attention. 
        The key (K) and value (V) matrices are aggregated over time by caching the step vectors k_n and v_n in a FIFO queue. During each step, only the attention output associated with q is computed.
  </div>
  <br>
</div>


<!-- <div align="center">
  <img src="figures/CoReDotProductAttention.png" width="500">
  <br>
  <div align="left">
  Fig. 2: Continual Retroactive Dot-Product Attention. 
  The query (Q), key (K), and value (V) matrices are aggregated over time by caching the step vectors q_n, k_n, and v_n in a FIFO queue. During each step, only the entries of A associated with q_n, k_n, and the oldest K step, k_o are computed. 
  The diagonal entries of the row-normalisation matrix D as well as the AV can be updated retroactively by subtracting features corresponding to k_o and adding features related to k_n to the cached outputs of the previous step, D_{mem} and AV_{mem}, respectively.
  </div>
  <br>
</div> -->

### [Circular Positional Embedding](tests/test_circular_embedding.py)
```python
from continual_transformers_tf import CircularPositionalEncoding

layer = CircularPositionalEncoding(max_len=10, embed_dim=4)
```

<div align="center">
  <img src="figures/CircularPositionalEncoding.png" width="500">
  <br>
  <div align="left">
  Fig. 2: Circular Positional Encoding.
        At each step, a positional encoding is added in a round-robin fashion.
  </div>
  <br>
</div>

### [Continual Single-output Transformer Encoder](tests/test_co_si_trans_enc.py)
```python
from continual_transformers_tf import CoSiTransformerEncoder

layer = CoSiTransformerEncoder(
    seq_len=10,
    embed_dim=4,
    num_heads=2,
    ff_dim=16,
    dropout_rate=0.1,
)
```

## Citation   
```
@article{hedegaard2022cotrans,
  title={Continual Transformers: Redundancy-Free Attention for Online Inference},
  author={Lukas Hedegaard and Alexandros Iosifidis},
  journal={preprint, arXiv:2201.06268},
  year={2022}
}
```