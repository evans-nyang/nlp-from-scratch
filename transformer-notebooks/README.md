# Transfomers

This repository contains notebooks that demonstrate how to use the `transformers` library from Huggingface. The notebooks are based on the tutorials from the official documentation of the `transformers` library.

## Issues

When using the `transformers` library, you may encounter `keras` library error since `transformers` uses `tf.keras` which is different from `keras`. To fix this, you can install `tf-keras` library which is a wrapper around `tf.keras` and `keras` libraries.

```sh
pip install tf-keras
```
