libdeep is a PyTorch-like deep learning library written in C++ and Python with: automatic differentiation; gradient-based optimization of models; support for standard
operators like convolutions, recurrent structure, self-attention; and efficient linear algebra on both CPU and GPU devices.

## Setup

Create a virtual environment
```
$ python3 -m venv .venv 
$ source .venv/bin/activate  
```

Install the libraries required

```
$ pip3 install pybind11 numdifftools numpy pytest
```

## Tests

```
python3 -m pytest
```