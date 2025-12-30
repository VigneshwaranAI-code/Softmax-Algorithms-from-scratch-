![Project Banner](path/to/your/banner_image.png)

<p align="center">
  <img src="https://img.shields.io/badge/Softmax-Algorithms-blue?style=for-the-badge&logo=python" alt="Project Logo">
</p>

# Softmax Algorithms

This repository contains Python implementations of various Softmax algorithms, ranging from the naive approach to numerically stable and online versions.

<p align="center">
  <a href="https://arxiv.org/pdf/1805.02867">
    <img src="https://img.shields.io/badge/arXiv-1805.02867-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="Research Paper">
  </a>
  <br>
  <i>Based on "Online Normalizer Calculation for Softmax"</i>
</p>

---

## 📌 Overview

The project implements a `softmax` class with methods to handle standard calculations, numerical stability, and streaming data.

<p align="center">
  <img src="image/Screenshot 2025-12-31 001234.png" width="600">
  <img src="image/Screenshot 2025-12-31 001242.png" width="600">
  <img src="image/Screenshot 2025-12-31 001249.png" width="600">
  <img src="image/Screenshot 2025-12-31 001234.png" width="600">
  <br>
  <em>Figure 1: Visual representation of the Safe Softmax shifting logic.</em>
</p>

Key Features:
* **Standard Softmax** (Naive implementation)
* **Numerical Stability** (Handling overflow/underflow)
* **Dynamic Normalization**
* **Online / Streaming Data** (Top-K elements)

> **⚠️ Important Note on Precision:**
> The code in this repository is experimental and currently uses an approximate value for Euler's number (`e = 2.718`) for demonstration purposes.
> For production environments, it is **strongly recommended** to use `numpy.exp()` or `math.exp()`.

## 💻 Implementation

Below is the core code provided in `Softmax_Algorithms.ipynb`.

```python
class softmax:
    def softmax_navie(self, arr: list[float]) -> list[float]:
        self.n = len(arr)
        self.exp_vlaues = 0
        
        # Naive implementation
        for i in range(self.n):
            exp = 2.718 ** arr[i]
            self.exp_vlaues += exp
            
        return [(2.718 ** arr[j] / self.exp_vlau
