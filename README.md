<p align="center">
  <img src="image/Screenshot 2025-12-31 004953.png" width="600">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Softmax-Algorithms-blue?style=for-the-badge&logo=python">
</p>

# Softmax Algorithms

This repository contains Python implementations of multiple **Softmax algorithms**, starting from the naive formulation to **numerically stable** and **online (streaming)** variants.

<p align="center">
  <a href="https://arxiv.org/pdf/1805.02867">
    <img src="https://img.shields.io/badge/arXiv-1805.02867-B31B1B?style=for-the-badge&logo=arxiv">
  </a>
  <br>
  <i>Based on “Online Normalizer Calculation for Softmax”</i>
</p>

---

## 📌 Overview

The project implements a `softmax` class that demonstrates:

- Why naive softmax fails numerically
- How max-shifting stabilizes exponentials
- How normalization can be done **online**
- How **Top-K probabilities** can be extracted efficiently from streams

<p align="center">
  <img src="image/Screenshot 2025-12-31 001234.png" width="600">
  <img src="image/Screenshot 2025-12-31 001242.png" width="600">
  <img src="image/Screenshot 2025-12-31 001249.png" width="600">
  <img src="image/Screenshot 2025-12-31 001309.png" width="600">
  <br>
  <em>Figure: Safe Softmax using max-shift for numerical stability</em>
</p>

---

## ✨ Implemented Algorithms

- **Naive Softmax**
- **Safe Softmax (Max-Shift)**
- **Safe Softmax with Online Normalizer**
- **Online Softmax with Top-K Extraction**

> ⚠️ **Precision Note**
>
> This repository uses an approximate value of Euler’s number  
> `e = 2.718` **only for educational clarity**.
>
> 🚨 **For real systems**, always use:
> ```python
> math.exp(x)  # or numpy.exp(x)
> ```

---

## 💻 Core Implementation

```python
class Softmax:
    def naive_softmax(self, arr):
        exp_sum = 0.0
        for x in arr:
            exp_sum += 2.718 ** x
        return [(2.718 ** x) / exp_sum for x in arr]

    def safe_softmax(self, arr):
        m = max(arr)
        denom = 0.0
        for x in arr:
            denom += 2.718 ** (x - m)
        return [(2.718 ** (x - m)) / denom for x in arr]

    def online_softmax(self, arr):
        m = float("-inf")
        d = 0.0
        for x in arr:
            m_prev = m
            m = max(m, x)
            d = d * (2.718 ** (m_prev - m)) + (2.718 ** (x - m))
        return [(2.718 ** (x - m)) / d for x in arr]

    def online_top_k(self, arr, k):
        m = float("-inf")
        d = 0.0

        top_vals = [float("-inf")] * k
        top_idx = [-1] * k

        for i, x in enumerate(arr):
            m_prev = m
            m = max(m, x)
            d = d * (2.718 ** (m_prev - m)) + (2.718 ** (x - m))

            if x > top_vals[-1]:
                top_vals[-1] = x
                top_idx[-1] = i
                j = k - 1
                while j > 0 and top_vals[j] > top_vals[j - 1]:
                    top_vals[j], top_vals[j - 1] = top_vals[j - 1], top_vals[j]
                    top_idx[j], top_idx[j - 1] = top_idx[j - 1], top_idx[j]
                    j -= 1

        probs = [(2.718 ** (x - m)) / d for x in top_vals]
        return probs, top_idx

