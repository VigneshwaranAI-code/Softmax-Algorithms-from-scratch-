# Softmax Algorithms

This repository contains Python implementations of various Softmax algorithms, ranging from the naive approach to numerically stable and online versions.

## 📌 Overview

The project implements a `softmax` class with methods to handle:
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
            
        return [(2.718 ** arr[j] / self.exp_vlaues) for j in range(self.n)]
    
    def saft_softmax(self, arr: list[float]) -> list[float]:
        self.n = len(arr)
        self.max = max(arr)
        self.denameter = 0
        
        for i in range(self.n):
            # Subtracting max for stability
            self.denameter += 2.718 ** (arr[i] - self.max)   
        return [(2.718 ** (arr[j] - self.max) / self.denameter) for j in range(self.n)]
    
    def safe_softmax_with_norm(self, arr: list[float]) -> list[float]:
        self.n = len(arr)
        self.denameter = 0
        
        m = float("-inf")
        d = 0
        for i in range(self.n):
            m_prov = m 
            m = max(m_prov, arr[i])
            correction = 2.718 ** (m_prov - m)
            new_term = 2.718 ** (arr[i] - m)
            d = (d * correction) + new_term 
        return [(2.718**(arr[j] - m) / d) for j in range(self.n)]
    
    def online_with_top_K(self, arr: list[float], k: int):
        # ... (Implementation for streaming data)
        # See full code in Softmax_Algorithms.ipynb
        pass
