# Variational Bayesian Non-Negative Matrix Factorization
Variational Bayesian Non-Negative Matrix Factorization
also abbreviated as VB-NMF is a way to
factorize a non-negative matrix, X, into the product
of two low rank matrices, W & H in a way
that WH predicts an optimal solution for X. In
this report, VB-NMF is explored as a project for
the course COSC 5P77 Probabilistic Graphical
Models and Neural Generative Models, which is
supervised by Dr. Yifeng Li, Assistant Professor,
Department of Computer Science, Brock University.

# Import Libraries
  <pre>
    <code>
        import pandas as pd
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from google.colab import files
    </code>
    
  </pre>

