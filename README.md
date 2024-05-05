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
        import io
        from google.colab import files
    </code> 
  </pre>
# Upload Files to Colab from Local Machine
<pre>
    <code>
        uploaded = files.upload
    </code> 
  </pre>
# Load the Data
<pre>
    <code>
        data = pd.read_csv(io.BytesIO(uploaded['dataFilt.csv'], index_col=0))
        pathways = pd.read_csv(io.BytesIO(uploaded['kegg_legacy_ensembl.csv'], index_col=0))
        sample_classes = pd.read_csv(io.BytesIO(uploaded['sampletype.csv'], index_col=0))
    </code> 
  </pre>
