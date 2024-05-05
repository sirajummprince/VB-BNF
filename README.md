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
# Convert data to PyTorch tensors
<pre>
  <code>
    X = torch.tensor(data.values, dtype=torch.float32)
  </code>
</pre>
# # Implement VB-NMF model
<pre>
  <code>
    class VBNMF(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VBNMF, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.W = nn.Parameter(torch.randn(input_dim, latent_dim))
        self.H = nn.Parameter(torch.randn(latent_dim, input_dim))
        self.a = nn.Parameter(torch.randn(input_dim))
        self.b = nn.Parameter(torch.randn(input_dim))

    def forward(self, X, n_samples=1):
        KL_loss = 0
        recon_loss = 0
        for _ in range(n_samples):
            Q_W = torch.distributions.Normal(0, 1).sample(self.W.shape).to(X.device)
            Q_H = torch.distributions.Normal(0, 1).sample(self.H.shape).to(X.device)
            W_sample = self.W + Q_W
            H_sample = self.H + Q_H
            recon = torch.matmul(W_sample, H_sample)
            recon_loss += F.binary_cross_entropy_with_logits(recon, X, reduction='sum')
            KL_loss += 0.5 * torch.sum(self.W**2 + self.H**2 - 1 - self.a.log() - self.b.log())

        recon_loss /= n_samples
        KL_loss /= n_samples
        ELBO = recon_loss + KL_loss
        return ELBO
  </code>
</pre>
# Example usage
<pre>
  <code>
    input_dim = data.shape[1]
    latent_dim = 10
    model = VBNMF(input_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
  </code>
</pre>
# Training
<pre>
  <code>
    n_epochs = 100
    for epoch in range(n_epochs):
      optimizer.zero_grad()
      loss = model(X, n_samples=5)
      loss.backward()
      optimizer.step()
      print(f'Epoch {epoch:03d}, Loss {loss:.4f}')
  </code>
</pre>
