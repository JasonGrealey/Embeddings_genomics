# Embeddings_genomics
Machine learning embeddings of SNP data

This project is attempting to use feature extraction methods on genomics data. These features are extracted from single nucleotide polymorphisms from the genome. Where there are common varyations (mutations) in the human genome. These features usually heavily outweigh the number of samples (usually orders of magnitude) and so, I am working on feature extraction methods to tackle this problem whilst mainting predicability. 

The outcome that I am working on predicting are molecular traits. for example blood plasma metabolites.  

There are some methods of interest:
1. Denoising Autoencoders for lower dimensional latent variable representations /feature extractions of our SNPs.
2. PCA or a localised PCA.
3. UMAP (Uniform manifold Approximation and projection).

Currently I am testing these methods with real and simulated data.
