# Embeddings_genomics
Machine learning for embeddings and dimension reduction of SNP (single nucleotide polymorphisms) data to be used for improved prediction of molecular traits.

This project is attempting to use feature extraction methods on genomics data. These features are extracted from single nucleotide polymorphisms from the genome. Where there are common variations (mutations) in the human genome. These variations are measured using genotyping arrays and are used as a basis of understanding genomic effects on many phenotypes, including disease. 

However in experiments involving genotypes, often the number of SNPs sampled (features) greatly exceeds the number of samples. Meaning that predicting with these features falls into the large P small N problem. It is common to use linear methods to select features of importance. Although, I want to know if more nonlinear methods can perform better by taking into account non-linear interactions amongst and combinations of the features.

I have been working on dimension reduction methods like PCA (Principal Component Analysis), UMAP (Uniform Manifold Approximation and Projection), and Denoising autoencoders to tackle this problem whilst mainting predicability. 

As well as this I am looking at using feature embedding to incorporate as sense of relatability between features. One would expect that due to the heavy correlation structure in the genome (linkage disequilibrium), that emebeddings could take account of this. I.e. SNPs in high LD (correlation) with each other would likely be closer together in the embedded space. This allows the prediction algorithm to have more useful information about the SNPs rather than just their values (i.e. how many mutations are present in the sample for a given SNP).

The outcome that I am working on predicting are molecular traits. for example blood plasma metabolites.  

There are some methods of interest:
1. Denoising Autoencoders for lower dimensional latent variable representations /feature extractions of our SNPs.
2. PCA or a localised PCA.
3. UMAP (Uniform manifold Approximation and projection).

Currently I am testing these methods with real and simulated data.


