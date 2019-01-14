#  Embeddings genomics  #

This work is being carried out inside the  Cambridge-Baker Systems Genomics Initiative (CBSGI - https://sysgenresearch.org/).


The CBSGI is an international research partnership between the University of Cambridge (UK - https://www.cam.ac.uk/) and the Baker Heart & Diabetes Institute (Australia - https://baker.edu.au/) which aims to:
* Drive the development of next-generation analytics.
* Uncover biological insights from multi-omic datasets.
* Build clinically useful tools to predict and prevent disease.

### Machine Learning Techniques for Genomic Prediction ###

I am interested in applying and developing machine learning techniques for embedding or dimension reduction of SNP (single nucleotide polymorphisms) data to be used for improved prediction of molecular traits. SNPs are common variations (mutations) in the human genome and are generally used as the features for prediction. These SNPs are measured using genotyping arrays and are used as a basis of understanding genomic effects on many phenotypes, including disease. 

However, in experiments involving genotypes, often the number of SNPs sampled (features) greatly exceeds the number of samples. Thanks to imputation (inferring values of highly correlated but not directly measured SNPs), the number of SNPs can be in the millions. Note that imputation is commonplace in genomics research thanks to the strong correlation structure inside the genome, that is linkage disequilibrum, essentailly correlations between two SNPs is a function of distance and generally decreases with distance between them. The efficiency of Genotype assays allows up to half a million SNPs to be measured per experiment, then with imputation on top of this, an experiment can easily produce millions of SNPs. Meaning that these datasets often suffer from the large P small N problem as it is difficult (expensive and logistically problematic) to gather millions of independent samples. 

It is common to use linear methods to select features of importance. Although, I want to know if more nonlinear methods can perform better by taking into account non-linear interactions amongst and combinations of the features (SNPs).

I have been working on dimension reduction methods like PCA (Principal Component Analysis), UMAP (Uniform Manifold Approximation and Projection), and Denoising autoencoders to tackle this problem whilst mainting predicability. 

As well as this I am looking at using feature embedding to incorporate as sense of relatability between features. One would expect that due to the heavy correlation structure in the genome (linkage disequilibrium), that emebeddings could take account of this. I.e. SNPs in high LD (correlation) with each other would likely be closer together in the embedded space. This allows the prediction algorithm to have more useful information about the SNPs rather than just their values (i.e. how many mutations are present in the sample for a given SNP).

The outcome that I am working on predicting are molecular traits. for example blood plasma metabolites.  

### Methods of interest ### 
1. Denoising Autoencoders for lower dimensional latent variable representations /feature extractions of our SNPs.
2. PCA or a localised PCA.
3. UMAP (Uniform manifold Approximation and projection).
4. Neural networks with embeddings generated from Denoising Autoencoders.

Currently I am testing these methods with real and simulated data.


