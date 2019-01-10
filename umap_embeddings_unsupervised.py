import numpy as np
import pandas as pd
#from mnist import MNIST
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
import umap
import time
'''
This file will take the input matrix of n samples and p snps
then it will project to a lower amount of dimensions using UMAP (uniform manifold approximation and projection).
Given input number of neighbours (like perplexity in t-SNE) and number of output dimensions it will perform the projections.
It will then plot the first two dimensions for all of the samples.
Then it will save the embeddings in a directory -     results_dir="/home/jgrealey/embedding/ten_k_samples/results/umap/"

This is the file for unsupervised projections

'''

#data=read_csv("loadstring - standardised csv")
#phenotypes = aoinrgpaingapinhapirhna
#chec embeddings

#embedding = umap.UMAP().fit_transform(data, y=target)
'''
def draw_umap(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title='',data=None):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
        )
    u = fit.fit_transform(data);
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)), c=data)
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1], c=data)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], c=data, s=100)
    plt.title(title, fontsize=18)
    plt.savefig(plots_dir+str(title)+"testembeddings.pdf")
'''

def load(location,index_column=None,tpos=None):
    if index_column!=None:#if index column is specified, reload and pickle
        print("Reloading to reindex pickle")
        #print("creating pickle")
        if isinstance(index_column,int):
            print("reindexing")
            df=pd.read_csv(location+".csv",index_col=index_column)#reload but reindex
            print("reindexed")
            #,nrows=10000)#each new row read is another SNP to take into account.
        else:
            print("not reindexing")
            df=pd.read_csv(location+".csv")#reload but not index specified
            print("loaded")
        print("pickling")
        if tpos!=None:
            print("transposing")
            df=df.T#perform a transpose
            print("transposed")
        print("pickling")
        df.to_pickle(location+".pkl")
        print("pickled")
    else:#check if previous pickle is created and load
        if os.path.exists(location+".pkl"):# and 1<0:
            print("pickle exists\nloading pickle file")
            df=pd.read_pickle(location+".pkl")
            print("pickle loaded")
        else:#createpickle
            print("reading csv file")
            df=pd.read_csv(location+".csv")#,index_col=0)#,nrows=10000)#each new row read is another SNP to take into account.
            if tpos!=None:
                print("transposing")
                df=df.T
                print("transposed")
            print("creating pickle")
            df.to_pickle(location+".pkl")
            print("pickled")
    return df
def main():
    path_input="/home/jgrealey/Simulations/ten_k_samples/test/genotypes"
    #Maybe i want the standardised genotype
    path_pheno="/home/jgrealey/Simulations/ten_k_samples/test/phenotype0"
    results_dir="/home/jgrealey/embedding/ten_k_samples/results/umap/"
    plots_dir="/home/jgrealey/embedding/ten_k_samples/plots/umap/"
    numsam=1000
    nphenotoload=numsam-3#
    listtoload=range(numsam)
    numsnps=300
    #print(listtoload)
    #for x, rows are snps and columns are samples so usecoles

    #x=pd.read_csv(path_input+".csv",index_col=0,usecols=listtoload,nrows=numsnps)#rows are snps#nrows=100)#testing for faster I/O
    x=pd.read_csv(path_input+".csv",index_col=0)#,usecols=listtoload,nrows=numsnps)#rows are snps#nrows=100)#testing for faster I/O
    x=x.T
    #y=pd.read_csv(path_pheno+".csv",index_col=0,nrows=nphenotoload,header=None)#tasting for faster I/O
    y=pd.read_csv(path_pheno+".csv",index_col=0,header=None)
    y.columns=["phenotype"]
    print(x.tail())
    #print(y.tail())

    index=x.index
    cols=x.columns
    #print(index)
    #print(cols)
    
    samples=index[2:-3]
    testsamples=index[2:]
    #print(testsamples)
    #print(y)
    #xnorm=normalise(x,[2,-3])
    #del x
    #x=xnorm
    #print(xnorm)
    
    genotypes=x.loc[samples]
    #genotypes=x.loc[testsamples]
    # calculate sparsity
    #sparsity = 1.0 - np.count_nonzero(genotypes) /genotypes.size
    #print(sparsity)0.7520695419592109
    labels=x.loc[index[-1]].astype('int')
    #labels.index=xnorm.index
    #print(labels)
    beta=x.loc[index[-2]].astype('float')
    maf=x.loc[index[-3]].astype('float')
    SNP_id=x.loc[index[0]]
    SNP_pos=x.loc[index[1]]
    #print(genotypes)
    df=genotypes.join(y)
    #print(df)
    #print(df.loc[testsamples])
    #print(df["phenotype"])
    data=df.loc[testsamples]
    #embedding = umap.UMAP().fit_transform(df.loc[testsamples], y=df["phenotype"])#
    #print(embedding)
    #print(df)
    for n in ( 10, 100, 200,500):#numneighbours
        for d in (2,10,100):#dimensions
            #draw_umap(n_neighbors=n, title='n_neighbors-{}'.format(n),data=df.loc[testsamples])
            start_time = time.time()
            #print("--- %s seconds ---" % (time.time() - start_time))
            fit = umap.UMAP(
            n_neighbors=n,
            #min_dist=min_dist,
            n_components=d#n_components,
            #metric=metric
            )
            #    u = fit.fit_transform(data);
            print("dimensions is {}".format(d))
            print("neighbours in consideration is {}".format(n))
            #print(fit)
            cols=["dim"+str(i+1) for i in range(d)] 
            #embedding_unsup=fit.fit_transform(df.loc[testsamples])#,n_neighbors=n)
            embedding_unsup=fit.fit_transform(df.loc[samples])
            embedding_unsup=embedding_unsup.T#now samples are rows + columns are dimensions
            #print(embedding_unsup[0])
            #print(len(embedding_unsup[1]))
            print(df.head())
            embedding=pd.DataFrame(embedding_unsup.T,index=genotypes.index,columns=cols)

            #print(df["phenotype"].head())
            #print(len(embedding_unsup))
            #print(len(embedding_unsup)[0])
            
            plt.scatter(embedding_unsup[0],embedding_unsup[1],c=df["phenotype"].values)
            plt.title("testing with neighbours n = {} and comps = {}".format(n,d))
            plt.savefig(plots_dir+"n_neighbours"+str(n)+"ncomps"+str(d)+"testembeddings.pdf")

            print(embedding)
            embedding.to_csv(results_dir+"UMAP_embedding_unsupervised"+str(d)+"comps"+str(n)+"nneigh.csv")
            print("for neigh = {} and comps = {}".format(n,d))
            print("--- %s seconds ---" % (time.time() - start_time))

    #embedding = umap.UMAP(n_neighbors=5).fit_transform(data)
    print("finished unsupervised embedding")
    #fig, ax = plt.subplots(1, figsize=(14, 10))
    #plt.scatter(*embedding.T, s=0.3, c=df["phenotype"], cmap='Spectral', alpha=1.0)
    #plt.setp(ax, xticks=[], yticks=[])
    #cbar = plt.colorbar(boundaries=np.arange(11)-0.5)
    #cbar.set_ticks(np.arange(10))
    #cbar.set_ticklabels(classes)

if __name__ == '__main__':
        main()

