import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
import sys
import statsmodels.api as sm
import scipy.cluster.hierarchy as sch#import dendrogram, linkage, distance
from scipy import stats#.spearmanr as spear
'''
=============================================================================================================================
This file will take the input matrix of n samples and p snps
then it will project to a lower amount of dimensions using UMAP (uniform manifold approximation and projection).
Given input number of neighbours (like perplexity in t-SNE) and number of output dimensions it will perform the projections.
It will then plot the first two dimensions for all of the samples.
Then it will save the embeddings in a directory -     results_dir="/home/jgrealey/embedding/ten_k_samples/results/umap/"

This is the file for supervised projections

=============================================================================================================================
#data=read_csv("loadstring - standardised csv")
#phenotypes = aoinrgpaingapinhapirhna
#chec embeddings

#embedding = umap.UMAP().fit_transform(data, y=target)
'''
def clustering(dataframe):
    df_known=dataframe.copy()
    print("clustering known")
    X = df_known.corr().values
    print(X)
    print("correlations calculated for known")
    d = sch.distance.pdist(X)   # vector of ('55' choose 2) pairwise distances
    print("distance matrix created for known")
    L = sch.linkage(d, method='complete')
    print("linkage martix created for known")
    t=0.5*d.max()
    ind = sch.fcluster(L, t, 'distance')
    columns = [df_known.columns.tolist()[i] for i in list((np.argsort(ind)))]#sorting them by index.i.e. grouping
    df_known = df_known.reindex(columns, axis=1)
    print("dataframe realigned for known")
    return df_known
def spearman(inputmatrix,phenotype):
    indexes=inputmatrix.index.tolist()
    columns=inputmatrix.columns.tolist()
    #print("This must be a sample")
    #print(inputmatrix.loc[indexes[0]])
    #print("this must be a feature")
    #print(inputmatrix[columns[0]])
    print("now testing correlation between phenotype and embedding")
    pvals=pd.Series(data=np.zeros(len(columns)),index=inputmatrix.columns)
    #pvals=np.zeros(len(columns)
    corrs=pd.Series(data=np.zeros(len(columns)),index=inputmatrix.columns)

    #corrs=np.zeros(len(columns))
    for i in range(len(columns)):#need a test for each embedding passed
        rho,p=stats.spearmanr(inputmatrix[columns[i]].values,phenotype.values)
        pvals[i]=p
        corrs[i]=rho
        #print(p)
        #print(rho)
    #pvalues=pd.DataSeries(data=pvals,index
    return corrs,pvals


def plot_corr(df,size=20):
                '''Plot a graphical correlation matrix for a dataframe.
                #https://github.com/TheLoneNut/CorrelationMatrixClustering/blob/master/CorrelationMatrixClustering.ipynb
                Input:
                                df: pandas DataFrame
                                size: vertical and horizontal size of the plot'''
                # Compute the correlation matrix for the received dataframe
                #corr = df.corr(method='pearson', min_periods=1)
                corr = df.corr(method='spearman', min_periods=1)
                # Plot the correlation matrix
                fig, ax = plt.subplots(figsize=(size, size))
                #cax = ax.matshow(corr, cmap='RdYlGn_r')
                cax = ax.matshow(corr, cmap='Blues')

                #plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
                #plt.yticks(range(len(corr.columns)), corr.columns);
                # Add the colorbar legend
                v = np.linspace(-1.0, 1.0, 15, endpoint=True)
                cbar = fig.colorbar(cax, ticks=v, aspect=40, shrink=.7)



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
    if len(sys.argv)!=4:
        print("incorrect number of arguments - exiting program")
        sys.exit(0)
    embed_type=str(sys.argv[1])
    if embed_type not in ["pca","dae","umap"]:
        print("invalid embedding type")
        sys.exit(0)
    embed_file=str(sys.argv[2])
    print("Embedding type{} for file {}".format(embed_type,embed_file))
    pheno_file=str(sys.argv[3])
    pheno_save=pheno_file.replace('/', "_", 1)
    print(pheno_save)
    print("filename for phenotype is {}".format(pheno_file))
    #dae,pca,umap
    path_input="/projects/jgrealey/embedding/ten_k_samples/results/"+embed_type+"/"
    
    path_pheno="/projects/jgrealey/Simulations/ten_k_samples/"
    results_dir="/projects/jgrealey/embedding/ten_k_samples/results/"+embed_type+"/"
    plots_dir="/projects/jgrealey/embedding/ten_k_samples/plots/"+embed_type+"/"
    numsam=1000
    save_string=embed_type+embed_file[:-4]+pheno_save
    print(save_string)
    nphenotoload=numsam-3#
    listtoload=range(numsam)
    numsnps=300
    testcols=np.arange(100)
    #print(listtoload)
    #for x, rows are snps and columns are samples so usecoles
    embeds=pd.read_csv(path_input+embed_file,index_col=0)
    print(embeds.head())
    #print(embeds.dtypes)
    print(embeds.shape)
    #x=pd.read_csv(path_input+snp_file+".csv",index_col=0,usecols=listtoload,nrows=numsnps)#rows are snps#nrows=100)#testing for faster I/O
    #full laod
    #cols_read=pd.read_csv(path_input+snp_file+".csv",usecols=[0])
    #print(cols_read)

    #sys.texit(0)
    #x=pd.read_csv(path_input+snp_file+".csv",index_col=0,dtype='object')#,usecols=listtoload,nrows=numsnps)#rows are snps#nrows=100)#testing for faster I/O
    
    #testing
    #x=pd.read_csv(path_input+snp_file+".csv",index_col=0,usecols=listtoload)#,nrows=numsnps)#rows are snps#nrows=100)#testing for faster I/O
    #ONLY TRANSPOSE IF READING GENOTYPE FILE 
    #STANDARDISED FILE DOESNT NEED TO BE TRANSPOSED
    #x=x.T
    y=pd.read_csv(path_pheno+pheno_file+".csv",index_col=0,header=None)#,dtype='float')#tasting for faster I/O
    #y=pd.read_csv(path_pheno+pheno_file+".csv",index_col=0,nrows=nphenotoload,header=None)#tasting for faster I/O
    y.columns=["phenotype"]
    #print(y)
    indexs=np.asarray(embeds.index)
    y=y.loc[indexs]
    embed_cols=embeds.columns
    '''
    pvals=pd.Series(data=np.zeros(len(embed_cols)),index=embeds.columns)
    #print(y.shape)
    pvals.columns=['pvals']
    for i in range(len(embed_cols)):
        #print(embeds[embed_cols[i]])
        model =sm.OLS(y,embeds[embed_cols[i]])
        results=model.fit()
        if i==(len(embed_cols)-1):
            print(results.summary())
        #print(results.pvalues)
        pvals[embed_cols[i]]=results.pvalues
    '''
    correlations,pvals=spearman(embeds,y)
    print(correlations)
    print(pvals)
    plt.figure()
    ax = sns.distplot(correlations.values)
    plt.title("Spearman correlations of all embeddings with phenotype")
    plt.savefig(plots_dir+save_string+"_spearman_pheno_correlations.pdf",bbox_inches="tight")
    plt.show()
    plt.close()
    
    print("done with correlations")
    #print(pvals)
    pvals_to_plot=pvals[pvals<=0.05]#under 5%
    pvals_to_plot=pvals_to_plot[pvals_to_plot!=0]#must be non zero pvalues
    per_assc=(float(len(pvals_to_plot))/len(pvals))*100#percentage
    

    embedsnew=embeds.append(pvals,ignore_index=True)
    embedsnew=embedsnew.append(correlations,ignore_index=True)
    indexnew=embeds.index.tolist()
    indexnew.append('pvals')
    indexnew.append('correlations')
    embedsnew.index=indexnew#embeds.index+'pvals']
    print(embedsnew)
    embedsnew=embedsnew.sort_values('pvals',axis=1)
    print(embedsnew)

    if len(pvals_to_plot)!=0:
        pvals_to_plot.hist(xrot=90, bins=10,label=str(per_assc)+' = percentage associated')
        plt.title("histogram of Pvalues for " +embed_type)
        plt.legend()
        plt.savefig(plots_dir+save_string+"_pvalues.pdf",bbox_inches="tight")
        plt.show()
        plt.close()
        #ax.tick_params(axis='y', which='major', pad=15)
        #pylab.rcParams['ytick.major.pad']='8'
        pvals.to_csv(results_dir+save_string+"pvalues.csv")
    else:
        print("no association under 5%")
    #print(np.sort(pvals.values))
    '''
    df_known=embeds.copy()
    print("clustering known")
    X = df_known.corr().values
    print(X)
    print("correlations calculated for known")
    d = sch.distance.pdist(X)   # vector of ('55' choose 2) pairwise distances
    print("distance matrix created for known")
    L = sch.linkage(d, method='complete')
    print("linkage martix created for known")
    t=0.5*d.max()
    ind = sch.fcluster(L, t, 'distance')
    columns = [df_known.columns.tolist()[i] for i in list((np.argsort(ind)))]#sorting them by index.i.e. grouping
    df_known = df_known.reindex(columns, axis=1)
    print("dataframe realigned for known")
    
    
    if len(pvals_to_plot)!=0:
        ass_embeds=embeds[pvals_to_plot.index]
        ass_cluster=clustering(ass_embeds)
        plot_corr(ass_cluster)
        plt.title("Correlation Structure of associated Embeddings")
        plt.savefig(plots_dir+save_string+"associated_correlation_embeddings.pdf",bbox_inches="tight")
        #print(cormetabs_known.head())
        plt.show()
        plt.close()
    '''
    '''
    else:
        print("NO ASSOCIATION under 5%")
        print("plotting embeddings vs phenotype")
        testplot=pvals[pvals==0]
        indexes_test=testplot.index.tolist()
        for i in range(5):
            print("plotting unassociated embeddings")
            randint=np.random.randint(0,len(testplot))
            #print(testplot
            #plt.scatter(testplot[randint].values,y.values)
            plt.scatter(embeds[indexes_test[randint]].values,y.values,marker='x')
            plt.title(str(indexes_test[randint])+" dimension vs phenotype")
            plt.xlabel(str(indexes_test[randint]))
            plt.ylabel("phenotype")
            plt.savefig(plots_dir+save_string+str(indexes_test[randint])+"_unassociated_testplot_embed_vs_phenotype.pdf")
            plt.show()
            plt.close()
    #indexes_test=pvals_to_plot.index.tolist()
    #indexes_test=embedsnew.columns.tolist()
    '''
    #sorted_pvals=pvals.sort_values()#will sort pvals
    #indexes_test=sorted_pvals.index.tolist()
    #print(indexes_test)
    indexes_test=embedsnew.columns.tolist()#sorted pca by pvals index
    for i in range(10):
        print("plotting associated dimensions")
        #randint=np.random.randint(0,len(pvals_to_plot))
        #print(testplot
        #plt.scatter(testplot[randint].values,y.values)
        plt.scatter(embeds[indexes_test[i]].values,y.values,marker='x')
        plt.title(str(indexes_test[i])+" dimension vs phenotype"+str(embedsnew.loc['pvals'][i])+"corr"+str(embedsnew.loc['correlations'][i]))
        plt.xlabel(str(indexes_test[i]))
        plt.ylabel("phenotype")
        plt.savefig(plots_dir+save_string+str(indexes_test[i])+"ploting_associated_embed_vs_phenotype.pdf")
        plt.show()
        plt.close()
    '''
    plot_corr(df_known)
    plt.title("Correlation Structure of Embeddings")
    plt.savefig(plots_dir+save_string+"correlation_embeddings.pdf",bbox_inches="tight")
    #print(cormetabs_known.head())
    plt.show()
    plt.close()
    '''
    if len(pvals_to_plot)!=0:
        fig, ax = plt.subplots()
        pvals_to_plot.hist(ax=ax, bins=100, bottom=0.1,label=str(per_assc)+' = percentage associated')
        plt.title("histogram of pvalues for "+embed_type)
        ax.set_xscale('log')
        plt.legend()
        plt.savefig(plots_dir+save_string+"histogram_pvalues.pdf",bbox_inches="tight")
        plt.show()
        plt.close()
        fig, ax = plt.subplots()
        
        #pvals_to_plot.sort_values().plot.bar(stacked=True)
        pvals_to_plot=pvals_to_plot.sort_values()
        pvals_to_plot=pvals_to_plot.iloc[0:50]
        pvals_to_plot.plot.barh(ax=ax,label=str(per_assc)+' = percentage associated')
        #pvals_to_plot.plot.barh()
        plt.title("Bar of Pvalues for " +embed_type)
        ax.set_xscale('log')
        plt.legend()#label='$y = numbers'
        plt.savefig(plots_dir+save_string+"_bar_pvalues.pdf",bbox_inches="tight")
        plt.show()
        plt.close()

    print("percentage associated {}".format(per_assc))
    
    #embeds=embeds.append(pvals,ignore_index=True)
    #print(embeds)
    #embeds=embeds.sort_values(by=['pvals'],axis=1)
    #print(embeds)
    #pvals.plot.barh()
    #plt.title("Bar of Pvalues for " +embed_type)
    #plt.savefig(plots_dir+embed_type+embed_file[:-4]+"_bar_pvalues.pdf",bbox_inches="tight")
    #plt.show()

if __name__ == '__main__':
        main()

