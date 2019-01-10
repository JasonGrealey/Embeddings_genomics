import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
import sys

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

def load_json(location,index_column=None,tpos=None):
    if index_column!=None:#if index column is specified, reload and pickle
        print("Reloading to reindex the Dumped file")
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
        print("dumping to JSON")
        if tpos!=None:
            print("transposing")
            df=df.T#perform a transpose
            print("transposed")
        print("dumping")
        df.to_json(location+".json")
        print("dumped")
    else:#check if previous pickle is created and load
        if os.path.exists(location+".json"):# and 1<0:
            print("JSON dump exists\nloading file")
            df=pd.read_json(location+".json")
            print("loaded")
        else:#createpickle
            print("reading csv file")
            df=pd.read_csv(location+".csv")#,index_col=0)#,nrows=10000)#each new row read is another SNP to take into account.
            if tpos!=None:
                print("transposing")
                df=df.T
                print("transposed")
            print("dumping to JSON ")
            df.to_json(location+".json")
            print("dumped")
    return df

def load_hdf(location,index_column=None,tpos=None):
    if index_column!=None:#if index column is specified, reload and pickle
        print("Reloading to reindex the Dumped file")
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
        print("dumping to hdf5")
        if tpos!=None:
            print("transposing")
            df=df.T#perform a transpose
            print("transposed")
        print("dumping")
        df.to_hdf(location+".h5",'df')
        print("dumped")
    else:#check if previous pickle is created and load
        if os.path.exists(location+".h5"):# and 1<0:
            print("hdf5 dump exists\nloading file")
            df=pd.read_hdf(location+".h5")
            print("loaded")
        else:#createpickle
            print("reading csv file")
            df=pd.read_csv(location+".csv")#,index_col=0)#,nrows=10000)#each new row read is another SNP to take into account.
            if tpos!=None:
                print("transposing")
                df=df.T
                print("transposed")
            print("dumping to hdf5 ")
            df.to_hdf(location+".h5",'df')
            print("dumped")
    return df
#>>> df.to_hdf('./store.h5', 'data')
#def binomvar(

def normalise(df,position=[]):
    print("normalising")
    result = df.copy()
    print(position)
    #select only samples
    index=df.index
    cols=df.columns
    
    index_to_norm = index[int(position[0]):int(position[1])]#[2:-3]#need to drop other metrics
    print(df.loc[index_to_norm])
    #maf=index[-3]
    #cols_to_norm=cols[
    #calculate variance of all columns
    #print(index[-3])
    #print("norm probs")
    #calculating normlised pvals from equation 4 in flashpca - Fast Principal Component Analysis of Large-Scale Genome-Wide Data
    p_i=df.loc[index[-3]]/(2.0)#.astype(float)
    
    #print(p_i.loc[0])
    #print("float cast")
    p_i = p_i.astype('float')
    #print(p_i.loc[0])
    #print("denominator of equation")
    denominator=np.sqrt(p_i*(1-p_i),dtype=np.float32)
    #print(denominator.loc[cols[1]])
    #print(cols)
    #print(denominator)
    #taking away mean
    #normalized_df=(df-df.mean())/df.std()
    #print(result.loc[index_to_norm])
    result.loc[index_to_norm]=result.loc[index_to_norm].apply(pd.to_numeric)
    result.loc[index_to_norm]=result.loc[index_to_norm]-result.loc[index_to_norm].mean()
    #print(result.head())
    result.loc[index_to_norm]=result.loc[index_to_norm]/denominator
    #print(result.head())
    #print(denominator.head())
    
    '''
    for feature_name in cols:#index_to_norm:
        #rint(feature_name)
        mean_value = float(df.loc[index_to_norm,feature_name].mean())
        #print(len(mean_value))
        #min_value = df[feature_name].min()
        print(len(df.loc[index_to_norm,feature_name]))
        result[feature_name] = (np.asarray(df.loc[index_to_norm,feature_name],dtype='float')-mean_value) 
        print(result[feature_name])
        result[feature_name]=results[feature_name]/ (denominator.loc[feature_name])#*(1-p_i))^2max_value - min_value)
    '''        
    print(result)
    return result
    
def main():

    if(len(sys.argv)!=2):
        print("incorrect number of parameters\nexiting program!")
        print("please input number of components to caluculate")
        sys.exit(0)
    ncomps=int(sys.argv[1])
    print("NUMBER OF PCS {}".format(ncomps))
    npcstoplot=5
    loadstring="/home/jgrealey/Simulations/ten_k_samples/test/genotypes"
    phenoloadstring="/home/jgrealey/Simulations/ten_k_samples/test/phenotype0"
    '''
    if os.path.exists(loadstring+".pkl"):# and 1<0:
        print("loading pickle file")
        df=pd.read_pickle(loadstring+".pkl")
        print("pickle loaded")
    else:#createpickle
        print("creating pickle")
        df=pd.read_csv(loadstring+".csv",index_col=0)#,nrows=10000)#each new row read is another SNP to take into account.
        df.to_pickle(loadstring+".pkl")
        print("pickle created")
    '''
    
    #standardise function
    #plotPCA
    #save dimensions
    #done
    #x=load(loadstring,index_column=1.0)
    #x=load(loadstring,index_column="aj")
    #x=load(loadstring,index_column=0)
    #x=load_json(loadstring,index_column=0,tpos="s")
    #test=load_hdf(loadstring,index_column=0,tpos="s") 
    
    x=load_hdf(loadstring)
    #print(x)
    #x=pd.read_csv(loadstring+".csv",index_col=0,nrows=100)#testing for faster I/O
    #x=x.T
    #print(x.head)
    #y=load(phenoloadstring)#forPCA we do not need output
    #x=x.T
    #print(x)
    index=x.index
    cols=x.columns
    #print(index)
    #print(cols)
    
    samples=index[2:-3]
    #print(y)
    xnorm=normalise(x,[2,-3])#important to standardise before pca
    del x
    #x=xnorm
    print(xnorm)
    
    genotypes=xnorm.loc[samples]
    labels=xnorm.loc[index[-1]].astype('int')
    #labels.index=xnorm.index
    print(labels)
    beta=xnorm.loc[index[-2]].astype('float')
    maf=xnorm.loc[index[-3]].astype('float')
    SNP_id=xnorm.loc[index[0]]
    SNP_pos=xnorm.loc[index[1]]
    
    '''count     56318
    unique    56318
    top       56317
    freq          1
    Name: Unnamed: 0, dtype: int64
    count           56318
    unique          56318
    top       rs187503663
    freq                1
    Name: SNP, dtype: object
    count    56318.000000
    mean         0.001243
    std          0.039957
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          2.000000
    Name: label, dtype: float64
    count    56318.000000
    mean         0.000087
    std          0.004878
    min         -0.231977
    25%          0.000000
    50%          0.000000
    75%           0.000000
    max          0.387054
    Name: beta, dtype: float64
    count    56318.000000
    mean         0.126986
    std          0.117957
    min          0.005000
    25%          0.022350
    50%          0.085500
    75%          0.214750
    max          0.400000
    Name: maf, dtype: float64
    
    #print(SNP_id.describe())
    #print(SNP_pos.describe())
    #print(labels.describe())
    #print(beta.describe())
    #print(maf.describe())
    #print(genotypes.describe())
    '''
    

    #df[samples])
    
    
    #print(len(samples))
    #print(df.loc[:,cols[:-2]])
    #plate=df.loc[:,cols[-2]]
    #print(plate)
    #df_missing=pd.read_csv("../data/cleaned/df_missingvals.csv",index_col=0)
    #pca = PCA(n_components=20)#n_components=5)
    #pca.fit(df.loc[:,cols[:-2]])
    #pca.fit_transform(df.loc[:,cols[:-2]].T)#.values maybe needed 
    #df2.index=df.index
    pca = PCA(n_components=ncomps)
    #pca.fit(df)
    #allsamples=df.loc[:,cols[:-2]]
    #print(allsamples.shape)#X : array-like, shape (n_samples, n_features), 8k x 3k
    pcafit=pca.fit_transform(genotypes)#returns Nsamples x ncompoents
    print(pcafit)
    print("pcashape")
    print(pcafit.shape)
    columns = ['pca_%i' % i for i in range(1,ncomps+1,1)]
    df_pca = pd.DataFrame(pcafit, columns=columns, index=genotypes.index)
    print("pca final index")
    print(df_pca.head())
    print("pca final shape")
    print(df_pca.shape)
    #df_pca=pd.concat([df_pca, labels], axis=0)#f_pca.join(plate)i
    #print(df_pca)
    print(df_pca.columns)
    cols_pca=df_pca.columns
    #print(cols_pca[0])
    #print(pca.components_.shape)
    #print(pca.components_)
    #print(df2)
    #numdims X numsamples
    #taking 0 is 1st dim
    #print(pca.components_.shape[0])#first component
    #print(pca.components_.shape[1])#second component
    #print(pca.components_ )
    print(pca.explained_variance_)
    #print(len(pca.components_))
    #print(len(pca.components_[0]))  
    print("let's get reconstruction error")
    x_recon=pca.inverse_transform(X=df_pca)
    print(x_recon.shape)
    print(genotypes.shape)
    recon_error=mean_squared_error(genotypes,x_recon)
    print("Reconstrction Error with {} components\nis {} %".format(ncomps,recon_error))

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.savefig("/projects/jgrealey/Simulations/ten_k_samples/plots/pca/cumulativePCAplot.pdf",bbox_inches="tight")
    plt.show()
    
    #columns.append("plate") 
    
    #df_pca.to_csv("/home/jgrealey/Simulations/ten_k_samples/test/pca_ncom_"+str(ncomps)+".csv")
    df_pca.to_csv("/projects/jgrealey/embedding/ten_k_samples/results/pca/pca_ncom_"+str(ncomps)+".csv")#/home/jgrealey/embedding/ten_k_samples/results

    print(columns[10:-1])
    #print(columns[50])
    print(df_pca)#.loc[:,columns[0]])
    print(cols_pca[10:-1])
    df_pca=df_pca.drop(columns=np.asarray(cols_pca[npcstoplot:-1],dtype=str),axis=1)
    print(df_pca)
    
    #plt.plot(df_pca.pca_0,df_pca.pca_1,l)
    #scatted plotting all rows in the first ten pcas, using labels ==plate description
    plot_kws={"s": 3}
    g= sns.pairplot(df_pca)#, hue="labels")#df.loc[:,columns[-1]])#"plate")
    g.fig.suptitle("PCA of Mean Imputed Metabolites")
    #axes = pd.plotting.scatter_matrix(df_pca.loc[:,columns[0:10]], alpha=0.2,color=df.loc[:,columns[-1]])
    plt.tight_layout()
    
    plt.savefig('/projects/jgrealey/embedding/ten_k_samples/plots/pca/scatter_matrix'+"npcs"+str(ncomps)+"numplot"+str(npcstoplot)+'test.png')    
    #    plt.savefig('/home/jgrealey/Simulations/ten_k_samples/plots/pca/scatter_matrix'+"npcs"+str(ncomps)+"numplot"+str(npcstoplot)+'test.png')

    '''
    sklearn_pca = sklearnPCA(n_components=2)
    sklearn_transf = sklearn_pca.fit_transform(all_samples.T)

    plt.plot(sklearn_transf[0:20,0],sklearn_transf[0:20,1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
    plt.plot(sklearn_transf[20:40,0], sklearn_transf[20:40,1], '^', markersize=7, color='red', alpha=0.5, label='class2')
    
    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.legend()
    plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')
    
    plt.show()
    '''
    #principalDf = pd.DataFrame(data = principalComponents
    #         , #FOR LOOP LOOPING OVER EACH PC AND PLOT THE FIRST 5 SAYcolumns = ['principal component 1', 'principal component 2',])
    # plot data
    #plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
    #for length, vector in zip(pca.explained_variance_, pca.components_):
    #           v = vector * 3 * np.sqrt(length)
    #   draw_vector(pca.mean_, pca.mean_ + v)
    #plt.axis('equal');
if __name__ == '__main__':
        main()

