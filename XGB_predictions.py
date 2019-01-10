import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#matplotlib.use("TkAgg")
import xgboost as xgb
import sys
from sklearn.metrics import r2_score
def compare_index(df1,df2):
    '''
    This function compares the indexes of two dataframes and returns a boolean
    if they share the same indexes
    '''

    idx1=df1.index
    idx2=df2.index
    len_idx1=len(idx1)
    len_idx2=len(idx2)
    compar_len=len(idx1.intersection(idx2))
    if compar_len == (len_idx1 and len_idx2):
        booles=True#
        print("The two dataframes contain the same indexes")
    else:
        booles=False    
        print("The two datafrmes contain different indexes")
        
    print("dataframe one is of size {}".format(df1.shape))
    print("datframe two is of size {}".format(df2.shape))

    return booles

def main():
    if len(sys.argv)!=4:
        print("incorrect number of arguments - exiting program")
        sys.exit(0)
    embed_type=str(sys.argv[1])
    if embed_type not in ["pca","dae","umap"]:
        print("invalid embedding type")
        sys.exit(0)
    embed_file=str(sys.argv[2])
    print("Embedding type {} for file {}".format(embed_type,embed_file))
    pheno_file=str(sys.argv[3])
    pheno_save=pheno_file.replace('/', "_", 1)#cannot save name of file with a / in it 
    print(pheno_save)
    print("filename for phenotype is {}".format(pheno_file))
    #dae,pca,umap
    path_input="/projects/jgrealey/embedding/ten_k_samples/results/"+embed_type+"/"
    path_pheno="/projects/jgrealey/Simulations/ten_k_samples/"
    results_dir="/projects/jgrealey/embedding/ten_k_samples/results/"+embed_type+"/"
    plots_dir="/projects/jgrealey/embedding/ten_k_samples/plots/"+embed_type+"/"
    save_string=embed_type+embed_file[:-4]+pheno_save
    print(save_string)
    #for x, rows are snps and columns are samples so usecoles
    embeds=pd.read_csv(path_input+embed_file,index_col=0)
    print(embeds.head())
    y=pd.read_csv(path_pheno+pheno_file+".csv",index_col=0,header=None)#,dtype='float')#tasting for faster I/O
    y.columns=["phenotype"]
    indexs=np.asarray(embeds.index)
    y=y.loc[indexs]
    #print(y)
    #df=embeds.join(y)
    #print(df)
    test_size=0.2#30% of 7200 (2700)
    seed=10
    x_train, x_test, y_train, y_test = train_test_split(embeds, y, test_size=test_size, random_state=seed)
    #x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.15,random_state=seed)
    compare_train=compare_index(x_train,y_train)
    compare_test=compare_index(x_test,y_test)
    #compare_val=compare_index(x_val,y_val)
    if compare_train and compare_test != True:
        print("indexes of datasets are incorrect\nexiting program")
        sys.exit(0)
    
    train=xgb.DMatrix(x_train.values,y_train.values)
    #print(train)

    #params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
    #            'max_depth': 5, 'alpha': 10}

    #cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
    #                num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
    #xgtrain = xgb.DMatrix(X_sel, label=y)
    clf = xgb.XGBRegressor(max_depth = 7,
                n_estimators=700,
                learning_rate=0.1, 
                nthread=-1,
                subsample=1.0,
                colsample_bytree=0.5,
                min_child_weight = 3,
                seed=1)
    xgb_param = clf.get_xgb_params()
    #do cross validation
    print ('Start cross validation')
    cvresult = xgb.cv(xgb_param, train, num_boost_round=5000, nfold=10, metrics=['rmse'],
    early_stopping_rounds=50, seed=1)
    print(cvresult)
    print('Best number of trees = {}'.format(cvresult.shape[0]))
    clf.set_params(n_estimators=cvresult.shape[0])#setting params to best values based on CV
    print('Fit on the trainings data')
    clf.fit(x_train,y_train, eval_metric='rmse')
    #print('Overall AUC:', roc_auc_score(y, clf.predict_proba(X_sel)[:,1]))
    print('Predict the probabilities based on features in the test set')
    pred = clf.predict(x_test, ntree_limit=cvresult.shape[0])
    print(pred)
    print(pred.shape)
    print(y_test.shape)
    submission = pd.DataFrame({"ID":y_test.index, "TARGET":pred[:]})
    rsq=r2_score(y_test, pred)  
    print(rsq)
    #plt.scatter(y_test,pred,marker='+',label="Rsq="+str(rsq))
    #plt.xlabel("true phenotype")
    #plt.ylabel("predicted phenotype")
    #plt.legend()
    #plt.savefig(plots_dir+save_string+"predicted_phenotype_true_phenotype.pdf")
    from xgboost import plot_importance

    print(clf.feature_importances_)
    # plot
    #plt.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
    plot_importance(clf,max_num_features=30,show_values=False)
    plt.savefig(plots_dir+save_string+"importances.pdf")
    plt.show()
    
    
    fig, ax = plt.subplots()
    ax.scatter(y_test, pred, s=25, marker="+")
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75)
    ax.set_aspect('equal')
    ax.set_title('XGB Predictions')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("true phenotype")
    ax.set_ylabel("predicted phenotype")
    fig.savefig(plots_dir+save_string+"predicted_phenotype_true_phenotype.pdf")
    
    '''
    train = pd.read_csv("train.csv")
    target = train['target']
    train = train.drop(['ID','target'],axis=1)
    test = pd.read_csv("test.csv")
    test = test.drop(['ID'],axis=1)

    xgtrain = xgb.DMatrix(train.values, target.values)
    xgtest = xgb.DMatrix(test.values) 
    
    
    xgb.plot_tree(xg_reg,num_trees=0)
    plt.rcParams['figure.figsize'] = [50, 10]
    plt.show()
    '''
if __name__ == '__main__':
    main()
