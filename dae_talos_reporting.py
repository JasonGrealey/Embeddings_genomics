print("importing keras")
import keras
print("importing numpy")
import numpy as np
print("importing Pandas")
import pandas as pd
print("importing Talos")
import talos as ta
print("importing Sys")
import sys
print("importing matplotlib")
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
def main():
    plot_dir='/projects/jgrealey/embedding/ten_k_samples/plots/dae/'
    if len(sys.argv)!=2:
        print("number of required inputs is no reached")
        print("please pass csv file to the program")
        print("exiting program")
        sys.exit(0)
    experiment=str(sys.argv[1])#'dae_test_model_1.csv'
    load=experiment+".csv"
    print(experiment)
    report=ta.Reporting(load)
    df=report.data
    print(df.shape)
    #print(type(report.data))
    #print(df.columns)
    #for i in df.columns:
    #    print(i)
    #print(df['round'])
    print(df['val_acc'])
    df=df.sort_values(by='val_acc',ascending=False)#reorder dataframe by validation accuracy
    print(df['val_acc'])
    #oldindex=
    
    df=df.reset_index(drop=True)#reset indexes
    index=df.index#best_params)
    #print(index)
    #print(index[0])
    print("best performing model is:")
    print(df.iloc[index[0]])
   
    print("now in order of most accurate")
    print("optimizers")
    print(df['optimizer'])
    print("first_neuron")
    print(df['first_neuron'])
    print("embedding size")
    print(df['embedding_size'])
    print(df['last_activation'])
    report.plot_corr()
    print(plot_dir)
    plt.title("Heatmap showing correlation between hyperparameters")
    plt.savefig(plot_dir+"test_corr"+experiment+".png")
    plt.show()
    plt.close()
    report.plot_hist()
    plt.title("histogram for validation accuracy for each experiment")
    plt.savefig(plot_dir+"test_hist"+experiment+".png")
    plt.close()
if __name__ == "__main__":
    main()
