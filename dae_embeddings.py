print("importing Keras")
import keras
print("importing Random, Matplotlib, Numpy, Seaborn, sys, Zipfile, Pandas, and Sklearn")

import random
import matplotlib
matplotlib.use("TkAgg")
import matplotlib as plt
import numpy as np
import seaborn as sns#; sns.set()
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import zipfile
print("importing Keras Modules")
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam
print("importing Talos - May get stuck?")
import talos as ta
from talos.model.early_stopper import early_stopper
from talos.model.normalizers import lr_normalizer
from talos.model.layers import hidden_layers
from keras.layers import Input, Dropout

'''
Talos for optimisation
corruption for snps


building denoising autoencoder from datasets ten_k_samples
need to mask input
and train autoencoder
then push test set throuh trained autoencoder and store embeddings
'''



def mask_function( dataframe,batch_sizes,noise=0.3):
    random.seed(10)#fixing seed of randomness so each training set gets the same mask
    #current error when taking arrays and not pandas dataframes - hmmm
    nsamp=dataframe.shape[0]
    to_go=nsamp
    nSNPs=dataframe.shape[1]
    print(dataframe.shape)
    noise_data=dataframe.copy()
    for i in range(0,nsamp,batch_sizes):
        #print("{} left to mask".format(to_go))
        batch=min(to_go,batch_sizes)
        #noise_data.iloc[i:i+batch_size,:]=noise_data[i:i+batch_size,:]
        #targets=noise_data.copy()
        #n_in_batch=noise_data.shape[0]
        mask=np.random.binomial(1,1-noise,(batch,nSNPs))#creating mask
        #print(mask.shape)
        #print(isinstance(noise_data,pd.DataFrame))
        #print(isinstance(noise_data,np.ndarray))
        batch=min(to_go,batch_sizes)#ensuring we don't fall outside the array
        if isinstance(noise_data, pd.DataFrame):#if passed data is a dataframe
            if i ==0:
                print("data is in Pandas Dataframe Format")
            #print(noise_data.iloc[i:i+batch_size,:].shape)
            noise_data.iloc[i:i+batch,:]=np.multiply(noise_data.iloc[i:i+batch,:],mask)#sliding mask
        elif isinstance(noise_data,np.ndarray):#if passed data is a numpy array
            if i == 0:
                print("data is in Numpy Array Format")
            noise_data[i:i+batch,:]=np.multiply(noise_data[i:i+batch,:],mask)#sliding mask
        #noise_data=np.multiply(noise_data,mask)
        else:
            print("unrecognised data format")
            sys.exit(0)
        to_go=to_go-batch_sizes
        #print(to_go)
    return noise_data

def dae_model(x_train,y_train, x_val,y_val, params):
    #print(params['x_train_noise'].shape)
    print(x_train.shape)
    print("masking training")
    x_train_noise=mask_function(dataframe=x_train,noise=float(params['noise']),batch_sizes=300)#masking training
    print("masking validation")
    x_val_noise=mask_function(dataframe=x_val,noise=float(params['noise']),batch_sizes=300)#masking validation
    
    print("building autoencoder network")
    model = Sequential()
    model.add(Dense(params['first_neuron'],  activation='relu', input_shape=(x_train.shape[1],)))
    #m.add(Dense(128,  activation='elu'))
    model.add(Dense(params['embedding_size'],    activation='relu', name="bottleneck"))
    model.add(Dense(params['first_neuron'],  activation=params['last_activation']))
    #m.add(Dense(512,  activation='elu'))
    model.add(Dense(x_train.shape[1],  activation='relu'))
    #m.compile(loss='mean_squared_error', optimizer = params['optmizer'])
    model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                  loss=params['loss'],
                  metrics=['accuracy'])
    print("training neural network")
    out= model.fit(x_train,x_train_noise,#x_train_noise,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    verbose=0,
                    validation_data=[x_val, x_val_noise],#x_val_noise],
                    callbacks=early_stopper(params['epochs'], mode='strict'))#noisy_train, train, batch_size=128, epochs=params['epochs'], verbose=1,
    return out, model

def dae_model_hl(x_train,y_train, x_val,y_val, params):
    #print(params['x_train_noise'].shape)
    print(x_train.shape)
    print("masking training")
    x_train_noise=mask_function(dataframe=x_train,noise=float(params['noise']),batch_sizes=300)#masking training
    print("masking validation")
    x_val_noise=mask_function(dataframe=x_val,noise=float(params['noise']),batch_sizes=300)#masking validation

    print("building autoencoder network")
    model = Sequential()
    model.add(Dense(params['first_neuron'],  activation=params['activation'], input_shape=(x_train.shape[1],)))
    model.add(Dropout(params['dropout'])) 

    #m.add(Dense(128,  activation='elu'))
    hidden_layers(model, params, 1)
    model.add(Dense(params['embedding_size'],    activation=params['activation'], name="bottleneck"))
    hidden_layers(model, params, 1)
    model.add(Dense(params['first_neuron'],  activation=params['activation']))
    #m.add(Dense(512,  activation='elu'))
    model.add(Dropout(params['dropout']))

    model.add(Dense(x_train.shape[1],  activation=params['last_activation']))
    #m.compile(loss='mean_squared_error', optimizer = params['optmizer'])
    model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                  loss=params['loss'],
                  metrics=['accuracy'])
    print("training neural network")
    out= model.fit(x_train,x_train_noise,#x_train_noise,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    verbose=0,
                    validation_data=[x_val, x_val_noise],#x_val_noise],
                    callbacks=early_stopper(params['epochs'], mode='moderate'))
                    #callbacks=early_stopper(params['epochs'], mode='strict'))#noisy_train, train, batch_size=128, epochs=params['epochs'], verbose=1,
    return out, model

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = x_train.reshape(60000, 784) / 255
#x_test = x_test.reshape(10000, 784) / 255
def main():
    from keras import backend as K
    K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=32, inter_op_parallelism_threads=32)))
    print("using 32 CPUs or 64 Cores on cluster")
    if len(sys.argv)!=3:
        print("incorrect number of arguments - exiting program")
        sys.exit(0)
    s = random.getstate()
    snp_file=str(sys.argv[1])
    print("filename for SNPs {}".format(snp_file))
    #pheno_file=str(sys.argv[2])
    #print("filename for phenotype is {}".format(pheno_file))
    path_input="/home/jgrealey/Simulations/ten_k_samples/test/"
    #Maybe i want the standardised genotype
    #path_pheno="/home/jgrealey/Simulations/ten_k_samples/test/"
    results_dir="/projects/jgrealey/embedding/ten_k_samples/results/dae/"
    plots_dir="/projects/jgrealey/embedding/ten_k_samples/plots/dae/"
    numsam=1000
    nphenotoload=numsam-3#
    listtoload=range(numsam)
    numsnps=300
    testcols=np.arange(100)
    datafile=snp_file
    model_name=str(sys.argv[2])#"dae_test_hiddenlay_model"
    experiment_name=model_name+"_"+snp_file
    print(experiment_name)
    print(model_name)
    #'weight_regulizer':[None]}
    #print(listtoload)
    #for x, rows are snps and columns are samples so usecoles
    
    #x=pd.read_csv(path_input+snp_file+".csv",index_col=0,usecols=listtoload,nrows=numsnps)#rows are snps#nrows=100)#testing for faster I/O
    #full laod
    #cols_read=pd.read_csv(path_input+snp_file+".csv",usecols=[0])
    #print(cols_read)

    #sys.texit(0)
    #full load
    x=pd.read_csv(path_input+snp_file+".csv",index_col=0,dtype='object')#,usecols=listtoload,nrows=numsnps)#rows are snps#nrows=100)#testing for faster I/O
    
    #testing
    #TESTING#
    #x=pd.read_csv(path_input+snp_file+".csv",index_col=0,usecols=listtoload)#,nrows=numsnps)#rows are snps#nrows=100)#testing for faster I/O
    #ONLY TRANSPOSE IF READING GENOTYPE FILE 
    #STANDARDISED FILE DOESNT NEED TO BE TRANSPOSED
    #x=x.T
    
    #TESTING#
    #y=pd.read_csv(path_pheno+pheno_file+".csv",index_col=0,header=None)#,dtype='float')#tasting for faster I/O
    #full load
    #y=pd.read_csv(path_pheno+pheno_file+".csv",index_col=0,nrows=nphenotoload,header=None)#tasting for faster I/O

    #y.columns=["phenotype"]
    #print(x.tail())
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
    genotypes=genotypes.astype(np.float)#pd.to_numeric(genotypes)
    #print(genotypes)
    #noisy_genotypes=mask_function(dataframe=genotypes,noise=0.2,batch_size=100)
    #print(noisy_genotypes)
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
    #df=genotypes.join(y)
    #print(df)
    #train, test = train_test_split(df, test_size=0.60, random_state=42)#40% used in training
    train, test = train_test_split(genotypes, test_size=0.80, random_state=42)#40% used in training

    train_samples=train.index
    #noisy_train=mask_function(dataframe=train,noise=0.3,batch_sizes=100)

    vali,test=train_test_split(test,test_size=0.9,random_state=42)#
    vali_samples=vali.index
    test_samples=test.index
    #noisy_vali=mask_function(dataframe=vali,noise=0.3,batch_sizes=100)
    noisy_test=mask_function(dataframe=test,noise=0.3,batch_sizes=100)
    #print(train.head)
    #print(noisy_train.head)
    #print(train_samples)
    #print(test_samples)
    #print(train.loc[train_samples])
    #print(train.shape)#(4000, 1000
    #print(noisy_train.shape)
    #print(test.shape)#(5400, 1000)
    #print(vali.shape)#(600, 1000)
    #sys.exit(0)
    #print(train[samples])
    #print(test)
    #print(df)
    num_dum=100
    numSNPs=train.shape[1]
    print(numSNPs)

    pfull = {'lr': (0.07, 1.0, 5),#'lr': (0.1, 10),
    'first_neuron':[ 200,500,1000],
    'embedding_size':[250,500,750],
    'batch_size': [50,100],
    'epochs': [200],
    'noise':[0.3,0.2],
    'dropout': [0, 0.40],
    'hidden_layers':[0, 1, 2],
    'activation':[keras.activations.relu, keras.activations.elu],
    #'x_val_noise':noisy_vali.values,
    #'x_train_noise':noisy_train.values,
    'optimizer': [keras.optimizers.Adam,keras.optimizers.RMSprop,keras.optimizers.Adadelta],
    'loss': ["mse"],# categorical_crossentropy, logcosh],
    'last_activation': [keras.activations.relu,keras.activations.elu]}#,
    #'weight_regulizer':[None]}
    
    ptest={'lr': [0.1],#'lr': (0.1, 10),
    'first_neuron':[ 2000],
    'embedding_size':[1000],
    'batch_size': [100],
    'epochs': [200],
    'noise':[0.1],
    'dropout': [0.20],
    'hidden_layers':[ 1],
    'activation':[keras.activations.tanh],
    #'x_val_noise':noisy_vali.values,
    #'x_train_noise':noisy_train.values,
    'optimizer': [keras.optimizers.Adam],
    'loss': [keras.losses.cosine_proximity],# keras.losses.cosine_proximity categorical_crossentropy, logcosh],
    'last_activation': [keras.activations.relu]}#
    
    print("scanning")
    p = {'lr': (0.1, 1.0, 3),#'lr': (0.1, 10),
    'first_neuron':[ 300],
    'embedding_size':[200],
    'batch_size': [50],
    'epochs': [200],
    'noise':[0.5],
    'dropout': [0,],
    'hidden_layers':[ 1, 2],
    #'x_val_noise':noisy_vali.values,
    #'x_train_noise':noisy_train.values,
    'optimizer': [keras.optimizers.Nadam],
    'activation':[keras.activations.relu],
    'loss': ["mse"],# categorical_crossentropy, logcosh],
    'last_activation': [keras.activations.elu]}#,
    
    
    
    from talos.utils.gpu_utils import force_cpu

    # Force CPU use on a GPU system
    force_cpu()
    #h = ta.Scan(x=train.values,x_val=vali.values, y=train.values,y_val=vali.values, params=pfull, model=dae_model_hl,
    #dataset_name=model_name,#'dae_test_hiddenlay_model',
    #experiment_no='4',
    #grid_downsample=0.012)
    
    h = ta.Scan(x=train.values,x_val=vali.values, y=train.values,y_val=vali.values, params=ptest, model=dae_model_hl,
    dataset_name=model_name,#'dae_test_hiddenlay_model',
    experiment_no='5')
    #3grid_downsample=0.012)

    #h = ta.Scan(x=train.values,x_val=vali.values, y=train.values,y_val=vali.values, params=pfull, model=dae_model,
    #dataset_name='dae_fullsize_model',
    #experiment_no='2',
    #grid_downsample=0.1)

    print(h.data.head()) # accessing the results data frame
    print(h.peak_epochs_df)# accessing epoch entropy values for each round
    print(h.details)# accessing summary detailspoch entropy values for each round
    r = ta.Reporting(h)
    
    # returns the saved models (json)
    #print(h.saved_models)
    #h.saved_models
    # returns the saved model weights
    #print(h.saved_weights)
    #h.saved_models
    #h.saved_weights


    # get the number of rounds in the Scan
    print("number of rounds {}".format(r.rounds()))
    print("highest validation accuracy {} ".format(r.high()))
    # get the highest result ('val_acc' by default)
    #print(r.high())
    # get the highest result for any metric
    print("highest accuracy {}".format(r.high('acc')))
    #print(r.high('acc'))
    #r.plot_hist()
    # get the round with the best result
    
    print("round with best score {}".format(r.rounds2high()))
    
    # get the best paramaters
    print("best paramaters {}".format(r.best_params()))
    print("deploying best model")
    deploy_name=experiment_name+"deploy"

    dep_model=ta.Deploy(h,deploy_name)
    # get correlation for hyperparameters against a metric
    #print(r.correlate('val_acc'))
    print("loading best model and embedding snps")
    #deploy_load=experiment_name+"deploy"
    print(deploy_name)#name of deploy
    zf = zipfile.ZipFile(deploy_name+".zip",'r')#accessing zip archive
    print("loading Json model")
    json_file = zf.open(deploy_name+'_model.json', 'r')#opening json file for architecture
    loaded_model_json = json_file.read()
    json_file.close()
    print("loading model weights")
    h5_file=zf.extract(deploy_name+'_model.h5',path='tmp/')#extracting weights (not sure how to do it a different way)
    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(h5_file)#model fully loaded
    #encodings=Model(inputs=test.values,outputs=loaded_model.get_layer('bottleneck').output)
    #print(encodings)
    print(loaded_model.layers)
    num_layers=len(loaded_model.layers)
    #input_img = Input(shape=(input_dim,))
    #encoder_layer1 = loaded_model.layers[0]
    #if num_layers>=5:
    #    encoder_layer2 = loaded_model.layers[1]
    inputs = Input(shape=(test.shape[1],))
    ncomps=loaded_model.get_layer('bottleneck').output_shape[1]
    print(ncomps)
    layer_names=[layer.name for layer in loaded_model.layers]
    #layer_names=[layer_names[i] for i in layer_names if re.search(layer_names[i],Denese
    
    encoder_layers = loaded_model.layers[0:int((float(num_layers)/2))]#accessing middle layer
    print(encoder_layers.name for layer in encoder_layers)
    #funct=[layersfor layers in encoders_layers
    #ifstatement for number of layers
    print("number of layers")
    print(len(encoder_layers))
    #output_layers=[]
    #for j in range(len(encoder_layers)):
    #    if j ==0:
    #        output_layers=encoder_layers[0](inputs)
    #    else:
    #        output_layers=output_layers(output_layers)
    #print(output_layers)
    #encoder=Model(inputs=inputs, outputs=output_layers)
    #
    if len(encoder_layers)==5:
        print("5 layers")
        encoder = Model(inputs=inputs, outputs=encoder_layers[4](encoder_layers[3](encoder_layers[2](encoder_layers[1](encoder_layers[0](inputs))))))
    elif len(encoder_layers)==4:
        print("4 layers")
        encoder = Model(inputs=inputs, outputs=encoder_layers[3](encoder_layers[2](encoder_layers[1](encoder_layers[0](inputs)))))
    elif len(encoder_layers)==7:
        print("7 layers")
        encoder = Model(inputs=inputs, outputs=encoder_layers[6](encoder_layers[5](encoder_layers[4](encoder_layers[3](encoder_layers[2](encoder_layers[1](encoder_layers[0](inputs))))))))
    elif len(encoder_layers)==6:
        print("6 layers")
        encoder = Model(inputs=inputs, outputs=encoder_layers[5](encoder_layers[4](encoder_layers[3](encoder_layers[2](encoder_layers[1](encoder_layers[0](inputs)))))))
    elif len(encoder_layers)==3:
        print("3 layers")
        encoder = Model(inputs=inputs, outputs=encoder_layers[2](encoder_layers[1](encoder_layers[0](inputs))))
    elif len(encoder_layers)==2:
        print("2 layers")
        encoder = Model(inputs=inputs, outputs=encoder_layers[1](encoder_layers[0](inputs)))


    encoder.summary()
    encoded = encoder.predict(test)
    print("dimensions of test set are")
    print(test.shape)
    embed_cols = ['embed_%i' % i for i in range(1,ncomps+1,1)]
    print("dimension of embeddings from test set is")
    print(encoded.shape)
    embeddings_test=pd.DataFrame(data=encoded,index=test.index,columns=embed_cols)
    print(embeddings_test.head)
    print(embeddings_test.shape)
    embeddings_test.to_csv("/projects/jgrealey/embedding/ten_k_samples/results/dae/"+experiment_name+".csv")
    print("embeddings_saved at /projects/jgrealey/embedding/ten_k_samples/results/dae/{}.csv".format(experiment_name))
    '''
    we will need to load the best model here
    encoder = Model(m.input, m.get_layer('bottleneck').output)
    encodings = encoder.predict(test)  # bottleneck representation
    Renc = m.predict(test)        # reconstruction
    
    
    print(encodings) #encodings are the embeddings from the test set
    print(Renc)
    print(encodings.shape)#
    print(Renc.shape)
    '''
    #understand what is going on here 
    #let's do optimisation research - do we need grid search or AutoML?
    #

if __name__ == '__main__':
        main()
