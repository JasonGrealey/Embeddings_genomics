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
print("importing zipfile")
import zipfile
print("importing matplotlib")
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import datetime

def print_info(archive_name):
    zf = zipfile.ZipFile(archive_name)
    for info in zf.infolist():
        print(info.filename)
        print( '\tComment:\t', info.comment)
        print( '\tModified:\t', datetime.datetime(*info.date_time))
        print( '\tSystem:\t\t', info.create_system, '(0 = Windows, 3 = Unix)')
        print( '\tZIP version:\t', info.create_version)
        print( '\tCompressed:\t', info.compress_size, 'bytes')
        print( '\tUncompressed:\t', info.file_size, 'bytes')
        #print

def main():
    if len(sys.argv)!=2:
        print("number of required inputs is no reached")
        print("exiting program")
        sys.exit(0)
    #test_deploy is experiment
    experiment=str(sys.argv[1])#'dae_test_model_1.csv'
    load=experiment+".zip"
    print(experiment)
    #zf = zipfile.ZipFile(load, 'r')
    #print(zf.namelist())
    #print_info(load)
    zf = zipfile.ZipFile(experiment+".zip",'r')
    print(zf.namelist())
    #print(zf)
    # load json and create model
    print("loading Json model")
    json_file = zf.open(experiment+'_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    print("loading model weights")
    # load weights into new model
    #h5_file=zf.open(experiment+'_model.h5','r')#,path='tmp/')
    h5_file=zf.extract(experiment+'_model.h5',path='tmp/')
    #h5_file=zf.read(experiment+'_model.h5')#,path='tmp/')

    #print(h5_file)
    #h5_file_to_read=h5_file.read()
    #print(h5_file_to_read)
    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(h5_file)#experiment+"_model.h5")
    #h5_file.close()
    print("Loaded model from disk")
    #loaded_model=keras.models.load_weights(experiment+'_model.h5')
    print(loaded_model.layers)
    num_layers=len(loaded_model.layers)
    encodertest=loaded_model.layers[0:(int((float(num_layers)/2) +1))]
    print(encodertest)
    input_img = Input(shape=(input_dim,))
    #encoder_layer1 = loaded_model.layers[0]
    #if num_layers>=5:
    #    encoder_layer2 = loaded_model.layers[1]
    #encoder_layer3 = autoencoder.layers[int((float(numlayers)/2) +1)]#accessing middle layer
    #encoder = Model(input_img, encoder_layer3(encoder_layer2(encoder_layer1(input_img))))
    encoder=Model(input_img,encodertest(input_img))
    encoder.summary()


if __name__ == "__main__":
    main()
