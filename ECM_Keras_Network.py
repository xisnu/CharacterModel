from __future__ import print_function
from keras.layers import Dense, LSTM, Bidirectional
from keras.models import Sequential,load_model
from ECM_Read_Data import *
from keras.optimizers import RMSprop

class ECM_Keras:
    def __init__(self,maxtimesteps,inputdim,outputdim):
        self.timesteps=maxtimesteps
        self.input_dim=inputdim
        self.nb_classes=outputdim
        self.network=Sequential()
        print('Empty Model created')

    def create_network(self,load=False):
        if(load):
            self.network=load_model("Model/ECM")
            print("Existing network loaded")
        else:
            self.network.add(LSTM(64,input_shape=[self.timesteps,self.input_dim],return_sequences=True))
            self.network.add(Bidirectional(LSTM(128,return_sequences=True)))
            self.network.add(Bidirectional(LSTM(256)))
            self.network.add(Dense(self.nb_classes,activation='softmax'))
            opt=RMSprop(lr=0.001)
            self.network.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
            print('Network is configured')
            self.network.summary()

    def train_network(self,nbepochs,data_x,data_y,batchsize):
        for e in range(nbepochs):
            print('Epoch ',e)
            self.network.fit(data_x,data_y,batch_size=batchsize,verbose=1,epochs=1)
            self.network.save("Model/ECM")

    def load_network(self,network_file):
        self.network = load_model(network_file)
        print("Existing network loaded")

    def predict(self,test_x,characters):
        # test_x is a single word of shape [MWL,Nc]
        # Reshape it to [1,MWL,Nc]
        #test_x=np.expand_dims(test_x,axis=0)
        output=self.network.predict(test_x)
        print("Output shape ",output.shape)
        probabilities=output[0]
        predicted_characters=decode_probabilities(probabilities,characters,top=3)
        return predicted_characters

def main():
    data_x,data_y,charset=load_data("Data/ECM_Target_Vocabulary.txt","Data/ECM_Target_Characters.txt","Data/ECM_Train_Data.txt")
    mts=data_x.shape[1]
    classes=data_y.shape[1] 

    nw=ECM_Keras(mts,classes,classes)
    nw.create_network(load=True)
    nw.train_network(100,data_x,data_y,128)
    #nw.predict("Model/ECM",data_x[8])

#main()
