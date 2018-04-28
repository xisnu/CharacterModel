
## First lets explore Data folder
1. We have **English_Words.txt** file that contains a list of English words. We will pick up few from them to train our model. For  this we have a function **read_main_file()** inside **ECM_Read_Data.py**. This function will write the random words in a file *ECM_Target_Vocabulary.txt*
2. Now we use **find_charset_from_vocabulary()** to generate a list of distinct characters from **ECM_Target_Vocabulary.txt** and those characters are also written in a file **ECM_Target_Characters.txt** . From now on we will work with _ECM_Target_Characters.txt_ and _ECM_Target_Vocabulary.txt_
3. Now we create training data from these two files using **prepare_training_pairs()** and another file is created named as **ECM_Train_Data.txt**. Each row of this file is a single training sample having X and Y separated by **\t**. For each word in vocabulary we have prepared multiple samples. Suppose the word is "network", then we get following pairs, <n,e>,<ne,t>,<net,w>,<netw,o> and so on.
4. Another function **load_data()** return two arrays for X and Y from the training data file.
#
**This code is only for English words where the characters are easily separable as they are elements of the array, if your words are written in Unicode or some custom format you may need to device some method to segregate them. Alter corresponding function** 
#
## Now check the Model
1. A RNN based network is implemented using Keras 2.X with Tensorflow 1.6 in backend. Python version is 2.7. 
2. The model is implemented in **ECM_Keras_Network.py** file
3. ```nw=ECM_Keras(mts,classes,classes)``` initiates an empty model
4. ```nw.create_network(load=False)``` creates a network. If ```load=True``` it will load a pre-trained model from **Model** directory
5. ```nw.train_network(100,data_x,data_y,128)``` trains the network for 100 iterations and 128 batch size
6. ```nw.predict("Model/ECM",test_data)``` predicts few possible output characters given *a single sequence* ```test_data```. Notice the inside call ```predicted_characters=decode_probabilities(probabilities,characters,top=3)```, change the value of ```top``` to generate more or less suggestions.
#
## A simple interface
A simple interface for spelling suggestion is given in **ECM_Type.py**. That's not much but serves the purpose.


