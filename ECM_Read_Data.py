from __future__ import print_function
import numpy as np
from random import shuffle

def read_main_file(filename,nbwords):
    # Read a file filename containing English Words, each line contain a word
    # Put the words in an array
    # Pick nbwords number of words from the array randomly
    # Write the random words in a file ECM_Target_Vocabulary.txt
    f=open(filename)
    line=f.readline()
    all_words=[]
    while line:
        info=line.strip("\n")# remove \n from the end of word
        all_words.append(info)
        line=f.readline()
    print("All words are gatherd in array")
    f.close()

    total = len(all_words)
    shuffle(all_words)
    print("All words shuffled, Total ",total)

    nb=min(total,nbwords)
    random_words=all_words[:nb]
    print("Random words selected")

    f=open("Data/ECM_Target_Vocabulary.txt","w")
    for w in random_words:
        f.write(w+"\n")
    f.close()
    print("Random words written in file")

def find_charset_from_vocabulary(vocabulary):
    # Read a vocabulary file
    # Find the distinct characters
    # Write the distinct characters in Character set file
    f=open(vocabulary)
    line=f.readline()
    all_characters=[]
    while line:
        word=line.strip("\n")
        for ch in word:
            all_characters.append(ch)
        line=f.readline()
    f.close()
    print('All characters are written')
    charset=list(set(all_characters))
    print("Distinct characters found")
    f=open("Data/ECM_Target_Characters.txt","w")
    for ch in charset:
        f.write(ch+"\n")
    f.close()
    print("Characters are written in file")

def prepare_training_pairs(vocabularyfile):
    # Read all the words
    # Make train pairs ["Network" prepare pairs n-e , ne-t, net-w, netw-o, netwo-r, networ-k]
    # Write pairs in file "ECM_Train_Data
    f=open(vocabularyfile)
    wf=open("Data/ECM_Train_Data.txt","w")
    line=f.readline()
    while line:
        word=line.strip("\n")
        nbchars=len(word)
        for i in range(nbchars-1):
            x=word[:i+1]
            y=word[i+1]
            wf.write(x+"\t"+y+"\n")
        line=f.readline()
    print("Training data ready")
    f.close()
    wf.close()

def load_data(vocabulary,charset,traindata):
    # Read vocabulary to find maximum word length MWL
    # Read charset to prepare one hot find Nc characters
    # Read traindata for x,y pairs find total N data
    # make an array for X [N,MWL, Nc] and Y [N,Nc]
    f=open(vocabulary)
    line=f.readline()
    MWL=0
    while line:
        word=line.strip("\n")
        NcW=len(word) # Number of characters in this word
        if(NcW>MWL):
            MWL=NcW
        line=f.readline()
    f.close()
    print("Vocabulary processed , MWL=",MWL)

    characters=load_charset(charset)
    Nc=len(characters)
    print("Character set processed , Nc=",Nc)

    f=open(traindata)
    line=f.readline()
    data_x=[]
    data_y=[]
    while line:
        info=line.strip("\n").split("\t")
        x=info[0] # a sequence of characters
        y=info[1] # a single character

        y_one_hot=np.zeros([Nc])
        y_index=characters.index(y)
        y_one_hot[y_index]=1

        x_one_hot=np.zeros([MWL,Nc])
        char_pos=0
        for ch in x:
            ch_index=characters.index(ch)
            x_one_hot[char_pos][ch_index]=1
            char_pos+=1

        data_x.append(x_one_hot)
        data_y.append(y_one_hot)

        line=f.readline()
    f.close()
    data_x=np.asarray(data_x)
    data_y=np.asarray(data_y)
    print("X and Y data ready , X=",data_x.shape," Y=",data_y.shape)
    return data_x,data_y,characters

def decode_probabilities(one_hot,charset,top=1):
    # A one hot vector is given , each one is a probability
    # returns corresponding character based on charset
    #print(one_hot)
    if(top>1):
        chars=[]
        temp=list(one_hot)
        temp.sort(reverse=True)
        probs=temp[:top+1]
        for t in range(top):
            pos=np.where(one_hot==probs[t])
            pos=pos[0][0]
            char=charset[pos]
            chars.append(char)
    else:
        max_index=np.argmax(one_hot)
        chars=charset[int(max_index)]
    #print('Max Index',max_index)
    return chars

def encode_string(string,charset,MWL):
    # A string of characters is given Ns is number of characters
    # Return one hot representaion [MWL,Nc] Nc is number of characters in charset
    Nc=len(charset)
    one_hot=np.zeros([1,MWL,Nc])
    pos=0
    for ch in string:
        index=charset.index(ch)
        one_hot[0][pos][index]=1
    return one_hot

def load_charset(charset):
    # Read charset file
    # Return array of characters
    characters=[]
    f=open(charset)
    line=f.readline()
    while line:
        ch=line.strip("\n")
        characters.append(ch)
        line=f.readline()
    f.close()
    return characters

#main_file="Data/English_Words.txt"
# read_main_file(main_file,1000)
# find_charset_from_vocabulary("Data/ECM_Target_Vocabulary.txt")
# prepare_training_pairs("Data/ECM_Target_Vocabulary.txt")
#load_data("Data/ECM_Target_Vocabulary.txt","Data/ECM_Target_Characters.txt","Data/ECM_Train_Data.txt")





