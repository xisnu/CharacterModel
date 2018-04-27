from Tkinter import *
from ECM_Keras_Network import *

mts=14
characters=load_charset("Data/ECM_Target_Characters.txt")
classes=len(characters)
nw=ECM_Keras(mts,classes,classes)
nw.load_network("Model/ECM")

window = Tk()
window.title("DeepSpell")
window.geometry('350x200')
lbl = Label(window, text="Type here")
lbl.grid(column=0, row=0)
txt = Entry(window, width=20)
txt.grid(column=1, row=0)
lbl2 = Label(window, text="Suggestions")
lbl2.grid(column=0, row=1,sticky=E+W)


def suggest():
    inputstring = txt.get()
    one_hot_input=encode_string(inputstring,characters,mts)
    chars=nw.predict(one_hot_input,characters)
    result="Suggestion:\n"
    for ch in chars:
        suggestion=inputstring+ch
        result+=suggestion+"\n"
    lbl2.configure(text=result)


btn = Button(window, text="Suggest", command=suggest)

btn.grid(column=2, row=0)

window.mainloop() 
