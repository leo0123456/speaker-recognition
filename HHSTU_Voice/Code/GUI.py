from tkinter import *
from pywav import py_wav
from SpeakerIdentifier1 import spk_id


spk_per = spk_id()
luzhi = py_wav()

def va():
    luzhi.record_audio("output.wav",record_second=9)
def da():
    global a
    a = spk_per.person_identifer()
    b = Label(root, text=a)
    b.pack()

root = Tk()
root.title('说话人识别')
root.geometry('200x300')
Button(root, text="录音", command=va,height=3,width=3).place(x=79, y=130)
Button(root, text="开始识别", command=da).place(x=65, y=90)
w = Label(root, text="说话人为：" )

w.pack()

root.mainloop()

