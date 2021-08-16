#GRID STRUCTURE
from tkinter import *
import PIL
from PIL import Image, ImageDraw,ImageOps
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
from numpy import asarray
import os
import collections

img=[]
x=[]
model=tf.keras.models.load_model('mnist_99_12.h5')



def save():
    global image_number
    filename = f'image_0.png'   # image_number increments by 1 at every save
    image1.save(filename)
    resize_ops()
    predict_asarrray()


def predict_asarrray():
    img=image.load_img('image_resize_0.png')
    img=img.resize((28,28))
    img=img.convert('L')
    img=np.array(img)
    #img2=Image.fromarray(img)  
    img=img.reshape(1,28,28,1)
    img=img/255.0
    #plt.imshow(img2, cmap = plt.get_cmap('gray'))
    #plt.show()    
    result=model.predict([img])[0]
    print(result)
    print(np.argmax(result),"   " ,max(result))
    plot(result)


def  predicted():
    images=image.load_img('image_resize_0.png',target_size=(28,28))
    x = image.img_to_array(images)
    plt.imshow(x, cmap = plt.get_cmap('gray'))
    plt.show()
    x = x.reshape(-1,28,28,1)
    x = x/255.0
    img.append(x)
    y=model.predict(x)
    print('Predicted:', np.argmax(y))

def plot(result):
    fig1,ax1=plt.subplots()
    labels=[0,1,2,3,4,5,6,7,8,9]
    b=dict(zip(result,labels))
    ordered_b=collections.OrderedDict(reversed(sorted(b.items())))
    ordered_b.popitem()
    ordered_b.popitem()
    ordered_b.popitem()
    ordered_b.popitem()
    ordered_b.popitem()
    sizes=ordered_b.keys()
    labels=ordered_b.values()
    colors=["rosybrown","lightcoral","indianred","firebrick","maroon"]
    plt.pie(sizes,colors=colors,labels=labels,autopct='%1.2f%%')
    plt.legend(labels,loc="best")
    ax1.axis('equal')
    plt.show()
    
def clear():
    #clear image and new canvas
    os.remove('/home/milano/Desktop/digit_recog/image_0.png')
    os.remove('/home/milano/Desktop/digit_recog/image_resize_0.png')
    cv.delete('all')

def resize_28():
  img=Image.open("/home/milano/Desktop/digit_recog/image_0.png")
  resized_img=img.resize((28,28))
  img.thumbnail((28,28), Image.ANTIALIAS)
  resized_img.save("/home/milano/Desktop/digit_recog/resized_img_0.png",quality=100, optimize=True)

def resize_ops():
  im1=Image.open("image_0.png")
  im2 = ImageOps.fit(im1, (28, 28),Image.ANTIALIAS)
  im2.save('image_resize_0.png')

def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint_oval)
    lastx, lasty = e.x, e.y


def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    cv.create_line((lastx, lasty, x, y), width=8)
    #  --- PIL
    draw.line((lastx, lasty, x, y), fill='white', width=8)
    lastx, lasty = x, y

def paint_oval(e):
    r=8
    x, y = e.x, e.y
    global lastx, lasty
    cv.create_oval(lastx-r,lasty-r,x+r,y+r, fill='white')
    draw.line((lastx-r,lasty-r,x+r,y+r), fill='white')
    lastx, lasty = x, y


root = Tk()
root.title("HandWritten Digit Recognition")
lastx, lasty = None, None
image_number = 0


cv = Canvas(root, width=250, height=300, bg='black')
  # --- PIL
image1 = PIL.Image.new('RGB', (250, 300), 'black')
draw = ImageDraw.Draw(image1)


cv.bind('<1>', activate_paint)
cv.pack(expand=YES, fill=BOTH)

btn_save = Button(text="PREDICT", command=save)
btn_save.pack(side=RIGHT)

reset=Button(text='Reset canvas',command=clear)
reset.pack(side=LEFT)

root.mainloop()