import numpy
from PIL import Image
import os
import io
import random
import tensorflow.compat.v1 as tf
import tkinter
from tkinter import filedialog

tf.disable_v2_behavior()

dirString = r"C:\Users\Bryan\Desktop\toGit\Leaf Matcher\100 leaves plant species 2\data"
#Copy-paste the path to the '.../100 leaves plant species/data' folder path on your machine over the dirString string 

dirList = os.listdir(dirString)


dirListR=[]
for d in dirList:
    d2 = dirString + "\\" + d
    dirListR.append(d2)
    

fileList=[]

for f in dirListR:
    h = os.listdir(f)
    for y in h:
        if y[0]!='.':
            fileList.append(f+"\\"+y)
        
def extract_image(filename):
        im = Image.open(filename)
        im = im.convert('F')
        data = numpy.asarray(im, 'float32')
        data = data.flatten()
        data = numpy.multiply(data, 1.0 / 255.0)
        return data

def extract_label(filename):
    datar = int((fileList.index(filename))/16)
    data = [0] * 100
    data[datar] = 1
    return numpy.asarray(data)

# builds a batch of random images+labels from the set, reserving one of every category for the test batch
def build_batch(num):
    imResult=[]
    labResult=[]
 
    for x in range(0,num):
        r = random.choice([x for x in range(1600) if ((x + 1) % 16 != 0)])
        fileString = fileList[r]
        im = extract_image(fileString)
        lab = extract_label(fileString)
        imResult.append(im)
        labResult.append(lab)
    return imResult, labResult

# builds test batch
def test_data():
    imResult = []
    labResult = []
    use=[x for x in range(1600) if ((x + 1) % 16 == 0)]
    for x in use:
        fileString = fileList[x]
        im=extract_image(fileString)
        lab=extract_label(fileString)
        imResult.append(im)
        labResult.append(lab)
    return imResult, labResult


sess = tf.Session()


x = tf.placeholder(tf.float32, [None, 14400])
y_ = tf.placeholder(tf.float32, [None, 100])

W = tf.Variable(tf.ones([14400,100]))
b = tf.Variable(tf.ones([100]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(.0001).minimize(cross_entropy)


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


resarr = []
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)


cont = True
while(cont == True): 
    batch = build_batch(50)     
    sess.run(train_step,feed_dict={x: batch[0], y_: batch[1]})  
    res = (sess.run(accuracy,feed_dict={x: batch[0], y_: batch[1]}))
    print(res)
    resarr.append(res)
    #check the previous iterations for accuracy stasis/improvement, otherwise end training
    if(len(resarr) >= 5):
        test = resarr[len(resarr) - 5:]
        prev = 1
        for val in test:
            if((val - prev) > 0 or val < 0.70):
                break
            else:
                prev = val
        else:
            cont = False


#test against the reserved images; tends 0.5 accuracy with these settings 
batch1 = test_data()
print("Accuracy: " + str(sess.run(accuracy,feed_dict={x: batch1[0], y_: batch1[1]})))
print("Please select a leaf image for identification")
for p in range(1):
    file_path = filedialog.askopenfilename()
    file_path = file_path.replace("/","\\")
    print("Leaf Selected: " + file_path[len(dirString)+1:-10].split("\\")[0])
    prediction = (sess.run(tf.argmax(y, 1),feed_dict={x: [extract_image(file_path)], y_: [extract_label(file_path)]}))
    print(prediction)
    answer = dirList[prediction[0]]
    print("Neural network identification: " + answer)


sess.close()

