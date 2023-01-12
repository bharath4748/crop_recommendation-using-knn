#for connecting dataset

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler#to standardize the input in a uniform manner
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
#to test the data we use below 3 variables
from sklearn.metrics import accuracy_score 
import math
dataset = pd.read_csv('Crop_recommendation.csv')
print(len(dataset))
print(dataset.head())
print(dataset.keys())
print(dataset.label.unique())
sc_X= StandardScaler()
X = dataset.iloc[:,0:7]#including all rows and including only columns between 0 to 8 excluding 8 independent varaible
Y = dataset.iloc[:,7]#dependent variable
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=0,test_size=0.3)
classifier = KNeighborsClassifier(n_neighbors = 3,metric = 'euclidean')
classifier.fit(X_train, Y_train)
y_pred =pd.DataFrame(classifier.predict(X_test))
print(y_pred)

# for inuput

dataset.N=dataset.N.astype(int)
dataset.P=dataset.P.astype(int)
dataset.K=dataset.K.astype(int)
dataset.temperature=dataset.temperature.astype(float)
dataset.humidity=dataset.humidity.astype(float)
dataset.ph=dataset.ph.astype(float)
dataset.rainfall=dataset.rainfall.astype(float)


from tkinter import*
window=Tk()
N =IntVar()
P =IntVar()
K =IntVar()
Tem=DoubleVar()
Hum=DoubleVar()
PH=DoubleVar()
Rain =DoubleVar() 


def submitForm():
    a=N.get()
    b=P.get()
    c=K.get()
    d=Tem.get()
    e=Hum.get()
    f=PH.get()
    g=Rain.get()
    inp = np.asfarray([[a], [b], [c], [d], [e], [f], [g]])
    inp = inp.reshape(1, -1) 
    print('Crop is :', classifier.predict(inp))
    z=classifier.predict(inp)
    label2=Label(window,text='Crop is:',font=("Helvetica:",14)).grid(row=12,column=0)
    label2=Label(window,text=z,font=("Helvetica:",14)).grid(row=12,column=1)

label1=Label(window,text='\t CROP RECCOMENDATION',font=("Helvetica:",19)).grid(row=0,column=0)

    
label_Nitrogen=Label(window,text=" ENTER NITROGEN VALUE: ").grid(row=1,column=0)
Nitrogen=Entry(window,textvariable=N).grid(row=1,column=1)

label_Phosporous=Label(window,text=" ENTER PHOSPOROUS VALUE: ").grid(row=2,column=0)
Phosporous=Entry(window,textvariable=P).grid(row=2,column=1)

label_Potassium=Label(window,text="ENTER POTASSIUM VALUE: ").grid(row=3,column=0)
Potassium=Entry(window,textvariable=K).grid(row=3,column=1)

label_Temperature=Label(window,text="ENTER TEMPERATURE").grid(row=4,column=0)
Temperature=Entry(window,textvariable=Tem).grid(row=4,column=1)

label_Humidity=Label(window,text="ENTER HUMIDITY").grid(row=5,column=0)
Humidity=Entry(window,textvariable=Hum).grid(row=5,column=1)

label_PH=Label(window,text="ENTER PH VALUE OF SOIL").grid(row=6,column=0)
PH=Entry(window,textvariable=PH)
PH.grid(row=6,column=1)

label_Rainfall=Label(window,text="ENTER RAINFALL VALUE").grid(row=7,column=0)
Rainfall=Entry(window,textvariable=Rain).grid(row=7,column=1)



submit=Button(window,text="submit",command=submitForm).grid(row=8,column=0)

window.mainloop()

#for aacuray

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

print(confusion_matrix(Y_test, y_pred))

print(classification_report(Y_test, y_pred))