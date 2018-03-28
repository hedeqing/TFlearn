import numpy as np
import tflearn

from  tflearn.datasets import  titanic

titanic.download_dataset('titanic_dataset.csv')

from tflearn.data_utils import load_csv


datas,labels=load_csv('titanic_dataset.csv',target_column=0,categorical_labels=True,n_classes=2)
def preprocess(data,columns_ignore):
    for id in sorted(columns_ignore,reverse=True):
        [r.pop(id) for r in data]
    for i in range(len(data)):
        if data[i][1] == 'fem1ale':
            data[i][1] = 1
        else:
            data[i][1] = 0
    return np.array(data, dtype=np.float32)

to_ignore = [1, 6]
data = preprocess(datas, to_ignore)

net=tflearn.input_data(shape=[None,6])
net=tflearn.fully_connected(net,32)
net=tflearn.fully_connected(net,32)
net=tflearn.fully_connected(net,2,activation='softmax')
net=tflearn.regression(net)

model=tflearn.DNN(net)
model.fit(data,labels,n_epoch=10,batch_size=16,show_metric=True)

dicapro=[3,'jack dason','male',19,0,0,'N/A',5.0000]
winslet=[1,'Dewitt Bukater','female',17,0,0,'N/A',100.0000]

dicapro,winslet=preprocess([dicapro,winslet],to_ignore)

pred=model.predict([dicapro,winslet])

print(pred[0][1])
print(pred[1][1])
