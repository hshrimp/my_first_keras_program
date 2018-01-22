from keras.models import load_model
from arc128test.a14 import to_tensor,processing_text_dataset
import numpy as np
from sklearn import metrics
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
model=load_model('a14(2).h5')
# model.summary()
# plot_model(model,to_file='ARC_12.png',show_shapes=True)
train_label, x1text_train, x2text_train = processing_text_dataset('train')
test_label, x1text_test, x2text_test = processing_text_dataset('test')
word_index, x1_train, x2_train, y_train, x1_val, x2_val, y_val = to_tensor(x1text_train, x2text_train, train_label,
                                                                           x1text_test, x2text_test, test_label)

model.fit([x1_train, x2_train], y_train, batch_size=200, epochs=500, verbose=2,
          callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2)],
          validation_data=([x1_val, x2_val], y_val))
model.save('a14(2).h5')
print(model.evaluate([x1_val,x2_val],y_val))
print(model.metrics_names)

yp = model.predict([x1_val,x2_val], batch_size=64, verbose=1)
y=np.zeros(yp.shape)
for item in range(len(y)):
    if(yp[item]>=0.5):
        y[item]=1
else :y[item]=0
ypreds=y
print('\n',metrics.classification_report(y_val, ypreds,digits=4))
print('\n正确率：', metrics.accuracy_score(y_val, ypreds))
print('\n准确率macro：', metrics.precision_score(y_val, ypreds,average='macro'), '  召回率macro：', metrics.recall_score(y_val, ypreds,average='macro'))
print('\n准确率micro：', metrics.precision_score(y_val, ypreds,average='micro'), '  召回率micro：', metrics.recall_score(y_val, ypreds,average='micro'))

print('F1_macro:', metrics.f1_score(y_val, ypreds,average='macro'))
print('F1_micro:', metrics.f1_score(y_val, ypreds,average='micro'))
#for i in range(len(x_val)):
   # print(np.dot(model.predict(x_val[i]),np.array(y_val[i])).T)
