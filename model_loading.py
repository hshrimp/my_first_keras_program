from keras.models import load_model
from Classification_20news_by_keras import to_tensor,processing_text_dataset
import numpy as np
from sklearn import  metrics
from keras.utils import plot_model
# y_true = [0, 1, 2, 2, 0]
# y_pred = [0, 0, 2, 2, 0]
# target_names = ['class 0', 'class 1', 'class 2']
# print(metrics.classification_report(y_true, y_pred))

model=load_model('my_model.h5')
plot_model(model,to_file='20_newsgroup.png',show_shapes=True)
texts,labels,labels_index=processing_text_dataset()
word_index, x_train, y_train, x_val, y_val=to_tensor(texts,labels)
print(model.evaluate(x_val,y_val))
print(model.metrics_names)

yp = model.predict(x_val, batch_size=64, verbose=1)
ypreds = np.argmax(yp, axis=1)
y_val=np.argmax(y_val,axis=1)
print(metrics.classification_report(y_val, ypreds))
print('\n正确率：', metrics.accuracy_score(y_val, ypreds))
print('\n准确率macro：', metrics.precision_score(y_val, ypreds,average='macro'), '  召回率macro：', metrics.recall_score(y_val, ypreds,average='macro'))
print('\n准确率micro：', metrics.precision_score(y_val, ypreds,average='micro'), '  召回率micro：', metrics.recall_score(y_val, ypreds,average='micro'))

print('F1_macro:', metrics.f1_score(y_val, ypreds,average='macro'))
print('F1_micro:', metrics.f1_score(y_val, ypreds,average='micro'))
#for i in range(len(x_val)):
   # print(np.dot(model.predict(x_val[i]),np.array(y_val[i])).T)
