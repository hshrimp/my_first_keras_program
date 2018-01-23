#from __future__ import print_function

import os,sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from  keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense,Dropout,Input,Flatten,Lambda
from keras.layers import Conv1D,MaxPooling1D,Embedding,Masking,AveragePooling1D
from keras.models import Model
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping
#from keras import backend as K
from sklearn import metrics
import tensorflow as tf
from keras.optimizers import SGD
from keras.utils import  plot_model
from keras import regularizers
BASE_DIR='../'
glove_dir=BASE_DIR+'glove.6B/'
text_data_dir=BASE_DIR+'data/msr_paraphrase_'
max_sequence_length=35
max_nb_words=20000
embedding_dim=50
#validation_split=0.2

def index_word_vectors():
    print('indexing word vectors.')

    embeddings_index={}
    f=open(os.path.join(glove_dir,'glove.6B.50d.txt'))
    for line in f:
        values=line.split()
        word=values[0]
        coefs=np.asarray(values[1:],dtype='float32')
        embeddings_index[word]=coefs
    f.close()

    print('found %s word vectors.' %len(embeddings_index))
    return embeddings_index

# def processing_text_dataset():
#     print('processing text dataset')
#
#     texts=[]
#     labels_index={}
#     labels=[]
#     for name in sorted(os.listdir(text_data_dir)):
#         path = os.path.join(text_data_dir,name)
#         if os.path.isdir(path):
#             label_id=len(labels_index)
#             labels_index[name]=label_id
#             for fname in sorted(os.listdir(path)):
#                 if fname.isdigit():
#                     fpath=os.path.join(path,fname)
#                     if sys.version_info<(3,):
#                         f=open(fpath)
#                     else:
#                         f=open(fpath,encoding='latin-1')
#                     t=f.read()
#                     i=t.find('\n\n')
#                     if 0<i:
#                         t=t[i:]
#                     texts.append(t)
#                     f.close()
#                     labels.append(label_id)
#
#     print('found %s texts.' %len(texts))
#     return texts,labels,labels_index
def processing_text_dataset(str):
    print('processing text dataset')
    reader = open(text_data_dir + str + '.txt', 'r')
    unprocessed_data = np.array([example.split("\t") for example in reader.readlines()])[1:]
    #标签label值
    new_example = []
    #句子1
    raw_sentence_1=[]
    #句子2
    raw_sentence_2=[]
    for (i, example) in enumerate(unprocessed_data):
        new_example.append(float(example[0]))
        raw_sentence_1.append(example[3])
        raw_sentence_2.append(example[4])
    print('found %s sentences.' % len(raw_sentence_1)*2)
    return new_example,raw_sentence_1,raw_sentence_2


def to_tensor(texts11,texts12,labels1,texts21,texts22,labels2):
    tokenizer=Tokenizer(num_words=max_nb_words)
    tokenizer.fit_on_texts(texts11+texts12+texts21+texts22)
    sequences11=tokenizer.texts_to_sequences(texts11)
    #tokenizer.fit_on_texts(texts2)
    sequences12 = tokenizer.texts_to_sequences(texts12)
    sequences21 = tokenizer.texts_to_sequences(texts21)
    # tokenizer.fit_on_texts(texts2)
    sequences22 = tokenizer.texts_to_sequences(texts22)
    #
    # count_len(sequences11)
    # count_len(sequences12)
    # count_len(sequences21)
    # count_len(sequences22)

    word_index=tokenizer.word_index
    print('found %s unique tokens.'%len(word_index))
    #text11的张量
    data11=pad_sequences(sequences11,max_sequence_length)
    #text12的张量
    data12 = pad_sequences(sequences12, max_sequence_length)
    data21 = pad_sequences(sequences21, max_sequence_length)
    # text22的张量
    data22 = pad_sequences(sequences22, max_sequence_length)
    # labels1=to_categorical(np.asarray(labels1))
    # labels2 = to_categorical(np.asarray(labels2))
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)
    print('shape of the data11 tensor:',data11.shape)
    print('shape of the data12 tensor:', data12.shape)
    print('shape of label1 tensor:',labels1.shape)
    print('shape of the data21 tensor:', data21.shape)
    print('shape of the data22 tensor:', data22.shape)
    print('shape of label2 tensor:', labels2.shape)
    return word_index, data11,data12,labels1,data21,data22,labels2

#compute the max length of sentences
def count_len(seg):
    count=0
    for item in seg:
        if(len(item)>count):
            count=len(item)
    print('count=',count)

def prepare_embedding(word_index,embeddings_index):
    print('preparing embedding matrix.')
    # num_words=min(max_nb_words,len(word_index))
    # embedding_matrix=np.zeros((num_words,embedding_dim))
    # for word,i in word_index.items():
    #     if i>=max_nb_words:
    #         continue
    #     embedding_vector=embeddings_index.get(word)
    #     if embedding_vector is not None:
    #         embedding_matrix[i]=embedding_vector
    num_words=len(word_index)
    embedding_matrix=np.zeros((num_words+1,embedding_dim))
    for word,i in word_index.items():
        if(i>=len(word_index)):
            continue
        embedding_vector=embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector
    return num_words+1 , embedding_matrix

def train(num_words,embedding_matrix,x1_train,x2_train,y_train,x1_val,x2_val,y_val):
    #weight加入glove预先训练好的向量，trainable设置成这部分参数不训练（glove预先训练好了，也可以添加自己的向量）
    embedding_layer=Embedding(num_words,embedding_dim,weights=[embedding_matrix],input_length=max_sequence_length,trainable=False)
    print('training model.')

    sequence_input1=Input(shape=(max_sequence_length,),dtype='int32')
    embedded_sequences1=embedding_layer(sequence_input1)
    #embedded_sequences1 =Masking(mask_value=0)(embedded_sequences1)
    x1=Conv1D(500,3,activation='relu',kernel_regularizer=regularizers.l2(0.01))(embedded_sequences1)
    #x1=AveragePooling1D(3)(x1)
    #x1=Dropout(0.5)(x1)
    x1=Conv1D(500,3,activation='relu',kernel_regularizer=regularizers.l2(0.01))(x1)
    x1=Conv1D(500,3,strides=3,activation='relu',kernel_regularizer=regularizers.l2(0.01))(x1)

    #x1 = AveragePooling1D(3)(x1)
    #x1 = Dropout(0.5)(x1)
    # x1 = Conv1D(200, 3, activation='relu')(x1)
    # x1 = MaxPooling1D(3)(x1)
    # x1 = Dropout(0.25)(x1)
    output=Flatten()(x1)

    shall_model=Model(sequence_input1,output)

    sequence_input1 = Input(shape=(max_sequence_length,), dtype='int32')
    sequence_input2 = Input(shape=(max_sequence_length,), dtype='int32')
    output1=shall_model(sequence_input1)
    output2=shall_model(sequence_input2)

    x = concatenate([output1,output2])
    #x = Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    # x = Dropout(0.5)(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dropout(0.25)(x)
    preds = Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l2(0.01))(x)
    #preds=Lambda(out)(preds)
    #preds=Lambda(lambda i:1.0 if(i[0]>=0.5) else 0.0)(preds)
    #preds = Lambda(K.argmax,output_shape=(1,),dtype='float64')(preds)
    model=Model([sequence_input1,sequence_input2],preds)

    model.summary()
    #model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['acc'])
    sgd = SGD(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['acc'])

    model.fit([x1_train,x2_train],y_train,batch_size=200,epochs=500, verbose=2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=2)],validation_data=([x1_val,x2_val],y_val))
    #plot_model(shall_model, to_file='model_1.png', show_shapes=True)
    plot_model(model,to_file='ARC_127test.png',show_shapes=True)

    yp = model.predict([x1_val,x2_val], batch_size=200, verbose=1)
    #ypreds = np.argmax(yp, axis=1)
    y=np.zeros(yp.shape)
    for item in range(len(y)):
        if(yp[item]>=0.5):
            y[item]=1
        else :y[item]=0
    ypreds=y
    print('\n',metrics.classification_report(y_val, ypreds))
    print('\n正确率：', metrics.accuracy_score(y_val, ypreds))
    print('\n准确率macro：', metrics.precision_score(y_val, ypreds, average='macro'), '  召回率macro：',
          metrics.recall_score(y_val, ypreds, average='macro'))
    print('\n准确率micro：', metrics.precision_score(y_val, ypreds, average='micro'), '  召回率micro：',
          metrics.recall_score(y_val, ypreds, average='micro'))
    print('F1_macro:', metrics.f1_score(y_val, ypreds, average='macro'))
    print('F1_micro:', metrics.f1_score(y_val, ypreds, average='micro'))
    model.save('arc127test.h5')

def out(x):
    x=tf.argmax(x)

    x=tf.to_float(x)
    x=tf.convert_to_tensor(x)
    return x

if __name__ == '__main__':
    embeddings_index=index_word_vectors()
    #texts,labels,labels_index=processing_text_dataset()
    train_label,x1text_train,x2text_train=processing_text_dataset('train')
    test_label,x1text_test,x2text_test=processing_text_dataset('test')
    word_index, x1_train,x2_train, y_train, x1_val,x2_val, y_val=to_tensor(x1text_train,x2text_train,train_label,x1text_test,x2text_test,test_label)
    num_words, embedding_matrix=prepare_embedding(word_index,embeddings_index)
    train(num_words, embedding_matrix,  x1_train,x2_train, y_train, x1_val,x2_val, y_val)



