#from __future__ import print_function

import os,sys,jieba
import string

import numpy as np
# from keras.preprocessing.text import Tokenizer
from mytest.mytext import Tokenizer

from  keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense,Dropout,Input,Flatten,Lambda
from keras.layers import Conv1D,MaxPooling1D,Embedding
from keras.models import Model
from keras.layers.merge import concatenate
#from keras import backend as K
from sklearn import metrics
import tensorflow as tf
from keras.optimizers import SGD
from keras.utils import  plot_model

BASE_DIR=''
glove_dir='/home/wshong/Desktop/final vector/vectors.txt'
text_data_dir=BASE_DIR+'q_label.txt'
max_sequence_length=20
max_nb_words=20000
embedding_dim=300
#validation_split=0.2

def index_word_vectors():
    print('indexing word vectors.')

    embeddings_index={}
    f=open(glove_dir)
    for line in f:
        values=line.split()
        word=values[0]
        coefs=np.asarray(values[1:],dtype='float32')
        embeddings_index[word]=coefs
    f.close()

    print('found %s word vectors.' %len(embeddings_index))
    return embeddings_index

def processing_text_dataset():
    print('processing text dataset')
    reader = open(text_data_dir, 'r')
    unprocessed_data = np.array([example.split("#") for example in reader.readlines()])
    reader.close()
    #标签label值
    new_example = []
    #句子1
    raw_sentence=[]

    for (i, example) in enumerate(unprocessed_data):
        new_example.append(float(example[1]))
        raw_sentence.append(example[0])
    print('found %s sentences.' % len(raw_sentence))
    return new_example,raw_sentence


def to_tensor(texts,labels=[]):
    tokenizer=Tokenizer(num_words=max_nb_words)
    ntexts=[]
    for x in texts:
        ntexts.append(' '.join(jieba.cut(x)))
    tokenizer.fit_on_texts(ntexts)
    sequences=tokenizer.texts_to_sequences(ntexts)
    word_index=tokenizer.word_index
    print('found %s unique tokens.'%len(word_index))
    #text11的张量
    data=pad_sequences(sequences,max_sequence_length)
    labels=to_categorical(np.asarray(labels))
    print('shape of the data tensor:',data.shape)
    print('shape of label1 tensor:',labels.shape)
    return word_index, data,labels


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

def train(num_words,embedding_matrix,x_train,y_train):
    #weight加入glove预先训练好的向量，trainable设置成这部分参数不训练（glove预先训练好了，也可以添加自己的向量）
    embedding_layer=Embedding(num_words,embedding_dim,weights=[embedding_matrix],input_length=max_sequence_length,trainable=False)
    print('training model.')

    sequence_input=Input(shape=(max_sequence_length,),dtype='int32')
    embedded_sequences=embedding_layer(sequence_input)
    x1=Conv1D(300,3,activation='relu')(embedded_sequences)
    x1=MaxPooling1D(3)(x1)
    #x1=Dropout(0.4)(x1)
    x1=Conv1D(50,3,activation='relu')(x1)
    x1 = MaxPooling1D(4)(x1)
    x1=Flatten()(x1)
    x = Dense(128,activation='relu')(x1)
    #x = Dropout(0.25)(x)
    x = Dense(64, activation='relu')(x)
    #x = Dropout(0.25)(x)
    preds = Dense(3,activation='softmax')(x)
    model=Model(sequence_input,preds)

    model.summary()
    #model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['acc'])
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['acc'])

    model.fit(x_train,y_train,batch_size=64,epochs=50,validation_data=(x_train,y_train))
    plot_model(model,to_file='qa.png',show_shapes=True)

    yp = model.predict(x_train, batch_size=64, verbose=1)
    ypreds = np.argmax(yp, axis=1)
    y_train = np.argmax(y_train, axis=1)
    print('\n',metrics.classification_report(y_train, ypreds))
    print('\n正确率：', metrics.accuracy_score(y_train, ypreds))
    print('\n准确率macro：', metrics.precision_score(y_train, ypreds, average='macro'), '  召回率macro：',
          metrics.recall_score(y_train, ypreds, average='macro'))
    print('\n准确率micro：', metrics.precision_score(y_train, ypreds, average='micro'), '  召回率micro：',
          metrics.recall_score(y_train, ypreds, average='micro'))
    print('F1_macro:', metrics.f1_score(y_train, ypreds, average='macro'))
    print('F1_micro:', metrics.f1_score(y_train, ypreds, average='micro'))
    model.save('qa.h5')


if __name__ == '__main__':
    embeddings_index=index_word_vectors()
    #texts,labels,labels_index=processing_text_dataset()
    train_label,text_train=processing_text_dataset()
    word_index, x_train, y_train=to_tensor(text_train,train_label)
    num_words, embedding_matrix=prepare_embedding(word_index,embeddings_index)
    train(num_words, embedding_matrix,  x_train, y_train)



