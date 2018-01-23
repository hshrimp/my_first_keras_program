from __future__ import print_function

import os,sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from  keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical,plot_model
from keras.layers import Dense,Dropout,Input,Flatten
from keras.layers import Conv1D,MaxPooling1D,Embedding
from keras.models import Model

BASE_DIR=''
glove_dir=BASE_DIR+'glove.6B/'
text_data_dir=BASE_DIR+'20_newsgroup/'
max_sequence_length=1000
max_nb_words=20000
embedding_dim=100
validation_split=0.2

def index_word_vectors():
    print('indexing word vectors.')

    embeddings_index={}
    f=open(os.path.join(glove_dir,'glove.6B.100d.txt'))
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

    texts=[]
    labels_index={}
    labels=[]
    for name in sorted(os.listdir(text_data_dir)):
        path = os.path.join(text_data_dir,name)
        if os.path.isdir(path):
            label_id=len(labels_index)
            labels_index[name]=label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath=os.path.join(path,fname)
                    if sys.version_info<(3,):
                        f=open(fpath)
                    else:
                        f=open(fpath,encoding='latin-1')
                    t=f.read()
                    i=t.find('\n\n')
                    if 0<i:
                        t=t[i:]
                    texts.append(t)
                    f.close()
                    labels.append(label_id)

    print('found %s texts.' %len(texts))
    return texts,labels,labels_index


def to_tensor(texts,labels):
    tokenizer=Tokenizer(num_words=max_nb_words)
    tokenizer.fit_on_texts(texts)
    sequences=tokenizer.texts_to_sequences(texts)

    word_index=tokenizer.word_index
    print('found %s unique tokens.'%len(word_index))

    data=pad_sequences(sequences,max_sequence_length)

    labels=to_categorical(np.asarray(labels))
    print('shape of the data tensor:',data.shape)
    print('shape of label tensor:',labels.shape)

    #split thr data into a training set and a validation set,打乱数据，切分数据，训练集80%，验证集20%
    indices=np.arange(data.shape[0])
    np.random.shuffle(indices)
    data=data[indices]
    labels=labels[indices]
    num_validation_samples=int(validation_split*data.shape[0])

    x_train=data[:-num_validation_samples]
    y_train=labels[:-num_validation_samples]
    x_val=data[-num_validation_samples:]
    y_val=labels[-num_validation_samples:]
    return word_index, x_train, y_train, x_val, y_val


def prepare_embedding(word_index,embeddings_index):
    print('preparing embedding matrix.')
    num_words=min(max_nb_words,len(word_index))
    embedding_matrix=np.zeros((num_words,embedding_dim))
    for word,i in word_index.items():
        if i>=max_nb_words:
            continue
        embedding_vector=embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector
    return num_words,embedding_matrix

def train(num_words,embedding_matrix,labels_index,x_train,y_train,x_val,y_val):
    #weight加入glove预先训练好的向量，trainable设置成这部分参数不训练（glove预先训练好了，也可以添加自己的向量）
    embedding_layer=Embedding(num_words,embedding_dim,weights=[embedding_matrix],input_length=max_sequence_length,trainable=False)
    print('training model.')

    sequence_input=Input(shape=(max_sequence_length,),dtype='int32')
    embedded_sequences=embedding_layer(sequence_input)
    x=Conv1D(256,5,activation='relu')(embedded_sequences)
    x=MaxPooling1D(5)(x)
    x=Dropout(0.4)(x)
    x=Conv1D(256,5,activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Dropout(0.4)(x)
    x = Conv1D(256, 5, activation='relu')(x)
    x = MaxPooling1D(35)(x)
    x = Dropout(0.25)(x)
    x=Flatten()(x)
    x=Dense(128,activation='relu')(x)
    preds=Dense(len(labels_index),activation='softmax')(x)

    model=Model(sequence_input,preds)

    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])
    model.fit(x_train,y_train,batch_size=64,epochs=10,validation_data=(x_val,y_val))
    plot_model(model, to_file='class1.png', show_shapes=True)

    model.save('class1.h5')



if __name__ == '__main__':
    embeddings_index=index_word_vectors()
    texts,labels,labels_index=processing_text_dataset()
    word_index, x_train, y_train, x_val, y_val=to_tensor(texts,labels)
    num_words, embedding_matrix=prepare_embedding(word_index,embeddings_index)
    train(num_words, embedding_matrix, labels_index, x_train, y_train, x_val, y_val)



