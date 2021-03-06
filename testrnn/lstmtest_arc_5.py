#from __future__ import print_function

import os,sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from  keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense,Dropout,Input,Flatten,Lambda,Reshape
from keras.layers import Conv1D,Conv2D,MaxPooling2D,Embedding,LSTM
from keras.models import Model
from keras.layers.merge import concatenate
from keras import backend as K
from sklearn import metrics
import tensorflow as tf
from keras.optimizers import SGD
from keras.utils import  plot_model
from keras import regularizers
from keras.callbacks import EarlyStopping

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

    #生成交互序列，window=3
    # sequences1=to_interact(sequences11,sequences12)
    # sequences2=to_interact(sequences21,sequences22)

    word_index=tokenizer.word_index
    print('found %s unique tokens.'%len(word_index))
    #text11的张量
    data11 = pad_sequences(sequences11,max_sequence_length)
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
    #print('shape of label1 tensor:',labels21.shape)
    print('shape of the data21 tensor:', data21.shape)
    print('shape of the data22 tensor:', data22.shape)
    print('shape of label2 tensor:', labels2.shape)
    return word_index, data11,data12,labels1,data21,data22,labels2

#生成交互序列，window=3
def tensor_to_interact(sequences):
    #interact_matric=[]
    seg=[]
    for i in range(max_sequence_length-2):
        seg1 = sequences[:,i:i + 3,:]
        for j in range(max_sequence_length,max_sequence_length-2+max_sequence_length):
            seg2=sequences[:,j:j+3,:]
            # seg.append(seg1)
            # seg.append(seg2)
            #seg = np.concatenate([seg, seg1],axis=1)
            #seg_temp = np.concatenate((seg1, seg2))
            seg_temp=K.concatenate([seg1, seg2],axis=1)
            if(i==0 and j==max_sequence_length):
                seg=seg_temp
            else:
                #seg = np.concatenate((seg, seg_temp))
                seg=K.concatenate([seg, seg_temp], axis=1)

    #seg = [seg]
    #interact_matric.append(seg)
    return seg

def prepare_embedding(word_index,embeddings_index):
    print('preparing embedding matrix.')
    num_words=len(word_index)
    embedding_matrix=np.zeros((num_words+1,embedding_dim))
    for word,i in word_index.items():
        if(i>=len(word_index)):
            continue
        embedding_vector=embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector
    return num_words+1 , embedding_matrix

def train(num_words, embedding_matrix,  x1_train,x2_train, y_train, x1_val,x2_val, y_val):
    #weight加入glove预先训练好的向量，trainable设置成这部分参数不训练（glove预先训练好了，也可以添加自己的向量）
    embedding_layer=Embedding(num_words,embedding_dim,weights=[embedding_matrix],input_length=max_sequence_length,trainable=False)
    print('training model.')

    sequence_input=Input(shape=(max_sequence_length,),dtype='int32')

    embedded_sequences=embedding_layer(sequence_input)
    x=LSTM(embedding_dim,return_sequences=True)(embedded_sequences)
    output=Reshape((35,50,))(x)
    #bilstm = Bidirectional(LSTM(self.params['lstmOutDim'], return_sequences=True, dropout=0.2, recurrent_dropout=0.2), name='BiLSTM')(merge_layer)
    Embedding_model=Model(sequence_input,output)
    sequence_input1 = Input(shape=(max_sequence_length,), dtype='int32')
    sequence_input2 = Input(shape=(max_sequence_length,), dtype='int32')
    x1=Embedding_model(sequence_input1)
    x2=Embedding_model(sequence_input2)
    x3 = concatenate([x1, x2],axis=1)
    #feture_map=tensor_to_interact(x1,x2)
    feture_map=Lambda(tensor_to_interact)(x3)
    x=Conv1D(200,6,strides=6,activation='relu')(feture_map)
    x=Reshape((33,33,200))(x)
    x=MaxPooling2D((3,3),strides=3)(x)
    #x1=Dropout(0.5)(x1)
    x=Conv2D(200,(2,2),activation='relu')(x)
    x = MaxPooling2D((2,2),strides=2)(x)
    x = Conv2D(200, (4, 4),strides=1, activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=2)(x)
    #x1 = Dropout(0.5)(x1)
    # x1 = Conv1D(200, 3, activation='relu')(x1)
    # x1 = MaxPooling1D(3)(x1)
    # x1 = Dropout(0.25)(x1)
    x=Flatten()(x)

    # shall_model=Model(sequence_input1,output)
    #
    # output1=shall_model(sequence_input1)
    # output2=shall_model(sequence_input2)
    #
    # x = concatenate([output1,output2])
    x = Dense(128,activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dropout(0.25)(x)
    preds = Dense(1,activation='sigmoid')(x)
    #preds=Lambda(out)(preds)
    #preds=Lambda(lambda i:1.0 if(i[0]>=0.5) else 0.0)(preds)
    #preds = Lambda(K.argmax,output_shape=(1,),dtype='float64')(preds)
    model=Model([sequence_input1,sequence_input2],preds)

    model.summary()
    #model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['acc'])
    sgd = SGD(lr=0.01, decay=1e-6)
    model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['acc'])

    model.fit([x1_train,x2_train],y_train,batch_size=100,epochs=100,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=1)],validation_data=([x1_val,x2_val],y_val))
    # plot_model(shall_model, to_file='model_1.png', show_shapes=True)
    plot_model(model,to_file='ARC_5_2.png',show_shapes=True)

    yp = model.predict([x1_val,x2_val], batch_size=100, verbose=1)
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
    model.save('arc5_2.h5')


if __name__ == '__main__':
    embeddings_index = index_word_vectors()
    # texts,labels,labels_index=processing_text_dataset()
    train_label, x1text_train, x2text_train = processing_text_dataset('train')
    test_label, x1text_test, x2text_test = processing_text_dataset('test')
    word_index, x1_train, x2_train, y_train, x1_val, x2_val, y_val = to_tensor(x1text_train, x2text_train, train_label,
                                                                               x1text_test, x2text_test, test_label)
    num_words, embedding_matrix = prepare_embedding(word_index, embeddings_index)
    train(num_words, embedding_matrix, x1_train, x2_train, y_train, x1_val, x2_val, y_val)



