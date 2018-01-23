from keras.models import load_model
from mytest.qa import index_word_vectors
from mytest.mytext import Tokenizer
import jieba
import numpy as np
from  keras.preprocessing.sequence import pad_sequences
text_data_dir='q_label.txt'

def processing_text_dataset():
    print('processing text dataset')
    reader = open(text_data_dir, 'r')
    unprocessed_data = np.array([example.split("#") for example in reader.readlines()])
    reader.close()
    #句子1
    raw_sentence=[]
    for (i, example) in enumerate(unprocessed_data):
        raw_sentence.append(example[0])
    print('found %s sentences.' % len(raw_sentence))
    return raw_sentence

def to_tensor(texts,tokenizer):
    ntexts=[' '.join(jieba.cut(texts))]
    sequences=tokenizer.texts_to_sequences(ntexts)
    word_index=tokenizer.word_index
    print('found %s unique tokens.'%len(word_index))
    #text11的张量
    data=pad_sequences(sequences,20)
    print('shape of the data tensor:',data.shape)
    return data

if __name__ == '__main__':
    tokenizer=Tokenizer(num_words=20000)
    ntexts=[]
    texts=processing_text_dataset()
    for x in texts:
        ntexts.append(' '.join(jieba.cut(x)))
    tokenizer.fit_on_texts(ntexts)
    model=load_model('qa.h5')
    #embeddings_index = index_word_vectors()

    while True:
        input_term = input("\nPlease input you question (EXIT to break): ")
        if input_term == 'EXIT':
            break
        else:
            x_val = to_tensor(input_term,tokenizer)
            yp = model.predict(x_val, batch_size=1, verbose=0)
            print('Our prediction is',yp,'\nIt is class is :',np.argmax(yp, axis=1))


