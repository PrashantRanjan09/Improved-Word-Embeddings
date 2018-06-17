from nltk import pos_tag,word_tokenize
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,classification_report
from keras.layers import Embedding,Dense,Flatten,concatenate,Input
from keras.models import Model


class Embed:
    def __init__(self,vocab_size,embed_dim,pos_output_dim,max_len,pos_trainable_param):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pos_output_dim=pos_output_dim
        self.pos_input_dim = 20
        self.max_len = max_len
        self.char_to_int = {}
        self.int_to_char ={}
        self.pos_trainable_param = pos_trainable_param




    def embed_sentences(self,word_index,model,trainable_param,X_train_pad):

        embedding_matrix = np.zeros((self.vocab_size,self.embed_dim))
        for word, i in word_index.items():
            try:
                embedding_vector = model[word]
            except:
                pass
            try:
                if embedding_vector is not None:
                    embedding_matrix[i]=embedding_vector
            except:
                pass

        embed_layer = Embedding(self.vocab_size,self.embed_dim,weights =[embedding_matrix],trainable=trainable_param)

        input_seq = Input(shape=(X_train_pad.shape[1],))
        embed_seq = embed_layer(input_seq)
        return input_seq,embed_seq


    def tag_pos1(self,sentences):
        pos_tagged_sent = []
        pos_tagged_sent_all = []
        for sent in sentences:
            pos_tagged_sent.extend(pos_tag(sent))
            pos_tagged_sent_all.append(pos_tag(sent))
        tags = list(set([i[1] for i in pos_tagged_sent]))
        self.pos_input_dim = len(tags)
        self.char_to_int = dict((c, i) for i, c in enumerate(tags))
        self.int_to_char = dict((i, c) for i, c in enumerate(tags))

        X_pos_encoded =[]
        for i in range(len(pos_tagged_sent_all)):
            temp = [self.char_to_int[pos[1]] for pos in pos_tagged_sent_all[i]]
            X_pos_encoded.append(temp)

        return np.array(X_pos_encoded)


    def embed_pos(self,X_pos_arr):
        input_seq_pos = Input(shape=(X_pos_arr.shape[1],))
        embed_seq_pos = Embedding(self.pos_output_dim,self.pos_input_dim,input_length=self.max_len, dropout=0.2,trainable=self.pos_trainable_param)(input_seq_pos)

        return input_seq_pos,embed_seq_pos


'''
    def tag_pos(self,sentences,train_flag):

        if train_flag == True:
            pos_tagged_sent= []
            for sent in sentences:
                temp = pos_tag(sent)
                pos_tagged_sent.append(temp)

            X_pos=[]
            for i in range(len(pos_tagged_sent)):
                temp_p=[]
                for item_pair in pos_tagged_sent[i]:
                    _,p = item_pair
                    temp_p.append(p)
                X_pos.append(temp_p)

            tags=[]
            tags_sl =[]
            for j in range(len(pos_tagged_sent)):
                for i in range(len(pos_tagged_sent[j])):
                    _,temp = pos_tagged_sent[j][i]
                    tags_sl.append((temp))
                tags.append(tags_sl)

            all_tags = set(tags[0])
            self.pos_input_dim = len(set(tags[0]))
            self.char_to_int = dict((c, i) for i, c in enumerate(all_tags))
            self.int_to_char = dict((i, c) for i, c in enumerate(all_tags))

            X_pos_encoded =[]
            for i in range(len(X_pos)):
                temp = [self.char_to_int[pos] for pos in X_pos[i]]
                X_pos_encoded.append(temp)

            X_pos_arr = np.array(X_pos_encoded)
        else:
            pos_tagged_sent= []
            for sent in sentences:
                temp = pos_tag(sent)
                pos_tagged_sent.append(temp)


            X_pos=[]
            for i in range(len(pos_tagged_sent)):
                temp_p=[]
                for item_pair in pos_tagged_sent[i]:
                    _,p = item_pair
                    temp_p.append(p)
                X_pos.append(temp_p)

            X_pos_encoded =[]
            for i in range(len(X_pos)):
                temp = [self.char_to_int[pos] for pos in X_pos[i]]
                X_pos_encoded.append(temp)
            X_pos_arr = np.array(X_pos_encoded)

        return X_pos_arr
'''


def model_build(input_seq,input_seq_pos,embed_seq,embed_seq_pos,pad_train_x,X_pos_arr,train_y,
                epochs,batch_size,pad_test_x,X_pos_test_arr,test_y):
    print()
    x = concatenate([embed_seq, embed_seq_pos])
    x = Dense(256,activation ="relu")(x)
    x = Flatten()(x)
    preds = Dense(1,activation="sigmoid")(x)

    model = Model(inputs=[input_seq, input_seq_pos], outputs=preds)

    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
    model.fit([pad_train_x, X_pos_arr], train_y, epochs=epochs,batch_size=batch_size,
              validation_data=([pad_test_x, X_pos_test_arr], test_y))

    predictions = model.predict([pad_test_x, X_pos_test_arr])
    predictions = [0 if i<0.5 else 1 for i in predictions]
    print("Accuracy: ",accuracy_score(test_y,predictions))
    print("Classification Report: ",classification_report(test_y,predictions))

    return model
