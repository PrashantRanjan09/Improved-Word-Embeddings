import json
import numpy as np
from sklearn.model_selection import train_test_split
from pos_function import Embed
from pos_function import model_build
from word_embeddings import load_data,prepare_data_for_word_vectors,building_word_vector_model,classification_model,padding_input,prepare_data_for_word_vectors_imdb


def json_to_dict(json_set):
    for k,v in json_set.items():
        if v == "True":
            json_set[k]= True
        elif v == "False":
            json_set[k]=False
        else:
            json_set[k]=v
    return json_set

with open("config.json","r") as f:
    params_set = json.load(f)
params_set = json_to_dict(params_set)


with open("model_params.json", "r") as f:
    model_params = json.load(f)
model_params = json_to_dict(model_params)

'''
    load_data function works on imdb data. In order to load your data, comment line 27 and pass your data in the form of X,y
    X = text data column
    y = label column(0,1 etc)

'''
# for imdb data

if params_set["use_imdb"]==1:
    print("loading imdb data")
    x_train,x_test,y_train,y_test = load_data(params_set["vocab_size"],params_set["max_len"])
    X = np.concatenate([x_train,x_test])
    y = np.concatenate([y_train,y_test])
    sentences_as_words,word_ix = prepare_data_for_word_vectors_imdb(X)
    print(sentences_as_words[0])
    model_wv = building_word_vector_model(params_set["option"],sentences_as_words,params_set["embed_dim"],
                                       params_set["workers"],params_set["window"],y)
    print("word vector model built")
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=params_set["split_ratio"], random_state=42)

    x_train_pad,x_test_pad = padding_input(x_train,x_test,params_set["max_len"])
    print("padded")
else:
# for other data:
# put your data in the form of X,y

    X = ["this is a sentence","this is another sentence by me","yet another sentence for training","one more again"]
    y=np.array([0,1,1,0])

    sentences_as_words,sentences,word_ix = prepare_data_for_word_vectors(X)
    print(sentences_as_words[0])
    print("sentences loaded")
    model_wv = building_word_vector_model(params_set["option"],sentences,params_set["embed_dim"],
                                       params_set["workers"],params_set["window"],y)
    print("word vectors model built")
    x_train, x_test, y_train, y_test = train_test_split(sentences, y, test_size=params_set["split_ratio"], random_state=42)
    print("splitting done")
    x_train_pad,x_test_pad = padding_input(x_train,x_test,params_set["max_len"])
    print("padded")


if params_set["use_imdb"]==1:
    print("")
    embed = Embed(params_set["vocab_size"],params_set["embed_dim"],params_set["pos_embed_dim"],params_set["max_len"],True)
    print("embed class")
    inp_seq,sent_emb = embed.embed_sentences(word_ix,model_wv,False,x_train_pad)
    print("sentence embedding done")
    pos_enc = embed.tag_pos1(sentences_as_words)
    print("POS encoded")
    x_train_pos, x_test_pos, _, _ = train_test_split(pos_enc, y, test_size=params_set["split_ratio"], random_state=42)
    x_train_pos_pad,x_test_pos_pad = padding_input(x_train_pos,x_test_pos,params_set["max_len"])
    print("POS padded")
    inp_pos,pos_embed = embed.embed_pos(x_train_pos_pad)
    print("building model")
    model = model_build(inp_seq,inp_pos,sent_emb,pos_embed,x_train_pad,x_train_pos_pad,y_train,model_params["epochs"],model_params["batch_size"],x_test_pad,x_test_pos_pad,y_test)
    print("model built")
    print(model.summary())

else :

    embed = Embed(params_set["vocab_size"],params_set["embed_dim"],params_set["pos_embed_dim"],params_set["max_len"],True)
    inp_seq,sent_emb = embed.embed_sentences(word_ix,model_wv,False,x_train_pad)

    pos_enc = embed.tag_pos1(sentences_as_words)
    print("POS encoded")
    x_train_pos, x_test_pos, _, _ = train_test_split(pos_enc, y, test_size=params_set["split_ratio"], random_state=42)
    x_train_pos_pad,x_test_pos_pad = padding_input(x_train_pos,x_test_pos,params_set["max_len"])
    print("POS padded")
    inp_pos,pos_embed = embed.embed_pos(x_train_pos_pad)
    print("building model")
    model = model_build(inp_seq,inp_pos,sent_emb,pos_embed,x_train_pad,x_train_pos_pad,y_train,model_params["epochs"],model_params["batch_size"],x_test_pad,x_test_pos_pad,y_test)
    print("model built")
    print(model.summary())
