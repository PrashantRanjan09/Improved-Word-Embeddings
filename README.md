# Improved-Word-Embeddings
This is a paper implementation of https://arxiv.org/pdf/1711.08609.pdf. This paper increases the accuracy of word embeddings by enriching the word embedding with its associated POS (Part of Speech) tag. (See Below):

![Optional Text](../master/ImprovedWordEmbeddings.png)


However, this excludes the Lexicon2Vec conversion and follows the following steps:

       Inputs:
       S = {W1, W2,……., Wn} , Input sentence S contains n words
       PT = {T1, T2,……, Tm}, All POS tags
       Corpus = Imdb/Your corpus
       
      Output:
       IMV: Improved word vectors of corpus
 
 The steps follow the following algorithm:
       
    for j=1 to m do
        VTj GenerateVector ( Tj )
        Tj < Tj , VTj >
    end for

    for each Wi in S do
      If Wi exist in "your choice of word vector" then extract VecWi
        MVi VecWi
      endif
      
      # POS ExtractPOS ( Wi )
    for k=1 to m do
     If POS=Tk then ADD VTk into MVi
     end if
    end for
    
#### Usage
This gives you the option of testing it on the Imdb data and also to run it on your corpus.
To run it on the imdb data:

    In config.json assign use_imdb : 1
    
To run it on your corpus:

      In config.json assign use_imdb : 0
      
The model used is a very generic one. Feel free to make changes to the model as per your requirements.
To change model:

     In pos_function.py change model_build
