# SentimentAnalysis-RNN
Sentimenta Analysis using Recurrent Neural networks in Keras with tensorflow as backend



### Algorithm for Sentimental Analysis using RNN 


#### 1) First we need to convert the raw text-words into so-called tokens which are integer values.

#### 2) Then we convert these integer-tokens into so-called embeddings which are real-valued vectors, whose mapping will be trained along with the neural network, so as to map words with similar meanings to similar embedding-vectors. 

#### 3) Then we input these embedding-vectors to a Recurrent Neural Network which can take sequences of arbitrary length as input and output a kind of summary of what it has seen in the input.

#### 4) Output from the RNN is squashed by an activation function (Sigmoid in this case)

#### 5) output is between 0 and 1 
 
#### { 0: highly negative, 1 : highly positive }



###Loading Data

```python
x_train_text, y_train = imdb.load_data(train=True) #loading train data
x_test_text, y_test = imdb.load_data(train=False) # loading test data
```


```python
print("Train-set size: ", len(x_train_text))
print("Test-set size:  ", len(x_test_text))
```

  
## Tokeninzing

```python
x_train_tokens = tokenizer.texts_to_sequences(x_train_text) # converting all the text in training data to tokens
```


# Padding

```python
x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens,
                            padding=pad, truncating=pad)
```

# Creating the model

```python
model = Sequential()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 544, 8)            80000     
    _________________________________________________________________
    gru_1 (GRU)                  (None, 544, 16)           1200      
    _________________________________________________________________
    gru_2 (GRU)                  (None, 544, 8)            600       
    _________________________________________________________________
    gru_3 (GRU)                  (None, 4)                 156       
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 5         
    =================================================================
    Total params: 81,961
    Trainable params: 81,961
    Non-trainable params: 0
    _________________________________________________________________



# fitting the data

```python
%%time
history=model.fit(x_train_pad, y_train,
          validation_split=0.05, epochs=3, batch_size=50)
```

    Train on 23750 samples, validate on 1250 samples
    Epoch 1/3
    23750/23750 [==============================] - 421s 18ms/step - loss: 0.4581 - acc: 0.7688 - val_loss: 0.3666 - val_acc: 0.8376
    Epoch 2/3
    23750/23750 [==============================] - 409s 17ms/step - loss: 0.2644 - acc: 0.8986 - val_loss: 0.2749 - val_acc: 0.8864
    Epoch 3/3
    23750/23750 [==============================] - 376s 16ms/step - loss: 0.2031 - acc: 0.9281 - val_loss: 0.1821 - val_acc: 0.9328
    CPU times: user 59min 13s, sys: 18min 56s, total: 1h 18min 10s
    Wall time: 20min 8s





# Checking on unknown Real Data 

#### to check this I took two reviews from IMDB 

1) Positive review for Peaky Blinders(TV Series)

2) Negative Review for  Race 3 (Indian Movie)

Positive Review URL : https://www.imdb.com/review/rw2878383/?ref_=tt_urv

Negative Review URL : https://www.imdb.com/title/tt7431594/reviews?ref_=tt_ql_3

        
        


```python
positive_review='''I was not expecting it to be this good,I really enjoyed all 4 episodes. 
The story is interesting,the acting is brilliant and the cinematography is just beautiful!
I am eagerly waiting for the next episodes.When I compare Peaky Blinders to other popular TV shows that use
sex,brutality and violence to shock the audiences and get high ratings(which they actually do)this sincere work is 
like needlework;fine,classy and detailed.The makers of this drama have not chosen the easy way,they have set off to
make a first class period drama,that dares to be different.Cillian Murphy is at his best,I will even go as far as
to say that this is one of the best performances I have seen of him.Sam Neil and Helen McCrory must be praised,all
casting is perfect.Peaky Blinders sets high standards for other television dramas to follow.'''

negative_review=''' I don't know what kind of mental conditions these people are suffering from, who are rating this movie
10/10. Why couldn't they just make it simple why this whole addition of crap. Just another crappy amalgamation of
the movies which had a better script. I just don't think Salman will make any sensible movies in which he just acts
good and doesn't just say mindless dialogues.'''

text=[positive_review,negative_review]
```


```python
tokens = tokenizer.texts_to_sequences(text) # we need to tokenize
```


```python
tokens_pad = pad_sequences(tokens, maxlen=max_tokens,
                           padding=pad, truncating='pre')
# padding
```


```python
tokens_pad.shape
```




    (2, 544)




```python
pos_review=model.predict(tokens_pad)[0]
neg_review=model.predict(tokens_pad)[1]
```


```python
print(pos_review)

```

>>Positive Review with a score of 97.52593040466309 %


    


```python
print(neg_review)
```
>>Negative Review with a score of 2.249847538769245 %



