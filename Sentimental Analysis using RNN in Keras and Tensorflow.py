
# coding: utf-8

# # Algorithm for Sentimental Analysis using RNN 
# 
# ### 1) First we need to convert the raw text-words into so-called tokens which are integer values.
# 
# ### 2) Then we convert these integer-tokens into so-called embeddings which are real-valued vectors, whose mapping will be trained along with the neural network, so as to map words with similar meanings to similar embedding-vectors. 
# 
# ### 3) Then we input these embedding-vectors to a Recurrent Neural Network which can take sequences of arbitrary length as input and output a kind of summary of what it has seen in the input.
# 
# ### 4) Output from the RNN is squashed by an activation function (Sigmoid in this case)
# 
# ### 5) output is between 0 and 1 
#  
# ### { 0: highly negative, 1 : highly positive }

# In[1]:


#importing required Libraries

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from scipy.spatial.distance import cdist
# from tf.keras.models import Sequential  # This does not work!
from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[3]:


# Some of the code and explaination here is taken from https://github.com/Hvass-Labs/ :)


# In[4]:


import imdb # this is helper package to download and load the imdb dataset by https://github.com/Hvass-Labs/


# In[5]:


imdb.maybe_download_and_extract() #Downloading and Extracting the dataset


# In[6]:


x_train_text, y_train = imdb.load_data(train=True) #loading train data
x_test_text, y_test = imdb.load_data(train=False) # loading test data


# In[7]:


print("Train-set size: ", len(x_train_text))
print("Test-set size:  ", len(x_test_text))


# In[8]:


data_text = x_train_text + x_test_text


# In[9]:


x_train_text[100] # looking at an example text 


# In[10]:


y_train[100]


# In[11]:


num_words = 10000


# In[12]:


tokenizer = Tokenizer(num_words=num_words)


# In[13]:


get_ipython().run_cell_magic('time', '', 'tokenizer.fit_on_texts(data_text)')


# In[14]:


tokenizer.word_index
#. This is ordered by the number of occurrences of the words in the data-set.
#These integer-numbers are called word indices or "tokens" because they uniquely 
#identify each word in the vocabulary.


# In[15]:


x_train_tokens = tokenizer.texts_to_sequences(x_train_text) # converting all the text in training data to tokens


# In[16]:


x_train_text[1] # actual text without tokens


# In[17]:


np.array(x_train_tokens[1]) # text after tokenizing


# In[18]:


x_test_tokens = tokenizer.texts_to_sequences(x_test_text) # converting text data into tokens


# # Padding
# 
# The Recurrent Neural Network can take sequences of arbitrary length as input, but in order to use a whole batch of data, the sequences need to have the same length. 
# But we can't take the length of longest review and pad that many zeros to the shorter reviews because it will take lot of memory so we have to figure out a particular length that will be sufficent for most of our data

# In[19]:


num_tokens = np.array([len(tokens) for tokens in x_train_tokens + x_test_tokens])
#making a list to store the lengths of tokenized reviews in both training and test data set


# In[20]:


np.mean(num_tokens) #calculating average length 


# In[21]:


np.max(num_tokens) # maximum length of any tokenized review


# In[22]:


np.min(num_tokens)# minimum


# ## Visualizing Token Lengths

# In[23]:


import plotly.plotly as py
import plotly.graph_objs as go

trace0 = go.Box(
    y=num_tokens
)
data=[trace0]
py.iplot(data)


# ### We can see in the above box plot that most of the token lengths are between 0 and 500 but also we have some outliers which go upto 2000+ 

# In[24]:


max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
max_tokens


# In[ ]:


#The max number of tokens we will allow is set to the average plus 2 standard deviations.


# We have already seen that most of our data is between 0 and 500 length but we can verify it 

# In[25]:


str(np.sum(num_tokens < max_tokens) / len(num_tokens) * 100) +' %' 


# ## When we pad data we need to decide where to pad the data,wether pad in the beginning or in the end
# 
# #### - If we pad at the end then there might be a chance that RNN might get confused seeing lot of zeroes after processing some data 
# 
# #### - So we need to pad in beginning 

# In[26]:


pad = 'pre'


# In[27]:


x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens,
                            padding=pad, truncating=pad)


# In[28]:


x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)


# In[29]:


x_train_pad.shape


# In[30]:


x_test_pad.shape


# In[31]:


np.array(x_train_tokens[1]) # before padding


# In[32]:


np.array(x_train_pad[1]) # After Padding


# In[33]:


num_tokens_pad = np.array([len(tokens) for tokens in x_train_pad + x_test_pad])


# In[34]:


import plotly.plotly as py
import plotly.graph_objs as go

trace0 = go.Box(
    y=num_tokens_pad
)
data=[trace0]
py.iplot(data)


# In[ ]:


## We can see in the above plot that now all the data have same length (544)


# ## Alternatives to padding
# 
# ### There are various options if you don't want to do padding:
# 
# #### 1) Make your batch size equal to 1 i.e feed data one by one into RNN 
# 
# #### 2) Grouping sequences of same lengths i.e all the sequences of particular lengths like 100 or 500 together
# 
# 

# ## Creating the Recurrent Neural Network using Keras 
# 
# 

# In[35]:


model = Sequential()


# The first layer in the RNN is a so-called Embedding-layer which converts each integer-token into a vector of values
# Tokenized data is huge from 0 to vocaulary length (10000 in this case) and value of tokens(integer values) does not make any sense so tokens are converted into embedded vector  which is a vector that maps words with similar semantic meanings and of length usually 100-300
# 
# Watch detailed explaination of Embeddings by Andrew Ng :https://www.youtube.com/watch?v=DDByc9LyMV8 

# In[36]:


embedding_size = 8


# In[37]:


model.add(Embedding(input_dim=num_words,
                   output_dim=embedding_size,
                   input_length=max_tokens,
                   ))


# In[38]:


model.add(GRU(units=16, return_sequences=True))#layer below will be processing sequences so thats why return_sequences=True


# In[39]:


model.add(GRU(units=8, return_sequences=True))#layer below will be processing sequences so thats why return_sequences=True


# In[40]:


model.add(GRU(units=4))# now we don't need sequences because in the next layer we will predict the output


# In[41]:


model.add(Dense(1, activation='sigmoid'))# fully connected layer with output =1 since we will predict either positve or negative


# In[42]:


optimizer = Adam(lr=1e-3)


# In[43]:


model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


# In[44]:


model.summary()


# In[45]:


get_ipython().run_cell_magic('time', '', 'history=model.fit(x_train_pad, y_train,\n          validation_split=0.05, epochs=3, batch_size=50)')


# In[46]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)


trace0 = go.Scatter(
    x = list(epochs),
    y = list(acc),
    mode = 'lines',
    name = 'training accuracy'
)

trace1= go.Scatter(
    x = list(epochs),
    y = list(val_acc),
    mode = 'lines',
    name = 'Validation accuracy'
)

layout = go.Layout( title='Training and Validation Accuracy')


data = [trace0,trace1]
fig = go.Figure(data=data, layout=layout)


py.iplot(fig)


# In[47]:


trace0 = go.Scatter(
    x = list(epochs),
    y = list(loss),
    mode = 'lines',
    name = 'training loss'
)

trace1= go.Scatter(
    x = list(epochs),
    y = list(val_loss),
    mode = 'lines',
    name = 'Validation loss'
)

layout = go.Layout( title='Training and Validation loss')


data = [trace0,trace1]
fig = go.Figure(data=data, layout=layout)


py.iplot(fig)


# In[48]:


get_ipython().run_cell_magic('time', '', 'result = model.evaluate(x_test_pad, y_test)')


# In[49]:


print("Accuracy: {0:.2%}".format(result[1]))


# # Checking on unknown Real Data 

# #### to check this I took two reviews from IMDB 
# 
# 1) Positive review for Peaky Blinders(TV Series)
# 
# 2) Negative Review for  Race 3 (Indian Movie)
# 
# Positive Review URL : https://www.imdb.com/review/rw2878383/?ref_=tt_urv
# 
# Negative Review URL : https://www.imdb.com/title/tt7431594/reviews?ref_=tt_ql_3
# 
#         
#         

# In[140]:


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


# In[141]:


tokens = tokenizer.texts_to_sequences(text) # we need to tokenize


# In[142]:


tokens_pad = pad_sequences(tokens, maxlen=max_tokens,
                           padding=pad, truncating='pre')
# padding


# In[143]:


tokens_pad.shape


# In[144]:


a=model.predict(tokens_pad)[0]
b=model.predict(tokens_pad)[1]


# In[145]:


if a > 0.60: # I am thresholding it at 60%
    print('Positive Review with a score of {} %'.format(a[0]*100))
else:
    print('Negative Review ')


# In[146]:


if b > 0.50: # I am thresholding it at 50%
    print('Positive Review ')
else:
    print('Negative Review with a score of {} %'.format(b[0]*100))


# # We can see that it is classifying pretty good 
