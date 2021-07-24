# -*- coding: utf-8 -*-
"""

@author: Hossein
"""

import random
import time
import pandas as pd
import numpy as np
import re
import torch
from tqdm.auto import tqdm
tqdm.pandas(desc='Progress')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
import matplotlib.pyplot as plt


embed_size = 100 # how big is each word vector
max_features = 120000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 200 # max number of words in each sentence 
batch_size = 128 # how many samples to process at once
n_epochs = 8 # how many times to iterate over all samples
SEED = 10
debug = 0



#Choosing two books from Gutenberg Corpus
#book_id1,book_id2,book_id3 = input("Choose a number between 0-17 separated by space: ").split()
text1 = nltk.corpus.gutenberg.raw(nltk.corpus.gutenberg.fileids()[int(4)])
text2 = nltk.corpus.gutenberg.raw(nltk.corpus.gutenberg.fileids()[int(7)])
text3 = nltk.corpus.gutenberg.raw(nltk.corpus.gutenberg.fileids()[int(10)])
text4 = nltk.corpus.gutenberg.raw(nltk.corpus.gutenberg.fileids()[int(13)])
text5 = nltk.corpus.gutenberg.raw(nltk.corpus.gutenberg.fileids()[int(1)])
text6 = nltk.corpus.gutenberg.raw(nltk.corpus.gutenberg.fileids()[int(14)])

#print("\nYou chose the book :'{}' ".format(text1[text1.find('[')+1:text1.find(']')])
#      ,"and the book :'{}' ".format(text2[text2.find('[')+1:text2.find(']')])
#      ,"and the book :'{}' for the purpose of this assignment".format(text3[text3.find('[')+1:text3.find(']')]))
print("****************************************************************")


#finding the authors of the books
print("Finding the authors of the books for labeling.....")
Author1 = text1[text1.find(" by")+3:re.search(r"\d", text1).start()-1]
Author2 = text2[text2.find(" by")+3:re.search(r"\d", text2).start()-1]
Author3 = text3[text3.find(" by")+3:re.search(r"\d", text3).start()-1]
Author4 = text4[text4.find(" by")+3:re.search(r"\d", text4).start()-1]
Author5 = text5[text5.find(" by")+3:re.search(r"\d", text5).start()-1]
Author6 = text6[text6.find(" by")+3:re.search(r"\d", text6).start()-1]
print("The authors are'{}' and'{}' and'{}' respectively".format(Author1 , Author2,Author3))
print("****************************************************************")

#creating a dataframe adding 2000 partition of lenght 400 to the dataframe from each book at random
df_train = pd.DataFrame(data={"Text": ([None] * 18000), "Author":([None] * 18000)})

#these two dictionaries are used in the following for loops
dct = {'t_0':text1, 't_1': text2,'t_2': text3,'t_3': text4,'t_4': text5,'t_5': text6}
Authors = {'a_0': Author1,'a_1': Author2, 'a_2': Author3,'a_3': Author4,'a_4': Author5,'a_5': Author6}

#creating train dataframe
for i in [0,1,2,3,4,5]:
    temp_txt = dct['t_%s' %i]
    for j in range(0,3000):
        selected_doc = ""
        rn = random.sample(range(0, len(temp_txt)),1)
        for k in rn:
            selected_doc = temp_txt[k:k+400] + selected_doc
        df_train.at[(3000*i+j,"Text")] = selected_doc
        df_train.at[(3000*i+j,"Author")] = Authors['a_%s' %i]
print("Final dataframe ready for training a model")       
print(df_train) 



#df_train['len'] = df_train['Text'].apply(lambda s : len(s))

#shuffling the dataframe
df_train = df_train.sample(frac=1)

def clean_text(x):
    #removing non-alphabet characters except "'"
    text = re.sub(r"[^a-zA-z'\s]", '', x)
    #replacing single characters with a white space
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    # replacing multiple spaces with one space
    text = re.sub(r'\s+', ' ', text)
    return text
x = clean_text("ai3434,n%'t a   ")

#Replacing contractions with extended form
contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re
contractions, contractions_re = _get_contractions(contraction_dict)


def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)


# lower the text
df_train["Text"] = df_train["Text"].apply(lambda x: x.lower())

# Clean the text
df_train["Text"] = df_train["Text"].apply(lambda x: clean_text(x))

# Clean Contractions
df_train["Text"] = df_train["Text"].apply(lambda x: replace_contractions(x))


from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(df_train['Text'], df_train['Author'],
                                                    stratify=df_train['Author'], 
                                                    test_size=0.25)

print("Train shape : ",train_X.shape)
print("Test shape : ",test_X.shape)

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)
len(test_X[2])


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_y = le.fit_transform(train_y.values)
test_y = le.transform(test_y.values)




## FUNCTIONS TAKEN FROM https://www.kaggle.com/gmhost/gru-capsule

def load_glove(word_index):
    EMBEDDING_FILE = 'glove.6B.100d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:100]
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
    #len(embeddings_index['book'])
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.005838499,0.48782197
    embed_size = all_embs.shape[1]

    nb_words = min(max_features, len(word_index)+1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embeddings_index.get(word.capitalize())
            if embedding_vector is not None: 
                embedding_matrix[i] = embedding_vector
    return embedding_matrix


if debug:
    embedding_matrix = np.random.randn(120000,100)
else:
    embedding_matrix = load_glove(tokenizer.word_index) 
#embedding_matrix.shape

#***********************************************************    

import torch.nn as nn
import torch.nn.functional as F

class CNN_Text(nn.Module):
    
    def __init__(self):
        super(CNN_Text, self).__init__()
        filter_sizes = [1,2,3]
        num_filters = 30
        n_classes = len(le.classes_)
        self.embedding = nn.Embedding(embedding_matrix.shape[0], embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.convs1 = nn.ModuleList([nn.Conv1d(1, num_filters, (K, embed_size)) for K in filter_sizes])
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(filter_sizes)*num_filters, n_classes)


    def forward(self, x):
        x = self.embedding(x)  
        x = x.unsqueeze(1)  
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  
        x = torch.cat(x, 1)
        x = self.dropout(x)  
        logit = self.fc1(x) 
        return logit



class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.hidden_size = 128
        drp = 0.1
        n_classes = len(le.classes_)
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size*4 , 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(128, n_classes)


    def forward(self, x):
        #rint(x.size())
        h_embedding = self.embedding(x)
        #_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        conc = torch.cat(( avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out
    

#n_epochs = 1

model_type = input("Choose the model:\n 1: CNN \n 2: LSTM\n type 1 or 2: ")
if model_type == "1":
    model = CNN_Text()
else:
    model = BiLSTM()
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
summary(model,(1,1,100))
'''
loss_fn = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)


# Load train and test 
x_train = torch.tensor(train_X, dtype=torch.long)
y_train = torch.tensor(train_y, dtype=torch.long)
x_cv = torch.tensor(test_X, dtype=torch.long)
y_cv = torch.tensor(test_y, dtype=torch.long)

# Create Torch datasets
train = torch.utils.data.TensorDataset(x_train, y_train)
valid = torch.utils.data.TensorDataset(x_cv, y_cv)

# Create Data Loaders
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

train_loss = []
valid_loss = []

for epoch in range(n_epochs):
    start_time = time.time()
    # Set model to train configuration
    model.train()
    avg_loss = 0.  
    for i, (x_batch, y_batch) in enumerate(train_loader):
        # Predict/Forward Pass
        y_pred = model(x_batch)
        # Compute loss
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)
    
    # Set model to validation configuration -Doesn't get trained here
    model.eval()        
    avg_val_loss = 0.
    val_preds = np.zeros((len(x_cv),len(le.classes_)))
    
    for i, (x_batch, y_batch) in enumerate(valid_loader):
        y_pred = model(x_batch).detach()
        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
        # keep/store predictions
        val_preds[i * batch_size:(i+1) * batch_size] =F.softmax(y_pred).cpu().numpy()
    
    # Check Accuracy
    val_accuracy = sum(val_preds.argmax(axis=1)==test_y)/len(test_y)
    train_loss.append(avg_loss)
    valid_loss.append(avg_val_loss)
    elapsed_time = time.time() - start_time 
    print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f}  \t val_acc={:.4f}  \t time={:.2f}s'.format(
                epoch + 1, n_epochs, avg_loss, avg_val_loss, val_accuracy, elapsed_time))
    



def plot_graph(epochs):
    #fig = plt.figure(figsize=(12,12))
    plt.title("Train/Validation Loss")
    plt.plot(list(np.arange(epochs) + 1) , train_loss, label='train')
    plt.plot(list(np.arange(epochs) + 1), valid_loss, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')
    
plot_graph(n_epochs)
plt.show()

#Confusion Matrix
import scikitplot as skplt
y_true = [le.classes_[x] for x in test_y]
y_pred = [le.classes_[x] for x in val_preds.argmax(axis=1)]
skplt.metrics.plot_confusion_matrix(
    y_true, 
    y_pred,
    figsize=(12,12),x_tick_rotation=90)
plt.show()

def predict_single(x):    
    # lower the text
    x = x.lower()
    # Clean the text
    x =  clean_text(x)
    # Clean numbers
    x = replace_contractions(x)
    # tokenize
    x = tokenizer.texts_to_sequences([x])
    # pad
    x = pad_sequences(x, maxlen=maxlen)
    # create dataset
    x = torch.tensor(x, dtype=torch.long)

    pred = model(x).detach()
    pred = F.softmax(pred).cpu().numpy()

    pred = pred.argmax(axis=1)

    pred = le.classes_[pred]
    return pred[0]

user_text = input("Enter a text: ")
print("The author is: ", predict_single(user_text))

