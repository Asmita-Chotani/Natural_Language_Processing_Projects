#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import torch.nn as nn
import torch.optim as optim
import operator
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
import numpy as np


# In[5]:


from google.colab import drive
drive.mount('/content/drive')


# ### Train Data

# In[6]:


# reading training data
training_data=[]
with open('/content/drive/MyDrive/Colab Notebooks/data/train', 'r') as readFile:
        for inputs in readFile:
            inputs = inputs.rstrip()
            training_data.append(inputs.split(' '))

# create list of train tagged words
train_list=[]
for x in training_data:
    if len(x)>2:
        train_list.append([x[1],x[2]])
        


# ### Find number of sentence

# In[8]:


ct=0
for x in training_data:
    if len(x)>1:
        if x[0]=='1':
            ct=ct+1
ct


# In[9]:


train_sentence_list=[]
st=[]
flag=0
for x in training_data:
    if len(x)>1:
            if x[0]=='1':
                flag=flag+1
                if flag == 1:
                    st=[]
                    st.append((x[1],x[2]))
                elif flag==14987:
                    print(flag)
                    train_sentence_list.append(st)
                    st=[]
                    st.append((x[1],x[2]))  
                    train_sentence_list.append(st)
                    break
                else: 
                    train_sentence_list.append(st)
                    st=[]
                    st.append((x[1],x[2]))  
            else:
                st.append((x[1],x[2]))


# In[10]:


len(train_sentence_list)


# In[11]:


train_sentence_list[-2:]


# ### Dev Data

# In[12]:


# reading dev data
dev_data=[]
with open('/content/drive/MyDrive/Colab Notebooks/data/dev', 'r') as readFile:
        for inputs in readFile:
            inputs = inputs.rstrip()
            dev_data.append(inputs.split(' '))

dev_list=[]
for x in dev_data:
    if len(x)>1:
        dev_list.append((x[1],x[2]))


# ### Find number of sentence

# In[13]:


dev_ct=0
for x in dev_data:
    if len(x)>1:
        if x[0]=='1':
            dev_ct=dev_ct+1
dev_ct


# In[14]:


dev_sentence_list=[]
dev_st=[]
dev_flag=0
for x in dev_data:
    if len(x)>1:
            if x[0]=='1':
                dev_flag=dev_flag+1
                if dev_flag == 1:
                    dev_st=[]
                    dev_st.append((x[1],x[2]))
                elif dev_flag==3466:
                    print(dev_flag)
                    dev_sentence_list.append(dev_st)
                    dev_st=[]
                    dev_st.append((x[1],x[2]))  
                    dev_sentence_list.append(dev_st)
                    break
                else: 
                    dev_sentence_list.append(dev_st)
                    dev_st=[]
                    dev_st.append((x[1],x[2]))  
            else:
                dev_st.append((x[1],x[2]))


# In[15]:


len(dev_sentence_list)


# ### Vocab creation

# In[16]:


vocab_dict={}
count=0
for x in training_data:
    if len(x)>1:
        if x[1] in vocab_dict:
            temp=vocab_dict[x[1]]
            vocab_dict[x[1]]=temp+1
        else:
            vocab_dict[x[1]]=1

unk_c=0
unk_list=[]
for key in vocab_dict:
    if vocab_dict[key]<=2:
        unk_c=unk_c+vocab_dict[key]
        unk_list.append(key)
    else:
        continue


# In[17]:


sorted_d1 = sorted(vocab_dict.items(), key=operator.itemgetter(1),reverse=True)

vocab_txt_file={}
ind2=1
vocab_txt_file[ind2]=('<UNK>',unk_c)

for i in sorted_d1:
    ind2=ind2+1
    if i[0] not in unk_list:
      x= i[0]
      y= i[1]
      vocab_txt_file[ind2]= (x,y)
    else:
      continue


# ### Word2Idx Mapping

# In[18]:


vocab_txt_file2={}
ind2=1
for i in sorted_d1:
    ind2=ind2+1
    x= i[0]
    y= i[1]
    vocab_txt_file2[ind2]= (x,y)


# In[ ]:


# create a mapping from words to integers
word2idx= {word[0] : idx for idx, word in vocab_txt_file2.items()}
word2idx['<PAD>']= 0
word2idx['<UNK>']= 1
word2idx


# In[21]:


len(word2idx)


# ### Tag2Idx Mapping

# In[22]:


tag_dict={}
count=0
for x in training_data:
    if len(x)>1:
        if x[2] in tag_dict:
            temp=tag_dict[x[2]]
            tag_dict[x[2]]=temp+1
        else:
            tag_dict[x[2]]=1


# In[23]:


# create a mapping from tags to integers
tag2idx ={}
tag_ind=0
tag2idx['<PAD>']= tag_ind
for key, value in tag_dict.items():
        tag_ind=tag_ind+1
        tag2idx[key]=tag_ind

    


# In[24]:


tag2idx


# ### Test  data

# In[25]:


idx2tag={}
for key, value in tag2idx.items():
    idx2tag[value]=key


# In[26]:


# reading test data
test_data=[]
with open('/content/drive/MyDrive/Colab Notebooks/data/test', 'r') as readFile:
        for inputs in readFile:
            inputs = inputs.rstrip()
            test_data.append(inputs.split(' '))

test_list=[]
for x in test_data:
    if len(x)>1:
        test_list.append((x[0],x[1]))


# In[27]:


tct=0
for x in test_data:
    if len(x)>1:
        if x[0]=='1':
            tct=tct+1
tct


# In[28]:


test_sentence_list=[]
st_test=[]
flag_test=0
for x in test_data:
    if len(x)>1:
            if x[0]=='1':
                flag_test=flag_test+1
                if flag_test == 1:
                    st_test=[]
                    st_test.append((x[1]))
                elif flag_test==3684:
                    print(flag_test)
                    test_sentence_list.append(st_test)
                    st_test=[]
                    st_test.append((x[1]) )
                    test_sentence_list.append(st_test)
                    break
                else: 
                    test_sentence_list.append(st_test)
                    st_test=[]
                    st_test.append((x[1]))  
            else:
                st_test.append((x[1]))


# ### Data set creation

# In[29]:


# sentences and labels
sentences_train = [[t[0] for t in sublst] for sublst in train_sentence_list]
labels_train = [[t[1] for t in sublst] for sublst in train_sentence_list]

sentences_dev = [[t[0] for t in sublst] for sublst in dev_sentence_list]
labels_dev = [[t[1] for t in sublst] for sublst in dev_sentence_list]


# In[31]:


class creating_iterator(torch.utils.data.Dataset):
    
    def __init__(self,sentences, labels):
        self.sentences = sentences
        self.labels = labels
        
    def __len__(self):
        return len(self.sentences)
        
    def __getitem__(self, index):
        dataset = self.sentences[index]
        class_label  = self.labels[index]
        
        return dataset, class_label


# In[32]:


def st_code(sentences, char2idx,flag=True):
    st_code_lt=[]
    for sentence in sentences:
        st=[]
        for char in sentence:
            if char in char2idx:
                st.append(char2idx[char])
            else:
                if flag:
                  st.append(1)
        st_code_lt.append(st)
    return st_code_lt


# In[33]:


word2idx['<PAD>'],tag2idx['<PAD>']


# In[34]:


converted_sentences_train = st_code(sentences_train, word2idx, flag=True)
converted_labels_train = st_code(labels_train, tag2idx, flag=False)
converted_sentences_dev = st_code(sentences_dev, word2idx, flag=True)
converted_labels_dev = st_code(labels_dev, tag2idx, flag=False)


# In[35]:


train_dataset_fnn = creating_iterator(converted_sentences_train, converted_labels_train)
test_dataset_fnn = creating_iterator(converted_sentences_dev, converted_labels_dev)


# In[36]:


batch_size=16


# In[37]:


def collate_fn(batch):
    # Separate the sentences and labels in the batch
    sentences, labels = zip(*batch)

    # Pad the sentences with zeros using pad_sequence
    padded_sentences = pad_sequence([torch.LongTensor(sentence) for sentence in sentences], batch_first=True)
    
    # Pad the labels with zeros using pad_sequence
    padded_labels = pad_sequence([torch.LongTensor(label) for label in labels], batch_first=True)
    
    # Calculate the sentence lengths
    sentence_lengths = torch.LongTensor([len(s) for s in sentences])

    return padded_sentences, padded_labels, sentence_lengths


# In[38]:


# create PyTorch DataLoader objects for batching the data
train_loader = DataLoader(train_dataset_fnn, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(test_dataset_fnn, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# ### Model

# In[181]:


tag_pad_idx=tag2idx['<PAD>']
word_pad_idx=word2idx['<PAD>']


# In[182]:


word_pad_idx


# In[183]:


class BiLSTM(nn.Module):

  def __init__(self, input_dim, embedding_dim, hidden_dim, num_labels, lstm_layers, output_dim,
               emb_dropout, lstm_dropout, fc_dropout, word_pad_idx):
    super().__init__()
    self.embedding_dim = embedding_dim
    # LAYER 1: Embedding
    self.embedding = nn.Embedding(
        num_embeddings=input_dim, 
        embedding_dim=embedding_dim, 
        padding_idx=word_pad_idx
    )
    self.emb_dropout = nn.Dropout(emb_dropout)
    # LAYER 2: BiLSTM
    self.lstm = nn.LSTM(
        input_size=embedding_dim,
        hidden_size=hidden_dim,
        num_layers=lstm_layers,
        bidirectional=True,
        dropout=lstm_dropout if lstm_layers > 1 else 0
    )
    # LAYER 3: Fully-connected
    self.dropout3=nn.Dropout(lstm_dropout)
    self.elu = nn.ELU()
    self.fc = nn.Linear(hidden_dim * 2, output_dim)  
    self.linear2=nn.Linear(output_dim,num_labels)  

  def forward(self, sentence,sentence_lengths):
    
    embedded=self.embedding(sentence)
    
    # pack the sequences
    packed_embedded = pack_padded_sequence(embedded, sentence_lengths, batch_first=True, enforce_sorted=False)
        
    packed_output, (hidden, cell) = self.lstm(packed_embedded)
    
    # unpack the sequences
    output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
    
#     # concatenate the final hidden states from both directions
#     hidden = self.dropout3(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        
    ner_out = self.fc(self.elu(output))
    out=self.linear2(ner_out)
    return out

  def init_weights(self):
    # to initialize all parameters from normal distribution
    # helps with converging during training
    for name, param in self.named_parameters():
      nn.init.normal_(param.data, mean=0, std=0.1)

  def init_embeddings(self, word_pad_idx):
    # initialize embedding for padding as zero
    self.embedding.weight.data[word_pad_idx] = torch.zeros(self.embedding_dim)

  def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)


# In[184]:


bilstm = BiLSTM(
    input_dim=len(word2idx),
    embedding_dim=100,
    hidden_dim=256,
    num_labels = len(tag2idx),
    output_dim=128,
    lstm_layers=1,
    lstm_dropout=0.33,
    fc_dropout=0.25,
    emb_dropout=0.5,
    word_pad_idx=word_pad_idx
)
bilstm.init_weights()
bilstm.init_embeddings(word_pad_idx=word_pad_idx)
print(f"The model has {bilstm.count_parameters():,} trainable parameters.")
print(bilstm)


# #### Determining dev and test lengths to past in evaluation of model to obtain predicted tags.

# In[186]:


dev_lengths=[]
for sent in dev_sentence_list:
    dev_lengths.append(len(sent))


# In[187]:


test_lengths=[]
for sent in test_sentence_list:
    test_lengths.append(len(sent))


# ### Running

# In[188]:


from torch.optim.lr_scheduler import StepLR


# In[189]:


class NER(object):

  def __init__(self, model, train_loader, dev_loader,test_sentence_list,dev_sentence_list,optimizer_cls, loss_fn_cls):
    self.model = model
    self.data_train = train_loader
    self.data_dev=dev_loader
    self.optimizer = optimizer_cls(model.parameters(),lr=0.5)
    #print('ignore', tag_pad_idx)
    self.loss_fn = loss_fn_cls(ignore_index=tag_pad_idx)
    self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.1)
    print(self.scheduler)


  @staticmethod
  def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

  def accuracy(self, preds, y):
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    # print("max_preds",max_preds)
    non_pad_elements = (y != tag_pad_idx).nonzero()  # prepare masking for paddings
    # print("non_pad_elements",non_pad_elements)
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])


    
  def epoch(self):
      epoch_loss = 0
      epoch_acc = 0
      self.model.train()
      for text, true_tags,sentence_lengths in self.data_train:
        #print(true_tags.shape) [torch.Size([64, 41])
        self.optimizer.zero_grad()
        pred_tags = self.model(text,sentence_lengths)
        #print(pred_tags.shape) [torch.Size([64, 41, 10])]
        pred_tags = pred_tags.view(-1, pred_tags.shape[-1]) 
        #print(pred_tags.shape) [torch.Size([2624, 10])]
        true_tags = true_tags.view(-1)
        # print(true_tags.shape) [torch.Size([2624])]
        batch_loss = self.loss_fn(pred_tags, true_tags)
        batch_acc = self.accuracy(pred_tags, true_tags)
        batch_loss.backward()
        self.optimizer.step()
        epoch_loss += batch_loss.item()
        epoch_acc += batch_acc.item()

        self.scheduler.step( epoch_loss / len(self.data_train))
        # print("Last LR", self.scheduler.get_last_lr())
      return epoch_loss / len(self.data_train), epoch_acc / len(self.data_train)

  def evaluate(self):
      epoch_loss = 0
      epoch_acc = 0
      self.model.eval()
      with torch.no_grad():
          for text, true_tags,sentence_lengths in  self.data_dev:
              pred_tags = self.model(text,sentence_lengths)
              pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
              true_tags = true_tags.view(-1)
              batch_loss = self.loss_fn(pred_tags, true_tags)
              batch_acc = self.accuracy(pred_tags, true_tags)
              epoch_loss += batch_loss.item()
              epoch_acc += batch_acc.item()
              self.scheduler.step(epoch_loss / len(self.data_dev))
      return epoch_loss / len( self.data_dev), epoch_acc / len( self.data_dev)

  def train(self, n_epochs):
    valid_loss_min2 = np.Inf
    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc = self.epoch()
        end_time = time.time()
        epoch_mins, epoch_secs = NER.epoch_time(start_time, end_time)
        print(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrn Loss: {train_loss:.3f} | Trn Acc: {train_acc * 100:.2f}%")
        val_loss, val_acc = self.evaluate()
        print(f"\tVal Loss: {val_loss:.3f} | Val Acc: {val_acc * 100:.2f}%")
        if val_loss<valid_loss_min2:
            torch.save(self.model, 'colab_collate_20_Copy10.pt')
            valid_loss_min2=val_loss

    dev_inputs = []
    for sentence in dev_sentence_list:
        input_ids = [word2idx.get(word[0], word2idx['<UNK>']) for word in sentence]
        dev_inputs.append(torch.tensor(input_ids)) # Add batch dimension

    dev_inputs = nn.utils.rnn.pad_sequence(dev_inputs, batch_first=True)

    # Run the inputs through the model to get predicted tag sequences
    ner.model.eval()
    with torch.no_grad():
        outputs_dev = ner.model(dev_inputs,dev_lengths)
    _, dev_predicted_tags = torch.max(outputs_dev, dim=2)


    # Convert the predicted tag sequences to string representations
    predictions_dev = []
    for sentence_tags in dev_predicted_tags:
        predicted_tags_list = [idx2tag[idx.item()] for idx in sentence_tags]
        predictions_dev.append(predicted_tags_list)

    # Save the predictions to a file
    with open('colab_collate_predictions_dev_20_Copy11.txt', 'w') as f:
        for predicted_tags in predictions_dev:
            f.write(' '.join(predicted_tags) + '\n')


    inputs = []
    for sentence in test_sentence_list:
        input_ids = [word2idx.get(word, word2idx['<UNK>']) for word in sentence]
        inputs.append(torch.tensor(input_ids)) # Add batch dimension

    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)

    # Run the inputs through the model to get predicted tag sequences
    ner.model.eval()
    with torch.no_grad():
        outputs = ner.model(inputs,test_lengths)
    _, predicted_tags = torch.max(outputs, dim=2)

    # Convert the predicted tag sequences to string representations
    predictions_test = []
    for sentence_tags in predicted_tags:
        predicted_tags_list = [idx2tag[idx.item()] for idx in sentence_tags]
        predictions_test.append(predicted_tags_list)

    # Save the predictions to a file
    with open('colab_collate_predictions_test_20_Copy11.txt', 'w') as f:
        for predicted_tags in predictions_test:
            f.write(' '.join(predicted_tags) + '\n')


# In[190]:


import time


# In[191]:


get_ipython().run_cell_magic('time', '', '# this will continue training if the model has been trained before.\n# to restart training, run the bilstm creation cell (2 cells above) once again.\nner = NER(\n  model=bilstm,\n  train_loader=train_loader, \n  dev_loader=dev_loader,\n  optimizer_cls=optim.SGD,\n  loss_fn_cls=nn.CrossEntropyLoss,\n  test_sentence_list=test_sentence_list,\n  dev_sentence_list=dev_sentence_list\n)\nner.train(10)\n')


# ### Processing Dev Predictions to form .out file for submission and Score Check

# #### The dev_pred file create above is called, and the data is appended in a list

# In[192]:


pred_dev=[]
with open('colab_collate_predictions_dev_20_Copy11.txt', 'r') as readFile:
        for inputs in readFile:
            pred_dev.append(inputs.split(' '))


# In[193]:


len(pred_dev[2])


# #### A list of sentences is created using the dev data such that each sentence is a list whose individual elements store the index of the word in the sentence, the word and the tag from the dev file (basically all the info present in the file)

# In[194]:


dev_res_list=[]
dev_st=[]
dev_flag=0
for x in dev_data:
    if len(x)>1:
            if x[0]=='1':
                dev_flag=dev_flag+1
                if dev_flag == 1:
                    dev_st=[]
                    dev_st.append((x[0],x[1],x[2]))
                elif dev_flag==3466:
                    print(dev_flag)
                    dev_res_list.append(dev_st)
                    dev_st=[]
                    dev_st.append((x[0],x[1],x[2]))  
                    dev_res_list.append(dev_st)
                    break
                else: 
                    dev_res_list.append(dev_st)
                    dev_st=[]
                    dev_st.append((x[0],x[1],x[2]))  
            else:
                dev_st.append((x[0],x[1],x[2]))


# #### To create output file, the paddings are truncated from the output read from the DEV PRED file

# In[195]:


ik=0
tg=[]
for sent in pred_dev:
    th=[]
    lenk=0
    for jk in sent:
        if lenk<dev_lengths[ik]:
            th.append(jk)
            lenk=lenk+1
    tg.append(th)
    ik=ik+1


# #### To calculate the F1 score, a file is created which includes the dev file data along with predicted tag per word.

# In[196]:


result_dict = {}
idx = 0
for i in range(len(dev_res_list)):
    for j in range(len(dev_res_list[i])):
        result_dict[idx] = (dev_res_list[i][j][0], dev_res_list[i][j][1],dev_res_list[i][j][2], tg[i][j])
        idx += 1
        
print(result_dict)


# In[197]:


start_i=0
with open("colab_collate_predictions_dev_20_output_Copy11.txt", 'w') as f: 
    for key,i in result_dict.items() : 
        if i[0] == '1' and start_i!=0:
            f.write('\n')
            f.write('%s %s %s %s\n' % (i[0], i[1], i[2], i[3]))
        else:
            f.write('%s %s %s %s\n' % (i[0], i[1], i[2], i[3]))
            start_i=start_i+1


# In[198]:


start_i=0
with open("colab_collate_predictions_dev_20_output_Copy11.out", 'w') as f: 
    for key,i in result_dict.items() : 
        if i[0] == '1' and start_i!=0:
            f.write('\n')
            f.write('%s %s %s %s\n' % (i[0], i[1], i[2], i[3]))
        else:
            f.write('%s %s %s %s\n' % (i[0], i[1], i[2], i[3]))
            start_i=start_i+1


# ### Score Check using conll03eval

# In[199]:


get_ipython().system("perl conll03eval < {'colab_collate_predictions_dev_20_output_Copy11.txt'}")


# ### Processing Test Predictions to form .out file for submission

# #### A list of sentences is created using the test data such that each sentence is a list whose individual elements store the index of the word in the sentence and the word a (basically all the info present in the file)

# In[200]:


test_res_list=[]
st_test=[]
flag_test=0
for x in test_data:
    if len(x)>1:
            if x[0]=='1':
                flag_test=flag_test+1
                if flag_test == 1:
                    st_test=[]
                    st_test.append((x[0],x[1]))
                elif flag_test==3684:
                    print(flag_test)
                    test_res_list.append(st_test)
                    st_test=[]
                    st_test.append((x[0],x[1]) )
                    test_res_list.append(st_test)
                    break
                else: 
                    test_res_list.append(st_test)
                    st_test=[]
                    st_test.append((x[0],x[1]))  
            else:
                st_test.append((x[0],x[1]))


# #### The test_pred file create above is called, and the data is appended in a list

# In[201]:


pred_test=[]
with open('colab_collate_predictions_test_20_Copy11.txt', 'r') as readFile:
        for inputs in readFile:
            pred_test.append(inputs.split(' '))


# In[202]:


len(pred_test[2])


# #### To create output file, the paddings are truncated from the output read from the TEST PRED file

# In[203]:


test_ik=0
test_tg=[]
for sent in pred_test:
    test_th=[]
    test_lenk=0
    for jk in sent:
        if test_lenk<test_lengths[test_ik]:
            test_th.append(jk)
            test_lenk=test_lenk+1
    test_tg.append(test_th)
    test_ik=test_ik+1


# #### Output file is created which includes the dev file data along with predicted tag per word.

# In[204]:


test_dict = {}
test_idx = 0
for i in range(len(test_res_list)):
    for j in range(len(test_res_list[i])):
        test_dict[test_idx] = (test_res_list[i][j][0], test_res_list[i][j][1], test_tg[i][j])
        test_idx += 1
        
print(test_dict)


# In[205]:


start_ie=0
with open("colab_collate_predictions_test_20_output_Copy11.txt", 'w') as f: 
    for key,i in test_dict.items() : 
        if i[0] == '1' and start_ie!=0:
            f.write('\n')
            f.write('%s %s %s\n' % (i[0], i[1], i[2]))
        else:
            f.write('%s %s %s\n' % (i[0], i[1], i[2]))
            start_ie=start_ie+1


# In[206]:


start_ie=0
with open("colab_collate_predictions_test_20_output_Copy11.out", 'w') as f: 
    for key,i in test_dict.items() : 
        if i[0] == '1' and start_ie!=0:
            f.write('\n')
            f.write('%s %s %s\n' % (i[0], i[1], i[2]))
        else:
            f.write('%s %s %s\n' % (i[0], i[1], i[2]))
            start_ie=start_ie+1


# ### SAVING MODEL FOR SUBMISSION

# In[208]:


torch.save(ner.model,'epoch20_T1_Copy11.pt')


# ### Reloading Saved Model to verify that the correct model is saved and it reciprocates the actual result

# In[209]:


### SAVED MODEL


# In[210]:


# Load the model
modelw = torch.load("epoch20_T1_Copy11.pt")

dev_inputs = []
for sentence in dev_sentence_list:
    input_ids = [word2idx.get(word[0], word2idx['<UNK>']) for word in sentence]
    dev_inputs.append(torch.tensor(input_ids)) # Add batch dimension

dev_inputs = nn.utils.rnn.pad_sequence(dev_inputs, batch_first=True)

# Run the inputs through the model to get predicted tag sequences
modelw.eval()
with torch.no_grad():
    outputs_dev = modelw(dev_inputs,dev_lengths)
_, dev_predicted_tags = torch.max(outputs_dev, dim=2)


# Convert the predicted tag sequences to string representations
predictions_dev = []
for sentence_tags in dev_predicted_tags:
    predicted_tags_list = [idx2tag[idx.item()] for idx in sentence_tags]
    predictions_dev.append(predicted_tags_list)

# Save the predictions to a file
with open('dev_check_ep20_Copy11.txt', 'w') as f:
    for predicted_tags in predictions_dev:
        f.write(' '.join(predicted_tags) + '\n')


inputs = []
for sentence in test_sentence_list:
    input_ids = [word2idx.get(word, word2idx['<UNK>']) for word in sentence]
    inputs.append(torch.tensor(input_ids)) # Add batch dimension

inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)

# Run the inputs through the model to get predicted tag sequences
modelw.eval()
with torch.no_grad():
    outputs = modelw(inputs,test_lengths)
_, predicted_tags = torch.max(outputs, dim=2)

# Convert the predicted tag sequences to string representations
predictions_test = []
for sentence_tags in predicted_tags:
    predicted_tags_list = [idx2tag[idx.item()] for idx in sentence_tags]
    predictions_test.append(predicted_tags_list)

# Save the predictions to a file
with open('test_check_ep20_Copy11.txt', 'w') as f:
    for predicted_tags in predictions_test:
        f.write(' '.join(predicted_tags) + '\n')



# ### Processing Dev Predictions using LOADED MODEL to form .out file for submission and Score Check

# In[211]:


pred_dev_trial=[]
with open('dev_check_ep20_Copy11.txt', 'r') as readFile:
        for inputs in readFile:
            pred_dev_trial.append(inputs.split(' '))


# In[212]:


len(pred_dev_trial[2])


# In[213]:


dev_res_list=[]
dev_st=[]
dev_flag=0
for x in dev_data:
    if len(x)>1:
            if x[0]=='1':
                dev_flag=dev_flag+1
                if dev_flag == 1:
                    dev_st=[]
                    dev_st.append((x[0],x[1],x[2]))
                elif dev_flag==3466:
                    print(dev_flag)
                    dev_res_list.append(dev_st)
                    dev_st=[]
                    dev_st.append((x[0],x[1],x[2]))  
                    dev_res_list.append(dev_st)
                    break
                else: 
                    dev_res_list.append(dev_st)
                    dev_st=[]
                    dev_st.append((x[0],x[1],x[2]))  
            else:
                dev_st.append((x[0],x[1],x[2]))


# In[214]:


ikdev_trial=0
tgdev_trial=[]
for sent in pred_dev_trial:
    th=[]
    lenk=0
    for jk in sent:
        if lenk<dev_lengths[ikdev_trial]:
            th.append(jk)
            lenk=lenk+1
    tgdev_trial.append(th)
    ikdev_trial=ikdev_trial+1


# In[215]:


result_dictdev_trial = {}
idxdev_trial = 0
for i in range(len(dev_res_list)):
    for j in range(len(dev_res_list[i])):
        result_dictdev_trial[idxdev_trial] = (dev_res_list[i][j][0], dev_res_list[i][j][1],dev_res_list[i][j][2], tgdev_trial[i][j])
        idxdev_trial += 1
        
print(result_dictdev_trial)


# In[216]:


start_idev_trial=0
with open("dev_check_ep20_output_Copy11.txt", 'w') as f: 
    for key,i in result_dictdev_trial.items() : 
        if i[0] == '1' and start_idev_trial!=0:
            f.write('\n')
            f.write('%s %s %s %s\n' % (i[0], i[1], i[2], i[3]))
        else:
            f.write('%s %s %s %s\n' % (i[0], i[1], i[2], i[3]))
            start_idev_trial=start_idev_trial+1


# In[217]:


get_ipython().system("perl conll03eval < {'dev_check_ep20_output_Copy11.txt'}")


# In[75]:




