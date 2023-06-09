#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import operator
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
import numpy as np


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[3]:


import time


# ### Train Data

# In[4]:


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

# In[6]:


ct=0
for x in training_data:
    if len(x)>1:
        if x[0]=='1':
            ct=ct+1
ct


# In[7]:


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


# In[8]:


len(train_sentence_list)


# In[9]:


train_sentence_list[-2:]


# ### Dev Data

# In[10]:


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

# In[11]:


dev_ct=0
for x in dev_data:
    if len(x)>1:
        if x[0]=='1':
            dev_ct=dev_ct+1
dev_ct


# In[12]:


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


# In[13]:


len(dev_sentence_list)


# ### Vocab creation

# In[14]:


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


# In[15]:


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

# In[16]:


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


# In[19]:


len(word2idx)


# ### Tag2Idx Mapping

# In[20]:


tag_dict={}
count=0
for x in training_data:
    if len(x)>1:
        if x[2] in tag_dict:
            temp=tag_dict[x[2]]
            tag_dict[x[2]]=temp+1
        else:
            tag_dict[x[2]]=1


# In[21]:


# create a mapping from tags to integers
tag2idx ={}
tag_ind=0
tag2idx['<PAD>']= tag_ind
for key, value in tag_dict.items():
        tag_ind=tag_ind+1
        tag2idx[key]=tag_ind

    


# In[22]:


tag2idx


# ### Test  data

# In[23]:


idx2tag={}
for key, value in tag2idx.items():
    idx2tag[value]=key


# In[24]:


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


# In[25]:


tct=0
for x in test_data:
    if len(x)>1:
        if x[0]=='1':
            tct=tct+1
tct


# In[26]:


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


# In[26]:





# ### GloVE Embeddings

# In[27]:


# Load the GloVe word embeddings
gloveembeddings_index = {}
with open("/content/drive/MyDrive/Colab Notebooks/data/glove.6B.100d", 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:],dtype='float32')
        gloveembeddings_index[word] = coefs


# #### Finding a mean array- array with mean of values at that index( considering all words)

# In[28]:


a = np.array(list(gloveembeddings_index.values()))
# calculate the mean of the array along axis 0 (i.e., column-wise)
mean_array = np.mean(a, axis=0)
mean_array


# In[29]:


word2idx['<PAD>']


# In[30]:


gloveembeddings_index['<UNK>']=mean_array
gloveembeddings_index['<PAD>']=np.zeros(100)


# In[31]:


embedding_dim = 100
embedding_matrix = torch.zeros(len(word2idx), embedding_dim)

for word, i in word2idx.items():

    if word.lower() in gloveembeddings_index:
        gl_arr=gloveembeddings_index.get(word.lower())
    else:
        # use a separate vector for unknown words
        gl_arr=mean_array
        
    embedding_vector = torch.tensor([float(val) for val in gl_arr])
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[32]:


embedding_matrix[-1]


# In[33]:


embedding_matrix.shape


# ### Data set creation

# In[35]:


# sentences and labels
sentences_train = [[t[0] for t in sublst] for sublst in train_sentence_list]
labels_train = [[t[1] for t in sublst] for sublst in train_sentence_list]

sentences_dev = [[t[0] for t in sublst] for sublst in dev_sentence_list]
labels_dev = [[t[1] for t in sublst] for sublst in dev_sentence_list]


# In[36]:


sentences_test = [[t[0] for t in sublst] for sublst in test_sentence_list]


# In[37]:


sentences_train[:4]


# In[38]:


tag2idx


# #### Iterator for Train and Dev data- converts words to numbers, converts tags to numbers using index dictionaries created and creates a list of booolean mask per word in the sentences 

# In[39]:


class creating_iterator(torch.utils.data.Dataset):
    def __init__(self, sentences, labels, word2idx, tag2idx):
        self.sentences = sentences
        self.labels = labels
        self.word2idx = word2idx
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        class_label  = self.labels[idx]
        
        # Create a list of boolean flags where 1 corresponds to lowercase and 0 corresponds to uppercase
        sentence_flags = [] 
        for word in sentence:
          if word.lower()==word:
            sentence_flags.append(1)
          else:
            sentence_flags.append(0)

        # Convert the words to their corresponding indices using word2idx

        converted_sentence = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in sentence]

        # Convert the labels to their corresponding indices using word2idx
        converted_labels = [self.tag2idx.get(tag, 0) for tag in class_label]
        
        
        return converted_sentence, converted_labels, sentence_flags


# In[40]:


train_dataset_fnn = creating_iterator(sentences_train, labels_train, word2idx, tag2idx)
dev_dataset_fnn = creating_iterator(sentences_dev, labels_dev, word2idx, tag2idx)


# In[41]:


batch_size=16


# #### collate function used to pad the sentences, corresponding labels and boolean masks. I also calculates length of the sentences to be used in pack_padded in future.

# In[42]:


def collate_fn(batch):
    # Separate the sentences and labels in the batch
    sentences, labels, flags = zip(*batch)

    # Pad the sentences with zeros using pad_sequence
    padded_sentences = pad_sequence([torch.LongTensor(sentence) for sentence in sentences], batch_first=True)
    
    # Pad the labels with zeros using pad_sequence
    padded_labels = pad_sequence([torch.LongTensor(label) for label in labels], batch_first=True)
    
    # Calculate the sentence lengths
    sentence_lengths = torch.LongTensor([len(s) for s in sentences])

    sent_flags=pad_sequence([torch.LongTensor(flag) for flag in flags], batch_first=True)


    return padded_sentences, padded_labels, sentence_lengths, sent_flags


# In[43]:


# create PyTorch DataLoader objects for batching the data
train_loader = DataLoader(train_dataset_fnn, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_dataset_fnn, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# #### Similar iterator and collate functions for test, which does not include label related tasks and data

# In[44]:


class test_creating_iterator(torch.utils.data.Dataset):
    def __init__(self, sentences,word2idx):
        self.sentences = sentences
        self.word2idx = word2idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        # Create a list of boolean flags where 1 corresponds to lowercase and 0 corresponds to uppercase
        sentence_flags = [] 
        for word in sentence:
          if word.lower()==word:
            sentence_flags.append(1)
          else:
            sentence_flags.append(0)
        
        # Convert the words to their corresponding indices using word2idx
        converted_sentence = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in sentence]

        return converted_sentence,sentence_flags


# In[45]:


def test_collate_fn(batch):
    # Separate the sentences and labels in the batch
    sentences, flags = zip(*batch)

    # Pad the sentences with zeros using pad_sequence
    padded_sentences = pad_sequence([torch.LongTensor(sentence) for sentence in sentences], batch_first=True)
    
    # Calculate the sentence lengths
    sentence_lengths = torch.LongTensor([len(s) for s in sentences])

    sent_flags=pad_sequence([torch.LongTensor(flag) for flag in flags], batch_first=True)


    return padded_sentences, sentence_lengths, sent_flags


# In[46]:


test_dataset_fnn=test_creating_iterator(sentences_test,word2idx)
test_loader = DataLoader(test_dataset_fnn, batch_size=batch_size, shuffle=False, collate_fn=test_collate_fn)


# In[47]:


len(test_dataset_fnn)


# ### Model

# In[84]:


tag_pad_idx=tag2idx['<PAD>']
word_pad_idx=word2idx['<PAD>']


# In[85]:


word_pad_idx


# In[86]:


class Glove_EmbedLSTM(nn.Module):

  def __init__(self, input_dim, embedding_dim, hidden_dim, num_labels, lstm_layers, output_dim,
               emb_dropout, lstm_dropout, fc_dropout, word_pad_idx,pretrained_embed):
    super().__init__()
    self.embedding_dim = embedding_dim
    # LAYER 1: Embedding
    self.embedding=nn.Embedding.from_pretrained(pretrained_embed, freeze = False)
    
    self.emb_dropout = nn.Dropout(emb_dropout)
    # LAYER 2: BiLSTM
    self.lstm = nn.LSTM(
        input_size=101,
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

  def forward(self, sentence, sentence_lengths, sentence_flags):
    # Get the embeddings for the sentence
    embedded = self.embedding(sentence)
    # print(embedded.shape)
    concatenated_tensor = torch.cat((embedded, sentence_flags.unsqueeze(-1)), dim=-1)
    # print(concatenated_tensor.shape)
    # Pack the sequences
    packed_embedded = pack_padded_sequence(concatenated_tensor, sentence_lengths, batch_first=True, enforce_sorted=False)
    # print(packed_embedded.data.shape)
    packed_output, (hidden, cell) = self.lstm(packed_embedded)
    # print(packed_output.data.shape)
    # Unpack the sequences
    output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
    # print(packed_output.data.shape)
    ner_out = self.fc(self.elu(output))
    out = self.linear2(ner_out)
    
    return out

  def init_weights(self):
    # to initialize all parameters from normal distribution
    # helps with converging during training
    for name, param in self.named_parameters():
      nn.init.normal_(param.data, mean=0, std=0.1)

  # def init_embeddings(self, word_pad_idx):
  #   # initialize embedding for padding as zero
  #   self.embedding.weight.data[word_pad_idx] = torch.zeros(self.embedding_dim)

  def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)


# In[87]:


Gbilstm = Glove_EmbedLSTM(
    input_dim=len(word2idx),
    embedding_dim=100,
    hidden_dim=256,
    num_labels = len(tag2idx),
    output_dim=128,
    lstm_layers=1,
    lstm_dropout=0.33,
    fc_dropout=0.25,
    emb_dropout=0.5,
    word_pad_idx=word_pad_idx,
    pretrained_embed = embedding_matrix
)
Gbilstm.init_weights()
# Gbilstm.init_embeddings(word_pad_idx=word_pad_idx)
print(f"The model has {Gbilstm.count_parameters():,} trainable parameters.")
print(Gbilstm)


# In[88]:


dev_lengths=[]
for sent in dev_sentence_list:
    dev_lengths.append(len(sent))


# In[89]:


test_lengths=[]
for sent in test_sentence_list:
    test_lengths.append(len(sent))


# ### Running

# In[90]:


from torch.optim.lr_scheduler import StepLR


# In[91]:


class NER(object):

  def __init__(self, model, train_loader,test_loader, dev_loader,test_sentence_list,dev_sentence_list,optimizer_cls, loss_fn_cls):
    self.model = model
    self.data_train = train_loader
    self.data_dev=dev_loader
    self.data_test=test_loader
    self.optimizer = optimizer_cls(model.parameters(),lr=0.5)
    #print('ignore', tag_pad_idx)
    self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    self.scheduler = StepLR(self.optimizer, step_size=3, gamma=0.1)
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
      for text, true_tags,sentence_lengths,sentence_flags in self.data_train:
        # print(type(sentence_flags)).  tensor of list of lists
        # print("t1",true_tags.shape) #[torch.Size([64, 41])
        self.optimizer.zero_grad()
        pred_tags = self.model(text,sentence_lengths,sentence_flags) ##send bmask
        # print("p1",pred_tags.shape) #[torch.Size([64, 41, 10])]
        pred_tags = pred_tags.view(-1, pred_tags.shape[-1]) 
        # print(pred_tags)
        # print("p2",pred_tags.shape) #[torch.Size([2624, 10])]
        true_tags = true_tags.view(-1)
        # print(true_tags)
        # print("t2",true_tags.shape) #[torch.Size([2624])]
        batch_loss = self.loss_fn(pred_tags, true_tags)
        # print("batch_loss",batch_loss)
        batch_acc = self.accuracy(pred_tags, true_tags)
        batch_loss.backward()
        self.optimizer.step()
        epoch_loss += batch_loss.item()
        # print("epoch_loss",epoch_loss)
        epoch_acc += batch_acc.item()

        self.scheduler.step( epoch_loss / len(self.data_train))
        # print("Last LR", self.scheduler.get_last_lr())
      return epoch_loss / len(self.data_train), epoch_acc / len(self.data_train)

  def evaluate(self):
      epoch_loss = 0
      epoch_acc = 0
      self.model.eval()
      with torch.no_grad():
          for text, true_tags,sentence_lengths,sentence_flags in  self.data_dev:
              pred_tags = self.model(text,sentence_lengths,sentence_flags)
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
            torch.save(self.model, 'GloveEmbed_trial2.pt')
            valid_loss_min2=val_loss

    ############################################
    # Creating prediction file with the predicted tags for dev
    self.model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, targets,sent_len,sent_fl in self.data_dev:
            outputs = self.model(inputs,sent_len,sent_fl)
            _, preds = torch.max(outputs, dim=2)
            predictions.extend(preds.tolist())
            true_labels.extend(targets.tolist())

    print(predictions)
    # Convert the predicted tag sequences to string representations
    predictions_dev = []
    for sentence_tags in predictions:
        predicted_tags_list = [idx2tag[idx] for idx in sentence_tags]
        predictions_dev.append(predicted_tags_list)


    # Save the predictions to a file
    with open('GloveEmbed_trial2_dev_pred.txt', 'w') as f:
        for predicted_tags in predictions_dev:
            f.write(' '.join(predicted_tags) + '\n')

    ##########################################
    # Creating prediction file with the predicted tags for test
    self.model.eval()
    predictions_test = []
    with torch.no_grad():
        for inputs, sent_len,sent_fl in self.data_test:
            outputs = self.model(inputs,sent_len,sent_fl)
            _, preds = torch.max(outputs, dim=2)
            predictions_test.extend(preds.tolist())

    # Convert the predicted tag sequences to string representations
    test_preds = []
    for sentence_tags in predictions_test:
        predicted_tags_list = [idx2tag[idx] for idx in sentence_tags]
        test_preds.append(predicted_tags_list)


    # Save the predictions to a file
    with open('GloveEmbed_trial2_test_pred.txt', 'w') as f:
        for predicted_tags in test_preds:
            f.write(' '.join(predicted_tags) + '\n')




# In[92]:


get_ipython().run_cell_magic('time', '', '# this will continue training if the model has been trained before.\n# to restart training, run the bilstm creation cell (2 cells above) once again.\nner = NER(\n  model=Gbilstm,\n  train_loader=train_loader, \n  dev_loader=dev_loader,\n  test_loader=test_loader,\n  optimizer_cls=optim.SGD,\n  loss_fn_cls=nn.CrossEntropyLoss,\n  test_sentence_list=test_sentence_list,\n  dev_sentence_list=dev_sentence_list\n)\nner.train(10)\n')


# ### Processing Dev Predictions to form .out file for submission and Score Check

# #### The dev_pred file create above is called, and the data is appended in a list

# In[93]:


pred_dev=[]
with open('GloveEmbed_trial2_dev_pred.txt', 'r') as readFile:
        for inputs in readFile:
            pred_dev.append(inputs.split(' '))


# #### A list of sentences is created using the dev data such that each sentence is a list whose individual elements store the index of the word in the sentence, the word and the tag from the dev file (basically all the info present in the file)

# In[94]:


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


# #### To calculate the F1 score, a file is created which includes the dev file data along with predicted tag per word.

# In[ ]:


result_dict = {}
idx = 0
for i in range(len(dev_res_list)):
    for j in range(len(dev_res_list[i])):
        result_dict[idx] = (dev_res_list[i][j][0], dev_res_list[i][j][1],dev_res_list[i][j][2], pred_dev[i][j])
        idx += 1


# In[96]:


start_i=0
with open("GloveEmbed_trial2_dev_pred_out.txt", 'w') as f: 
    for key,i in result_dict.items() : 
        if i[0] == '1' and start_i!=0:
            f.write('\n')
            f.write('%s %s %s %s\n' % (i[0], i[1], i[2], i[3]))
        else:
            f.write('%s %s %s %s\n' % (i[0], i[1], i[2], i[3]))
            start_i=start_i+1


# In[97]:


start_i=0
with open("gl_dev2_trial2.out", 'w') as f: 
    for key,i in result_dict.items() : 
        if i[0] == '1' and start_i!=0:
            f.write('\n')
            f.write('%s %s %s\n' % (i[0], i[1], i[3]))
        else:
            f.write('%s %s %s\n' % (i[0], i[1], i[3]))
            start_i=start_i+1


# In[98]:


start_i=0
with open("GloveEmbed_trial2_dev_pred_out.out", 'w') as f: 
    for key,i in result_dict.items() : 
        if i[0] == '1' and start_i!=0:
            f.write('\n')
            f.write('%s %s %s %s\n' % (i[0], i[1], i[2], i[3]))
        else:
            f.write('%s %s %s %s\n' % (i[0], i[1], i[2], i[3]))
            start_i=start_i+1


# ### Score Check using conll03eval

# In[99]:


get_ipython().system("perl conll03eval < {'GloveEmbed_trial2_dev_pred_out.txt'}")


# ### Processing Test Predictions to form .out file for submission

# #### A list of sentences is created using the test data such that each sentence is a list whose individual elements store the index of the word in the sentence and the word a (basically all the info present in the file)

# In[101]:


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

# In[102]:


pred_test=[]
with open('GloveEmbed_trial2_test_pred.txt', 'r') as readFile:
        for inputs in readFile:
            pred_test.append(inputs.split(' '))


# #### Output file is created which includes the test file data along with predicted tag per word.

# In[106]:


test_dict = {}
test_idx = 0
for i in range(len(test_res_list)):
    for j in range(len(test_res_list[i])):
        test_dict[test_idx] = (test_res_list[i][j][0], test_res_list[i][j][1], pred_test[i][j])
        test_idx += 1


# In[107]:


start_ie=0
with open("GloveEmbed_trial2_test_pred_out.txt", 'w') as f: 
    for key,i in test_dict.items() : 
        if i[0] == '1' and start_ie!=0:
            f.write('\n')
            f.write('%s %s %s\n' % (i[0], i[1], i[2]))
        else:
            f.write('%s %s %s\n' % (i[0], i[1], i[2]))
            start_ie=start_ie+1


# In[108]:


start_ie=0
with open("GloveEmbed_trial2_test_pred_out.out", 'w') as f: 
    for key,i in test_dict.items() : 
        if i[0] == '1' and start_ie!=0:
            f.write('\n')
            f.write('%s %s %s\n' % (i[0], i[1], i[2]))
        else:
            f.write('%s %s %s\n' % (i[0], i[1], i[2]))
            start_ie=start_ie+1


# ### SAVING MODEL FOR SUBMISSION

# In[109]:


torch.save(ner.model,'GloveEmbed_trial2_m2.pt')


# In[110]:


### SAVED MODEL


# ### Reloading Saved Model to verify that the correct model is saved and it reciprocates the actual result

# In[111]:


# Load the model
modelw = torch.load("GloveEmbed_trial2_m2.pt")

############################################
# Creating prediction file with the predicted tags for dev
modelw.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for inputs, targets,sent_len,sent_fl in dev_loader:
        outputs = modelw(inputs,sent_len,sent_fl)
        _, preds = torch.max(outputs, dim=2)
        predictions.extend(preds.tolist())
        true_labels.extend(targets.tolist())

# Convert the predicted tag sequences to string representations
predictions_dev = []
for sentence_tags in predictions:
    predicted_tags_list = [idx2tag[idx] for idx in sentence_tags]
    predictions_dev.append(predicted_tags_list)

# Save the predictions to a file
with open('GloveEmbed_trial2_devcheck_pred.txt', 'w') as f:
    for predicted_tags in predictions_dev:
        f.write(' '.join(predicted_tags) + '\n')

##########################################
# Creating prediction file with the predicted tags for test
modelw.eval()
predictions_test = []
with torch.no_grad():
    for inputs, sent_len,sent_fl in test_loader:
        outputs = modelw(inputs,sent_len,sent_fl)
        _, preds = torch.max(outputs, dim=2)
        predictions_test.extend(preds.tolist())

# Convert the predicted tag sequences to string representations
test_preds = []
for sentence_tags in predictions_test:
    predicted_tags_list = [idx2tag[idx] for idx in sentence_tags]
    test_preds.append(predicted_tags_list)

# Save the predictions to a file
with open('GloveEmbed_trial2_testcheck_pred.txt', 'w') as f:
    for predicted_tags in test_preds:
        f.write(' '.join(predicted_tags) + '\n')



# ### Processing Dev Predictions using LOADED MODEL to form .out file for submission and Score Check

# In[112]:


pred_dev_trial=[]
with open('GloveEmbed_trial2_devcheck_pred.txt', 'r') as readFile:
        for inputs in readFile:
            pred_dev_trial.append(inputs.split(' '))


# In[113]:


len(pred_dev_trial[2])


# In[114]:


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


# In[ ]:


result_dictdev_trial = {}
idxdev_trial = 0
for i in range(len(dev_res_list)):
    for j in range(len(dev_res_list[i])):
        result_dictdev_trial[idxdev_trial] = (dev_res_list[i][j][0], dev_res_list[i][j][1],dev_res_list[i][j][2], pred_dev_trial[i][j])
        idxdev_trial += 1


# In[117]:


start_idev_trial=0
with open("GloveEmbed_trial2_devcheck_pred_out.txt", 'w') as f: 
    for key,i in result_dictdev_trial.items() : 
        if i[0] == '1' and start_idev_trial!=0:
            f.write('\n')
            f.write('%s %s %s %s\n' % (i[0], i[1], i[2], i[3]))
        else:
            f.write('%s %s %s %s\n' % (i[0], i[1], i[2], i[3]))
            start_idev_trial=start_idev_trial+1


# In[118]:


get_ipython().system("perl conll03eval < {'GloveEmbed_trial2_devcheck_pred_out.txt'}")


# In[120]:




