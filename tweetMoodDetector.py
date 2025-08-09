import torch
import spacy
import json
from tqdm import trange, tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch import optim
import matplotlib.pyplot as plt

#tweets = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding="latin1", header = None)
#tweets.columns = ['rating','id','date','flag','user','text']
#tweets = tweets[['rating','text']]
#tweets.to_parquet('tweets.parquet')


#-------------------------Tokenize-------------------------------------

nlp = spacy.load("en_core_web_sm")

def tokenize(text):

    processedTweet = nlp(text)

    tokens = [token.text for token in processedTweet]

    return tokens

def tokenizeMany(texts):
    return [[token.text.lower() for token in doc] for doc in tqdm(nlp.pipe(texts, batch_size=100, disable=["parser", "ner", "tagger"]))]

#-------------------------Vocabulary-------------------------------------

def buildVocab(counter, minFreq=1, specials=['<pad>', '<sos>', '<eos>', '<unk>']):
    vocab = {}
    idx = 0

    # Add special tokens
    for token in specials:
        vocab[token] = idx
        idx += 1

    # Add regular tokens
    for word, freq in counter.items():
        if freq >= minFreq:
            vocab[word] = idx
            idx += 1

    return vocab

def indexVocab(tokens, vocab):
    return [vocab.get(token, vocab['<unk>']) for token in tokens]

def newVocab(counter):
    vocab = buildVocab(counter, minFreq=3)
    with open('vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

def loadVocab():
    with open('vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab


# tokenizedTweets = tokenizeMany(tweets['text'])

# wordCounts = Counter()
# for tokens in tokenizedTweets:
#     wordCounts.update(tokens)
    

# newVocab(wordCounts) 

#-------------------------Pre-processing-------------------------------------

def preprocessTweet(text, vocab, unkID, sosID,eosID):
    tokens = [token.text.lower() for token in tokenize(text)]
    
    tokenNumbers = [vocab.get(token, unkID) for token in tokens]
    tokenNumbers = [sosID] + tokenNumbers + [eosID]
    
    return tokenNumbers

def preprocessData(data, vocab):
    unkID = vocab["<unk>"]
    sosID = vocab["<sos>"]
    eosID = vocab["<eos>"]
    padID = vocab["<pad>"]
    processedData = []
    
    for tweet in tqdm(data, desc="Preprocessing tweets"):
        processedTweet = preprocessTweet(tweet, vocab, unkID, sosID,eosID)
        processedData.append(torch.tensor(processedTweet, dtype = torch.long))
    
    paddedData = pad_sequence(processedData, batch_first=True, padding_value=padID)
    
    return paddedData


#-------------------------Model-------------------------------------



#-------------------------Load data-------------------------------------

tweets = pd.read_parquet('tweets.parquet')

X = tweets['text']      # features
y = tweets['rating']    # labels

xTrain, xTest, yTrain, yTest = train_test_split(
    X, y,
    test_size=0.2,       # 20% test data
    random_state=42,     # for reproducibility
    stratify=y           # keeps class distribution (optional but good for classification)
)

vocab = loadVocab()

#-------------------------Model-------------------------------------

class LSTM(nn.Module):
    def __init__(self, numEmbeddings, outputSize, numLayers = 1, hiddenSize = 128):
        super(LSTM, self).__init__()
        
        self.embedding = nn.Embedding(numEmbeddings, hiddenSize)
        
        self.lstm = nn.LSTM(input_size = hiddenSize, hidden_size = hiddenSize, num_layers = numLayers, batch_first = True, dropout = 0.5)
        
        self.fcOutput = nn.Linear(hiddenSize, outputSize)
    
    def forward(self, input, hiddenIn, memoryIn):
        embInput = self.embedding(input)
        
        output, (hiddenOut, memoryOut) =  self.lstm(embInput, (hiddenIn, memoryIn))
        
        return self.fcOutput(output), hiddenOut, memoryOut


hiddenSize = 128

layers = 2

lstmClassifier = LSTM(numEmbeddings = len(vocab), outputSize = 5, numLayers = layers, hiddenSize = hiddenSize)
#-------------------------Training-------------------------------------

device = torch.device(0 if torch.cuda.is_available() else 'cpu')

learningRate = 1e-4
epochs = 20
batchSize = 32

xTrainTensor = preprocessData(xTrain.tolist(), vocab)
yTrainTensor = torch.tensor(yTrain.values, dtype=torch.long)

xTestTensor = preprocessData(xTest.tolist(), vocab)
yTestTensor = torch.tensor(yTest.values, dtype=torch.long)

trainingSet = TensorDataset(xTrainTensor, yTrainTensor)

testingSet = TensorDataset(xTestTensor, yTestTensor)

optimiser = optim.Adam(lstmClassifier.parameters(), lr = learningRate)

lossFunction = nn.CrossEntropyLoss()


def training(model, optimiser, lossFunction, epochs, trainingData, testingData, device, hiddenSize, layers):
    model.to(device)
    
    trainingLoss = []
    testingLoss = []
    trainingAccuracy = []
    testingAccuracy = []
    
    trainingLoader = DataLoader(trainingData, batch_size=32, shuffle=True)
    testingLoader = DataLoader(testingData, batch_size=32, shuffle=True)

    progress = trange(0, epochs, leave=False, desc="Epoch")
    
    

# Loop through each epoch
    for epoch in progress:
        currentTrainingAcc = 0
        currentTrainingLoss = 0
        currentTestingAcc = 0
        currentTestingLoss = 0
        
        # Update progress bar description with current accuracy
        progress.set_postfix_str('Accuracy: Train %.2f%%, Test %.2f%%' % (currentTrainingAcc * 100, currentTestingAcc * 100))
        
        model.train()
        steps = 0
        for tweet, label in tqdm(trainingLoader, desc="Training", leave=False):
            batchSize = label.shape[0]
            
            tweet = tweet.to(device)
            label = label.to(device)
            
            #initialise hidden and memory states
            hidden = torch.zeros(layers, batchSize, hiddenSize, device=device)
            memory = torch.zeros(layers, batchSize, hiddenSize, device=device)
            
            #forward pass
            
            prediction, hidden, memory = model(tweet, hidden, memory)
            
            loss = lossFunction(prediction[:, -1, :], label)
            
            # Backprop and optimisation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            currentTrainingLoss += loss.item()
            
            currentTrainingAcc += (prediction[:, -1, :].argmax(1) == label).sum()
            steps += batchSize
            
        trainingLoss.append(currentTrainingLoss/len(trainingLoader))
        currentTrainingAcc = (currentTrainingAcc/steps).item()
        trainingAccuracy.append(currentTrainingAcc)
        
        #Testing
        model.eval()
        steps = 0
        
        with torch.no_grad():
            for tweet, label in tqdm(testingLoader, desc="Testing", leave=False):
                batchSize = label.shape[0]
                
                tweet = tweet.to(device)
                label = label.to(device)
                
                hidden = torch.zeros(layers, batchSize, hiddenSize, device=device)
                memory = torch.zeros(layers, batchSize, hiddenSize, device=device)
                
                #forward pass
                prediction, hidden, memory = model(tweet, hidden, memory)
                
                loss = lossFunction(prediction[:, -1, :], label)
                
                currentTestingLoss += loss.item()
                
                currentTestingAcc += (prediction[:, -1, :].argmax(1) == label).sum()
                steps += batchSize

            testingLoss.append(currentTestingLoss/len(testingLoader))
            currentTestingAcc = (currentTestingAcc/steps).item()
            testingAccuracy.append(currentTestingAcc)
    
    
    #plot loss
    _ = plt.figure(figsize=(10, 5))
    _ = plt.plot(np.linspace(0, epochs, len(trainingLoss)), trainingLoss)
    _ = plt.plot(np.linspace(0, epochs, len(testingLoss)), testingLoss)

    _ = plt.legend(["Train", "Test"])
    _ = plt.title("Training Vs Test Loss")
    _ = plt.xlabel("Epochs")
    _ = plt.ylabel("Loss")
    
    
    #plot accuracy
    _ = plt.figure(figsize=(10, 5))
    _ = plt.plot(np.linspace(0, epochs, len(trainingAccuracy)), trainingAccuracy)
    _ = plt.plot(np.linspace(0, epochs, len(testingAccuracy)), testingAccuracy)

    _ = plt.legend(["Train", "Test"])
    _ = plt.title("Training Vs Test Accuracy")
    _ = plt.xlabel("Epochs")
    _ = plt.ylabel("Accuracy")



    torch.save(lstmClassifier.state_dict(), 'lstmClassifier.pth')
    
#-------------------------Running-------------------------------------


training(
    model=lstmClassifier,
    optimiser=optimiser,
    lossFunction=lossFunction,
    epochs=epochs,
    trainingData=trainingSet,
    testingData=testingSet,
    device=device,
    hiddenSize=hiddenSize,
    layers=layers
)
#indexedTweets = [indexVocab(tokens, vocab) for tokens in tokenizedTweets]