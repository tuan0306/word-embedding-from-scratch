import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Word2VecDataset(Dataset):
    def __init__(self,training_data):
        self.data=torch.tensor(training_data,dtype=torch.long)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index][0],self.data[index][1]
    
class Word2VecPytorch(nn.Module):
    def __init__(self,vocab_size,embedding_dim=50):
        super().__init__()
        self.W1=nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)
        self.W2=nn.Linear(in_features=embedding_dim,out_features=vocab_size,bias=False)
    
    def forward(self,target_word_id):
        h=self.W1(target_word_id)
        u=self.W2(h)
        return u
    
    def get_similar_words(self,word_id,top=3):
        W1_weights=self.W1.weight.detach()
        target_vector=W1_weights[word_id]
        cos_sim=nn.functional.cosine_similarity(target_vector.unsqueeze(0),W1_weights)
        scores,indices=torch.topk(cos_sim,top+1)
        result=[]
        for i in range(1,top+1):
            result.append((indices[i].item(),scores[i].item()))
        return result
    
class Word2VecPytorchTrainer:
    def __init__(self,model):
        self.model=model
        
    def train(self,dataloader, epochs=50, learning_rate=0.01):
        loss_his=[]
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        optimizer=optim.Adam(self.model.parameters(),lr=learning_rate)
        
        criterion=nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for target_batch,context_batch in dataloader:
                target_batch=target_batch.to(device)
                context_batch = context_batch.to(device)
                
                optimizer.zero_grad()
                
                predictions=self.model(target_batch)
                loss=criterion(predictions,context_batch)
                loss.backward()
                optimizer.step()
                
                total_loss+=loss.item()
                
            avg_loss = total_loss / len(dataloader)
            loss_his.append(avg_loss)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:03d}/{epochs} | Loss: {avg_loss:.4f}")
        return loss_his
