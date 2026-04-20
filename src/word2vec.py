import numpy as np 

class Word2Vec:
    def __init__(self, vocab_size, embedding_dim=50, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        
        self.W1 = np.random.uniform(-1, 1, (vocab_size, embedding_dim))
        self.W2 = np.random.uniform(-1, 1, (embedding_dim, vocab_size))
        
    def get_one_hot_vector(self, word_id):
        vec = np.zeros(self.vocab_size)
        vec[word_id] = 1
        return vec
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def forward_propagation(self, word_id):
        x = self.get_one_hot_vector(word_id)
        h = np.dot(x, self.W1)
        u = np.dot(h, self.W2)
        y_pred = self.softmax(u)
        return y_pred, h, u
    
    def backward_propagation(self, y_true, y_pre, h, x):
        dL_du = y_pre - y_true
        dL_dW2 = np.outer(h, dL_du) 
        dL_dh = np.dot(dL_du, self.W2.T)
        dL_dW1 = np.outer(x, dL_dh)
        return dL_dW1, dL_dW2
    
    def update_weights(self, dL_dW1, dL_dW2):
        self.W1 -= self.learning_rate * dL_dW1
        self.W2 -= self.learning_rate * dL_dW2
    
    def train_step(self, target_id, context_id):
        y_true = self.get_one_hot_vector(context_id)
        x = self.get_one_hot_vector(target_id)
        y_pre, h, u = self.forward_propagation(target_id)
        dL_dW1, dL_dW2 = self.backward_propagation(y_true, y_pre, h, x)
        self.update_weights(dL_dW1, dL_dW2)
        loss = -np.log(y_pre[context_id] + 1e-10) 
        return loss
    
    def train(self, training_data, epochs=10):
        print(f"Bắt đầu huấn luyện Word2Vec trong {epochs} Epochs...")
        loss_history = []
        for epoch in range(epochs):
            total_loss = 0
            for target_id, context_id in training_data:
                loss = self.train_step(target_id, context_id)
                total_loss += loss
            avg_loss = total_loss / len(training_data)
            loss_history.append(avg_loss)
            print(f"Epoch {epoch + 1:03d}/{epochs} | Trị giá sai số (Loss): {avg_loss:.4f}")
        return loss_history
            