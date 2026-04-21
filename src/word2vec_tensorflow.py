import tensorflow as tf
import numpy as np

class Word2VecTF(tf.keras.Model):
    def __init__(self,vocab_size,embedding_dim=50):
        super().__init__()
        self.W1=tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim)
        self.W2=tf.keras.layers.Dense(units=vocab_size,use_bias=False)
        
    def call(self,target_word_id):
        h=self.W1(target_word_id)
        u=self.W2(h)
        return u
    
    def get_similar_words(self,word_id,id2word,top=3):
        W1_weights=self.W1.get_weights()[0]
        target_vector=W1_weights[word_id]
        
        W1_norm=tf.math.l2_normalize(W1_weights,axis=1)
        target_norm=tf.math.l2_normalize(target_vector,axis=0)
        
        cos_sim=tf.linalg.matvec(W1_norm,target_norm)
        
        scores,indices=tf.math.top_k(cos_sim,top+1)
        
        result=[]
        for i in range(1,top+1):
            score=scores[i].numpy()
            idx=indices[i].numpy()
            result.append((id2word[idx],score))
        return result
    
class Word2VecTFSequence(tf.keras.utils.Sequence):
    def __init__(self,target,context,batch_size):
        self.target=target
        self.context=context
        self.indices=np.arange(len(self.target))
        self.batch_size=batch_size
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.target) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_indices=self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_x=self.target[batch_indices]
        batch_y=self.context[batch_indices]
        return np.array(batch_x),np.array(batch_y)
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
