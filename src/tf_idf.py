import numpy as np
import math

class TFIDF_Model:
    def __init__(self,corpus):
        self.corpus=corpus
        self.vocab=self._build_vocab()
        self.num_docs=len(self.corpus)
        self.idf = self._compute_idf()
        self.tfidf_matrix = self._compute_tfidf_matrix()
    
    def _build_vocab(self):
        vocab=set()
        for sentence in self.corpus:
            for word in sentence:
                vocab.add(word)
        return list(vocab)
    
    def _compute_tf(self,doc):
        tf_dict={word: 0 for word in self.vocab}
        doc_len=len(doc)
        for word in doc:
            if word in tf_dict:
                tf_dict[word]+=1
            
        for word in tf_dict:
            tf_dict[word]/=doc_len
        return tf_dict
    
    def _compute_idf(self):
        idf_dict={}
        for word in self.vocab:
            doc_count=sum(1 for doc in self.corpus if word in doc)
            idf_dict[word]=math.log(self.num_docs/doc_count)
        return idf_dict
    
    def _compute_tfidf_matrix(self):
        matrix=[]
        for doc in self.corpus:
            tf_dict=self._compute_tf(doc)
            doc_vector=[tf_dict[word]*self.idf[word] for word in self.vocab]
            matrix.append(doc_vector)
        return matrix
        
    def transform_query(self,query_words):
        tf=self._compute_tf(query_words)
        query_vec=[tf[word]*self.idf.get(word,0) for word in self.vocab]
        return query_vec
    
    def cosine_similarity(self,vec1,vec2):
        dot_product=sum(a*b for a,b in zip(vec1,vec2))
        norm_a=math.sqrt(sum(a*a for a in vec1))
        norm_b=math.sqrt(sum(b*b for b in vec2))
        if norm_a==0 or norm_b==0: return 0.0
        return dot_product/(norm_a*norm_b)
    
    def search(self,query_words,top=3):
        query_vec=self.transform_query(query_words)
        
        scores=[]
        for i,doc_vec in enumerate(self.tfidf_matrix):
            score=self.cosine_similarity(query_vec,self.tfidf_matrix[i])
            scores.append((i,score))
            
        scores.sort(key=lambda x:x[1],reverse=True)
        return scores[:top]
    