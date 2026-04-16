import json
import random
import os

def extract_and_save_corpus(json_path,output_path, num_samples=100):
    try:
        with open(json_path,'r',encoding='utf-8') as f:
            nlp_data=json.load(f)
        all_description=[item['nlp_description'] 
                         for item in nlp_data if item.get('nlp_description')]
        random.seed(42)
        if len(all_description)>num_samples:
            toy_corpus=random.sample(all_description,num_samples)
        else:
            toy_corpus=all_description
        
        os.makedirs(os.path.dirname(output_path),exist_ok=True)
        with open(output_path,'w',encoding='utf-8') as f:
            f.write('\n'.join(toy_corpus))
        
    except Exception as e:
        print(f"Lỗi: {e}")
        
def load_corpus(file_path):
    corpus=[]
    try:
        with open(file_path,'r',encoding='utf-8') as f:
            for line in f:
                words=line.strip().split()
                if len(words)>0:
                    corpus.append(words)
        return corpus
            
    except Exception as e:
        print(f'Loi: {e}')
        
def build_and_save_vocab(corpus,output_path):
    try:
        word2id={}
        id2word={}
        vocab_size=0
        for sentence in corpus:
            for word in sentence:
                if word not in word2id:
                    word2id[word]=vocab_size
                    id2word[vocab_size]=word
                    vocab_size+=1
                    
        vocab_data={
            'word2id':word2id,
            'id2word':id2word
        }
        os.makedirs(os.path.dirname(output_path),exist_ok=True)
        with open(output_path,'w',encoding='utf-8') as f:
            json.dump(vocab_data,f,ensure_ascii=False,indent=4)
        return vocab_data
            
    except Exception as e:
        print(f'Loi: {e}')

def generate_training_data(corpus,word2id,window_size=2):
    training_data=[]
    for sentence in corpus:
        for i,word in enumerate(sentence):
            word_id=word2id[word]
            start=max(0,i-window_size)
            end=min(len(sentence)-1,i+window_size)
            for j in range(start,end+1):
                if j!=i:
                    context_id=word2id[sentence[j]]
                    training_data.append((word_id,context_id))
    return training_data

def run_data_pipeline():
    json_path='data/raw/nlp_data.json'
    input_path='data/processed/toy_corpus.txt'
    output_path='data/processed/vocab.json'
    
    extract_and_save_corpus(json_path,input_path)
    corpus=load_corpus(input_path)
    vocab_data=build_and_save_vocab(corpus,output_path)
    
    word2id=vocab_data['word2id']
    id2word=vocab_data['id2word']
    
    training_data=generate_training_data(corpus,word2id)
    
    print(f"Đã tạo ra {len(training_data)} cặp bài tập (Target, Context) cho AI")
        
    print("\n🔍 Xem thử 3 bài tập đầu tiên:")
    for i in range(3):
        t_id, c_id = training_data[i]
        print(f"   Bài {i+1}: '{id2word[t_id]}' (ID:{t_id}) ---> đoán ---> '{id2word[c_id]}' (ID:{c_id})")
    
        
if __name__=="__main__":
    run_data_pipeline()
    
    