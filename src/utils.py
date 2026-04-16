import json
import random
import os

def prepare_data_pipline(json_path,output_path, num_samples=100):
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
        
if __name__=="__main__":
    json_path='data/raw/nlp_data.json'
    output_path='data/processed/toy_corpus.txt'
    prepare_data_pipline(json_path,output_path)
    