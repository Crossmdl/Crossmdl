from transformers import DistilBertTokenizer, DistilBertModel
import torch

if __name__ == "__main__":
    from transformers import DistilBertTokenizer, DistilBertModel
    import torch
    
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    import pdb;pdb.set_trace()
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = model(**inputs)
    
