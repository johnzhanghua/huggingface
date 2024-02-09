import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def tokenizor():
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    sequence = "Using a Transformer network is simple"
    tokens = tokenizer.tokenize(sequence)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    input_ids = torch.tensor([ids])
    print("Input ids:", input_ids)

    output = model(input_ids)
    print("logits:",  output)

    decoded_string = tokenizer.decode(ids)

    print(decoded_string)
