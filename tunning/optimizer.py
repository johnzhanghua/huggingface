import torch
from transformers import pipeline,AdamW, AutoTokenizer, AutoModelForSequenceClassification

def optimizer():
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    sequence = ["I'm good, thank you!",
                "I'm fine, thank you!",
                "No worries!",
                "I can't be happier",
                "I'm more than happier"
                ]
    tokens = tokenizer.tokenize(sequence,padding=True, truncation=True, return_tensors="pt")
    # something new
    tokens["labels"] = torch.tensor([1,1,1,1,1])

    #  train by processing the data
    optimizer = AdamW(model.parameters())
    loss = model(**tokens).loss
    loss.backward()
    optimizer.step()




    #classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    #out = classifier(

    #)

    # print(out)