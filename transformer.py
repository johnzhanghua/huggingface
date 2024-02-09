from transformers import pipeline

def generator():
    generator = pipeline("text-generation", model="distilgpt2")
    out = generator(
        "In this course, we will teach you how to",
        max_length=30,
        num_return_sequences=2,
        truncation=True,
        pad_token_id=50256,
    )
    print(out)

def sentiment():
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    out = classifier(
        ["I'm good, thank you!",
         "I'm fine, thank you!",
         "No worries!",
         "I can't agree more",
         "I'm more than happier"
        ]
    )

    print(out)