import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

def trainer():
    # get token and model from checkpoint
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    # load dataset, convert to token
    def tokenize(input): # inner func
        return tokenizer(input["sentence1"], input["sentence2"], truncation=True)
    raw = load_dataset("glue", "mrpc")
    tokenized_datasets = raw.map(tokenize, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)




    # traning
    def compute_metrics(eval_preds):
        metric = evaluate.load("glue", "mrpc")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments("my-trainer")
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
