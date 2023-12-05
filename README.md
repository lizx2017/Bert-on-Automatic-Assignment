# Bert-on-Automatic-Assignment
We make attempt to conduct Bert on Automatic Bug Assignment. In our experiment, Automatic Bug assignment problem is treated as a classification problem, and descriptions of reports are inputted to the classifier, Bert. Although Bert is widely regarded as an advanced technique, results seem disappointed. Maybe the model requires delicate reform for Automatic Bug Assignment. We record our present code and datasets in this repository. Besides, we also make guesses on the disappointed results.

# Fine-tune
We reload Bert by Python package ``transformers'' with:

    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = num_labels)

Next, we fine-tune Bert with our datasets:

    optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
    model.compile(optimizer=optimizer, metrics=['accuracy'])
    model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=num_epochs)

Due to memory limit, we train on batches:

    num_epochs = 5
    batches_per_epoch = len(tokenized_train["train"]) # batch_size
    total_train_steps = int(batches_per_epoch * num_epochs)

The whole source code can be found in [source/Transformer.py](./source/Transformer.py)

# Results
| Project | SHIRO | PDFBOX | LUCENE | HBASE | CASSANDRA |
| :-: | :-: | :-: | :-: | :-: | :-: |
| Train | 63.33% | 66.25% | 57.40% | 33.82% | 23.25% |
| Test | 43.48% | 44.27% | 48.61% | 23.17% | 17.04% |

In our training process, accuracies get continuous improvements (losses continuous decrease) until losses in test increase abnormal. But the strange thing is that the accuracies in test neither increase nor decrease. In other words, the performance of Bert model on test datasets remains unchanged, although the model has been fine-tuned by our dataset.

# Guesses
If there is no mistakes in source code, that is to say Bert cannot work well on Automatic Bug Assignment, we guess that the Automatic Bug Assignment problem is a difficult task which resulres more complex information inputted. For example, one confusing thing is that two similar pieces of descriptions are assigned to A and B in training and testing separtely. We guess that the circumestance exist since these two developers work for the same department in succession. Since Bert holds great performance on analysing texts, it must decide to assign the description to A in high possible as learned from training datasets, while it has been assigned to B in actual. If it is the case, it is insufficient to update the NLP techniques for further improvement. More information, such as the exmployment period of developers, is required while they are unavailable from the public datasets and the open-source projects.
