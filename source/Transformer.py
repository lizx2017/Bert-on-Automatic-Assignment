import re
import pymysql
import pickle
import json
import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import DataCollatorWithPadding, create_optimizer
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datasets import load_dataset

def getData(project):
    db = pymysql.connect(user='root', password='137152', host='localhost', database='bugreportdb')
    cursor = db.cursor()
    # query = "SELECT issueID, Assignee, issue_summary, Description FROM issues AS TableA JOIN (SELECT Assignee, COUNT(*) AS Freq FROM issues WHERE Project='HBASE' GROUP BY Assignee HAVING Freq>=10 ORDER BY Freq DESC) AS TableB ON TableA.Project='HBASE' AND (TableA.Status='Closed' or TableA.Status='RESOLVED') AND TableA.Assignee = TableB.Assignee AND TableA.Assignee!='Unassigned';"
    query = "SELECT issueID, Assignee, issue_summary, Description FROM issues WHERE Project='"+project+"' AND (Status='Closed' or Status='RESOLVED') AND Assignee!='Unassigned' AND Self_assign!='True';"
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    db.close()

    Issues = dict()
    for result in results:
        Issues[result[0]] = dict()
        Issues[result[0]]['assignee'] = result[1]
        Issues[result[0]]['summary'] = result[2]
        Issues[result[0]]['description'] = result[3]
    return Issues

def dataPreprocessing(project):
    Issues = getData(project)
    IDs = list(Issues.keys())

    Assignees = []
    for id in IDs:
        Assignees.append(Issues[id]['assignee'])

    Summaries = []
    for id in IDs:
        summary = Issues[id]['summary'].lower()
        summary = re.sub(r'[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n]+', ' ', summary)
        temp = []
        for word in summary.split(' '):
            if not word.isdigit():
                temp.append(word)
        summary = ' '.join(temp)
        Summaries.append(summary)

    Descriptions = []
    for id in IDs:
        description = Issues[id]['description'].lower()
        # description = re.sub(r'[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\r]+', ' ', description)
        description = re.sub(r'[\t\n\r]+', ' ', description)
        # temp = []
        # for word in description.split(' '):
        #     if not word.isdigit():
        #         temp.append(word)
        # description = ' '.join(temp)
        Descriptions.append(description)
    CombinedTexts = [Summaries[i] + ' ' + Descriptions[i] for i in range(len(Summaries))]

    x_train, x_test, Y_train, Y_test = train_test_split(CombinedTexts, Assignees, test_size=0.2, random_state=42)
    y_labels = list(set(Assignees))
    labelEncoder = LabelEncoder()
    labelEncoder.fit(y_labels)
    num_labels = len(y_labels)
    y_train = labelEncoder.transform(Y_train)
    y_test = labelEncoder.transform(Y_test)

    jsonData = json.loads(json.dumps({"train":[], "test":[]}, indent=4))
    trainData = [{"text": x_train[i], "label": int(y_train[i])} for i in range(len(y_train))]
    trainJson = json.loads(json.dumps(trainData, indent=4))
    testData = [{"text": x_test[i], "label": int(y_test[i])} for i in range(len(y_test))]
    testJson = json.loads(json.dumps(testData, indent=4))
    jsonData['train'] = trainData
    jsonData['test'] = testData
    toJsonFile = json.dumps(jsonData, indent=4)
    f = open(project + ".json", "w")
    f.write(toJsonFile)
    f.close()

    f = open(project + "_train.json", "w")
    f.write(json.dumps(trainJson, indent=4))
    f.close()

    f = open(project + "_test.json", "w")
    f.write(json.dumps(testJson, indent=4))
    f.close()

    return num_labels

def finetuneModel(project):
    num_labels = dataPreprocessing(project)
    # data = load_dataset("json", data_files="D:/Python_Project/bugAssignment/" + project + ".json")
    train_data = load_dataset("json", data_files="./dataset/" + project + "_train.json")
    test_data = load_dataset("json", data_files="./dataset/" + project + "_test.json")
    num_labels = 1 + max(max(train_data['train']['label']), max(test_data['train']['label']))

    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = num_labels)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_train = train_data.map(preprocess_function, batched=True)
    tokenized_test = test_data.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    tf_test_set = tokenized_test["train"].to_tf_dataset(
        columns=["attention_mask", "input_ids", "label"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator
    )
    tf_train_set = tokenized_train["train"].to_tf_dataset(
        columns=["attention_mask", "input_ids", "label"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator
    )

    batch_size = 16
    num_epochs = 5
    batches_per_epoch = len(tokenized_train["train"]) # batch_size
    total_train_steps = int(batches_per_epoch * num_epochs)
    optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
    model.compile(optimizer=optimizer, metrics=['accuracy'])
    model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=num_epochs)
    model.save_pretrained(project.lower() + "-bert")
    tokenizer.save_pretrained(project.lower() + "-bert")
    return tokenizer, model, tf_test_set