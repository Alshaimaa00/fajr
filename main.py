# import pandas as pd
# from sklearn.model_selection import train_test_split
# from datasets import Dataset
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import re

# def preprocess(text):
#     if not isinstance(text, str):
#         return ""

#     text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)  # إزالة التشكيل
#     text = re.sub(r'[أإآ]', 'ا', text)
#     text = re.sub(r'ى', 'ي', text)
#     text = re.sub(r'ـ+', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()

#     return text

# # 1) قراءة البيانات


# correct = pd.read_csv("quran_tafseers.csv")
# incorrect = pd.read_csv("tafsir_incorrect_final.csv")
# real = pd.read_csv("real_data.csv")

# # 2) إضافة الليبل


# correct["label_id"] = 0
# incorrect["label_id"] = 1


# # 3) توحيد البيانات


# df = pd.concat([correct, incorrect, real])
# print(df["label_id"].value_counts())


# # 4) اختيار النص الصحيح

# df["ayah"] = df["ayah"].apply(preprocess)
# df["tafsir"] = df["tafsir"].apply(preprocess)

# df["text"] = df["ayah"] + " [SEP] " + df["tafsir"]

# # 5) تنظيف


# df = df.dropna(subset=["text", "label_id"])


# # 6) تقسيم البيانات
# train_df, test_df = train_test_split(
#     df,
#     test_size=0.2,
#     stratify=df["label_id"],
#     random_state=42
# )


# # 7) تحويل إلى Dataset


# train_dataset = Dataset.from_pandas(train_df)
# test_dataset = Dataset.from_pandas(test_df)


# # 8) tokenizer


# tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix")
# def tokenize(example):
#     return tokenizer(
#         example["text"],
#         padding="max_length",
#         truncation=True,
#         max_length=128
#     )

# train_dataset = train_dataset.map(tokenize)
# test_dataset = test_dataset.map(tokenize)


# # 9) تجهيز الليبل


# train_dataset = train_dataset.rename_column("label_id", "labels")
# test_dataset = test_dataset.rename_column("label_id", "labels")


# # 10) تحميل المودل


# model = AutoModelForSequenceClassification.from_pretrained(
#     "aubmindlab/bert-base-arabertv02",
#     num_labels=2
# )

# # 11) إعداد التدريب


# training_args = TrainingArguments(
#     output_dir="./results",
#     per_device_train_batch_size=8,
#     num_train_epochs=3,
#     logging_steps=50,
#     save_strategy="epoch",
#     learning_rate=2e-5
# )


# # 12) metrics


# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = logits.argmax(axis=1)
#     return {
#         "accuracy": accuracy_score(labels, preds)
#     }


# # 13) Trainer


# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     compute_metrics=compute_metrics
# )


# # 14) التدريب


# trainer.train()


# # 15) حفظ المودل


# model.save_pretrained("my_model")
# tokenizer.save_pretrained("my_model")


# # 16) التقييم


# results = trainer.evaluate()
# print("\nEvaluation:", results)

# predictions = trainer.predict(test_dataset)

# y_pred = predictions.predictions.argmax(axis=1)
# y_true = predictions.label_ids
# print(y_pred[:20])

# print("\nAccuracy:", accuracy_score(y_true, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_true, y_pred))
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_true, y_pred))

import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from datasets import Dataset, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# =========================
# 1) تنظيف النص
# =========================
def preprocess(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ـ+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# =========================
# 2) قراءة الملفات
# =========================
correct = pd.read_csv("quran_tafseers.csv")
incorrect = pd.read_csv("tafsir_incorrect_final.csv")
real = pd.read_csv("real_data.csv")


# =========================
# 3) إضافة الليبلات
# =========================
correct["label_id"] = 0
incorrect["label_id"] = 1

# real_data لازم يكون فيه label_id جاهز


# =========================
# 4) دمج البيانات
# =========================
df = pd.concat([correct, incorrect, real], ignore_index=True)


# =========================
# 5) تنظيف الليبل
# مهم جدًا عشان ما يطلع 0.0 / 1.0
# =========================
df["label_id"] = pd.to_numeric(df["label_id"], errors="coerce")
df = df.dropna(subset=["label_id"])
df["label_id"] = df["label_id"].astype(int)
df = df[df["label_id"].isin([0, 1])]


# =========================
# 6) تنظيف النصوص
# =========================
df["ayah"] = df["ayah"].apply(preprocess)
df["tafsir"] = df["tafsir"].apply(preprocess)

df["text"] = df["ayah"] + " [SEP] " + df["tafsir"]

df = df.dropna(subset=["text", "label_id"])
df = df.drop_duplicates(subset=["text"])

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Label distribution:")
print(df["label_id"].value_counts())


# =========================
# 7) تقسيم Train / Test
# =========================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label_id"],
    random_state=42
)

print("Train size:", len(train_df))
print("Test size:", len(test_df))


# =========================
# 8) تحويل إلى Dataset
# =========================
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


# =========================
# 9) AraBERT tokenizer
# =========================
model_name = "aubmindlab/bert-base-arabertv02"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize)
test_dataset = test_dataset.map(tokenize)


# =========================
# 10) تجهيز labels
# =========================
train_dataset = train_dataset.rename_column("label_id", "labels")
test_dataset = test_dataset.rename_column("label_id", "labels")

train_dataset = train_dataset.cast_column("labels", Value("int64"))
test_dataset = test_dataset.cast_column("labels", Value("int64"))

# نخلي بس الأعمدة اللي يحتاجها المودل
keep_cols = ["input_ids", "attention_mask", "labels"]

train_dataset = train_dataset.remove_columns(
    [col for col in train_dataset.column_names if col not in keep_cols]
)

test_dataset = test_dataset.remove_columns(
    [col for col in test_dataset.column_names if col not in keep_cols]
)

train_dataset.set_format("torch")
test_dataset.set_format("torch")


# =========================
# 11) تحميل AraBERT model
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)


# =========================
# 12) إعداد التدريب
# =========================
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)


# =========================
# 13) Metrics
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    return {
        "accuracy": accuracy_score(labels, preds)
    }


# =========================
# 14) Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


# =========================
# 15) التدريب
# =========================
trainer.train()


# =========================
# 16) حفظ المودل
# =========================
model.save_pretrained("my_model")
tokenizer.save_pretrained("my_model")


# =========================
# 17) التقييم
# =========================
results = trainer.evaluate()
print("\nEvaluation:")
print(results)

predictions = trainer.predict(test_dataset)

y_pred = predictions.predictions.argmax(axis=1)
y_true = predictions.label_ids

print("\nFirst predictions:")
print(y_pred[:20])

print("\nAccuracy:")
print(accuracy_score(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

