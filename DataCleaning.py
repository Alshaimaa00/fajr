import pandas as pd
import re

# دالة تنظيف النص العربي
def normalize_arabic(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # إزالة التشكيل
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)

    # توحيد الألف
    text = re.sub(r'[أإآ]', 'ا', text)

    # توحيد الياء
    text = re.sub(r'ى', 'ي', text)

    # إزالة التطويل
    text = re.sub(r'ـ+', '', text)

    # حذف المسافات الزائدة
    text = re.sub(r'\s+', ' ', text).strip()

    return text


print("Loading data...")

df = pd.read_csv("quran_tafseers.csv")

# تنظيف الأعمدة
df["ayah_clean"] = df["ayah"].apply(normalize_arabic)
df["tafsir_clean"] = df["tafsir"].apply(normalize_arabic)

# حفظ نسخة جديدة
df.to_csv("quran_tafseers_cleaned.csv", index=False, encoding="utf-8-sig")

print("Cleaning done!")

print(df[["ayah", "ayah_clean"]].head())