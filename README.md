# 🤖 AI vs Real Image Classifier

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

**نظام ذكي للتمييز بين الصور الحقيقية والصور المولّدة بالذكاء الاصطناعي**

[English](#english) | [العربية](#العربية)

</div>

---

## العربية

### 📋 نظرة عامة

هذا المشروع عبارة عن نظام تصنيف متقدم يستخدم نموذجًا مدربًا مسبقًا من Hugging Face للتمييز بكفاءة عالية بين:
- **الصور الحقيقية** (Real Images)
- **الصور المولّدة بالذكاء الاصطناعي** (AI Generated Images)

يستخدم المشروع نموذج **`jacoballessio/ai-image-detect-distilled`** الذي يتميز بدقة عالية وسرعة في المعالجة.

### ⭐ مميزات النموذج المستخدم
- **المصدر:** Hugging Face Hub
- **اسم الموديل:** `jacoballessio/ai-image-detect-distilled`
- **الحجم:** ~50MB (خفيف وسريع)
- **الدقة:** عالية مقارنة بالنماذج التقليدية
- **التقنية:** يعتمد على تقنيات Vision Transformers (ViT) المقطرة (Distilled).

### 🚀 التثبيت والتشغيل

#### المتطلبات الأساسية
- Python 3.8 أو أحدث
- اتصال بالإنترنت (لتحميل الموديل في المرة الأولى)

#### خطوات التثبيت

```bash
# 1. استنساخ المشروع (أو تحميل الملفات)
# (تأكد من وجودك داخل مجلد المشروع)

# 2. تثبيت المتطلبات
pip install -r requirements.txt

# 3. تشغيل التطبيق
python app.py
```

*ملاحظة: عند التشغيل لأول مرة، سيقوم التطبيق تلقائيًا بتحميل ملفات الموديل من Hugging Face.*

#### استخدام سطر الأوامر (CLI)
يمكنك استخدام أداة التنبؤ مباشرة دون واجهة الويب:

```bash
# فحص صورة واحدة
python predict.py path/to/image.jpg

# فحص مع تفعيل تحسين الدقة (Test Time Augmentation)
python predict.py path/to/image.jpg --tta
```

#### الوصول للتطبيق
افتح المتصفح وانتقل إلى: `http://localhost:5003`

### 📁 هيكل المشروع

```
AI_Project/
├── app.py                 # تطبيق Flask وتوجيه المسارات
├── predict.py             # كود التعامل مع موديل Hugging Face
├── train_model.py         # (اختياري) كود لتدريب موديل خاص
├── requirements.txt       # المكتبات المطلوبة
├── templates/             # واجهات المستخدم (HTML)
├── model_cache/           # مكان تخزين الموديل المحمل محليًا
└── uploads/               # مجلد مؤقت للصور المرفوعة
```

### 🛠️ الميزات

- ✅ **دقة عالية:** اعتمادًا على أحدث نماذج Hugging Face.
- ✅ **سهولة الاستخدام:** واجهة ويب بسيطة وجذابة.
- ✅ **فحص متعدد:** إمكانية رفع وتحليل مجموعة صور دفعة واحدة.
- ✅ **تحليل دقيق:** عرض نسبة الاحتمالية لكون الصورة حقيقية أو مولدة.
- ✅ **TTA:** دعم تقنية Test Time Augmentation لزيادة الموثوقية.

---

## English

### 📋 Overview

This project is an advanced image classification system leveraging a pre-trained Hugging Face model to accurately distinguish between:
- **Real Images**
- **AI Generated Images**

The project utilizes the **`jacoballessio/ai-image-detect-distilled`** model, known for its high accuracy and efficiency.

### ⭐ Model Features
- **Source:** Hugging Face Hub
- **Model Name:** `jacoballessio/ai-image-detect-distilled`
- **Size:** ~50MB (Lightweight & Fast)
- **Accuracy:** High performance on various datasets.
- **Technology:** Based on Distilled Vision Transformers (ViT).

### 🚀 Installation & Usage

#### Prerequisites
- Python 3.8 or newer
- Internet connection (to download the model on first run)

#### Installation Steps

```bash
# 1. Navigate to project directory

# 2. Install requirements
pip install -r requirements.txt

# 3. Run the application
python app.py
```

*Note: On the first run, the application will automatically download the necessary model files from Hugging Face.*

#### CLI Usage
You can use the prediction tool directly from the command line:

```bash
# Predict a single image
python predict.py path/to/image.jpg

# Predict with Test Time Augmentation (TTA)
python predict.py path/to/image.jpg --tta
```

#### Access the Application
Open your browser and navigate to: `http://localhost:5003`

### 📁 Project Structure

```
AI_Project/
├── app.py                 # Main Flask application
├── predict.py             # Hugging Face model integration
├── train_model.py         # (Optional) Custom training script
├── requirements.txt       # Dependencies
├── templates/             # HTML Templates
├── model_cache/           # Local cache for the downloaded model
└── uploads/               # Temporary folder for uploads
```

### 🛠️ Features

- ✅ **High Accuracy:** Powered by state-of-the-art Hugging Face models.
- ✅ **User Friendly:** Simple and clean web interface.
- ✅ **Batch Processing:** Upload and analyze multiple images at once.
- ✅ **Detailed Analysis:** Displays probability scores for Real vs AI.
- ✅ **TTA Support:** Test Time Augmentation for improved reliability.

---

## 📝 License

This project is licensed under the MIT License.

## 👤 Author

Eng-Azhar

---

<div align="center">

**⭐ Don't forget to star the repo if you like it! | إذا أعجبك المشروع، لا تنسَ إضافة نجمة! ⭐**

</div>
