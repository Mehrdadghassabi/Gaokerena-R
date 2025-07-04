# مستندات کامل Phase 2.ipynb: تحلیل احساسات با CNN، LSTM و Hybrid

## فهرست مطالب
1. [وارد کردن کتابخانه‌ها و بارگذاری داده‌ها](#cell-1)
2. [نمونه‌برداری و پیش‌پردازش داده‌ها](#cell-2)
3. [ایجاد نمایش Bag of Words](#cell-3)
4. [خوشه‌بندی K-means](#cell-4)
5. [نمونه‌برداری یکنواخت و Word2Vec](#cell-5)
6. [آماده‌سازی داده‌ها برای PyTorch](#cell-6)
7. [مدل CNN یک‌بعدی](#cell-7)
8. [مدل LSTM](#cell-8)
9. [مدل ترکیبی CNN-LSTM](#cell-9)
10. [توابع آموزش](#cell-10)
11. [آموزش همه مدل‌ها](#cell-11)
12. [مقایسه نتایج و تجسم](#cell-12)

---

## Cell 1: Import libraries and load data {#cell-1}

### توضیح خط به خط:

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
```
**توضیح:** وارد کردن کتابخانه‌های اصلی برای:
- `pandas`, `numpy`: پردازش داده
- `torch`: فریمورک یادگیری عمیق
- `nn`, `optim`, `F`: لایه‌ها، بهینه‌سازها و توابع فعال‌سازی
- `Dataset`, `DataLoader`: مدیریت داده‌ها
- `ReduceLROnPlateau`, `CosineAnnealingLR`: زمان‌بندی نرخ یادگیری

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
```
**توضیح:** ابزارهای scikit-learn برای:
- `CountVectorizer`: ایجاد Bag of Words
- `KMeans`: خوشه‌بندی
- `PCA`: کاهش ابعاد برای تجسم
- `train_test_split`: تقسیم داده‌ها
- ابزارهای ارزیابی

```python
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict
import gensim.downloader as api
```
**توضیح:** کتابخانه‌های کمکی:
- `matplotlib`, `seaborn`: تجسم داده‌ها
- `re`: عبارات منظم
- `nltk`: پردازش زبان طبیعی
- `gensim`: بارگذاری Word2Vec

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```
**توضیح:** انتخاب دستگاه محاسباتی (GPU یا CPU)

```python
train_data = pd.read_csv('./dataset/imdb_train.csv')
test_data = pd.read_csv('./dataset/imdb_test.csv')
```
**توضیح:** بارگذاری داده‌های IMDB برای تحلیل احساسات

---

## Cell 2: Data sampling and preprocessing {#cell-2}

```python
combined_data = pd.concat([train_data, test_data], ignore_index=True)
sampled_data = combined_data
```
**توضیح:** ترکیب داده‌های train و test (در اینجا نمونه‌برداری انجام نشده)

```python
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove special characters and digits, keep only alphabets
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
```
**توضیح:** مراحل پیش‌پردازش اولیه:
- تبدیل به حروف کوچک
- حذف تگ‌های HTML
- حذف کاراکترهای خاص و اعداد
- حذف فاصله‌های اضافی

```python
    # Tokenize
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    # Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
```
**توضیح:** 
- توکن‌سازی متن
- حذف stop words (کلمات بی‌معنی)
- فیلتر کردن کلمات کوتاه‌تر از 3 حرف

```python
word_counts = Counter(all_words)
words_to_keep = {word for word, count in word_counts.items() if count > 1}

def remove_rare_words(text):
    words = text.split()
    return ' '.join([word for word in words if word in words_to_keep])
```
**توضیح:** حذف کلماتی که فقط یک بار ظاهر شده‌اند (کاهش نویز)

---

## Cell 3: Bag of Words representation {#cell-3}

```python
vectorizer = CountVectorizer(max_features=10_000, binary=True)
bow_matrix = vectorizer.fit_transform(sampled_data['processed_review'])
feature_names = vectorizer.get_feature_names_out()
```
**توضیح:**
- `max_features=10_000`: حداکثر 10,000 کلمه پرتکرار
- `binary=True`: استفاده از 0 و 1 به جای تعداد تکرار
- ایجاد ماتریس BOW برای خوشه‌بندی

```python
bow_dense = bow_matrix.toarray()
```
**توضیح:** تبدیل ماتریس sparse به dense برای الگوریتم‌های خوشه‌بندی

---

## Cell 4: K-means clustering {#cell-4}

```python
k = 7
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(bow_dense)
```
**توضیح:**
- `n_clusters=7`: تعداد خوشه‌ها
- `random_state=42`: تضمین تکرارپذیری
- `n_init=10`: 10 بار اجرای مختلف برای بهترین نتیجه

```python
pca = PCA(n_components=2, random_state=42)
bow_2d = pca.fit_transform(bow_dense)
```
**توضیح:** کاهش ابعاد به 2 بعد برای تجسم خوشه‌ها

```python
def find_optimal_clusters(data, max_k=15):
    inertias = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
```
**توضیح:** الگوریتم Elbow Method برای یافتن بهترین تعداد خوشه

---

## Cell 5: Uniform sampling from clusters and Word2Vec {#cell-5}

```python
def uniform_sampling_from_clusters(data, cluster_labels, samples_per_cluster):
    sampled_indices = []
    
    for cluster_id in range(len(np.unique(cluster_labels))):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        if len(cluster_indices) >= samples_per_cluster:
            sampled = np.random.choice(cluster_indices, samples_per_cluster, replace=False)
        else:
            sampled = cluster_indices
            
        sampled_indices.extend(sampled)
```
**توضیح:** نمونه‌برداری یکنواخت از هر خوشه:
- `samples_per_cluster = 400`: 400 نمونه از هر خوشه
- اگر خوشه کمتر از 400 نمونه داشت، همه را انتخاب می‌کند

```python
try:
    wv = api.load('word2vec-google-news-300')
    print("Word2Vec model loaded successfully!")
    word2vec_available = True
except Exception as e:
    print(f"Word2Vec model not available: {e}")
    wv = None
    word2vec_available = False
```
**توضیح:** بارگذاری مدل Word2Vec از Google News با 300 بعد

---

## Cell 6: Data preparation for PyTorch {#cell-6}

```python
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab_to_idx, max_length=200):
        self.texts = texts
        self.labels = labels
        self.vocab_to_idx = vocab_to_idx
        self.max_length = max_length
```
**توضیح:** کلاس Dataset سفارشی:
- `max_length=200`: حداکثر طول متن

```python
def __getitem__(self, idx):
    text = self.texts[idx]
    label = self.labels[idx]
    
    # Convert text to indices
    tokens = text.split()
    indices = [self.vocab_to_idx.get(token, self.vocab_to_idx['<UNK>']) for token in tokens]
    
    # Pad or truncate
    if len(indices) > self.max_length:
        indices = indices[:self.max_length]
    else:
        indices.extend([self.vocab_to_idx['<PAD>']] * (self.max_length - len(indices)))
```
**توضیح:** 
- تبدیل کلمات به شاخص‌ها
- `<UNK>`: کلمات ناشناخته
- `<PAD>`: padding برای یکسان کردن طول

```python
def create_vocabulary_with_word2vec(texts, word2vec_model=None, min_freq=2):
    word_freq = Counter()
    for text in texts:
        word_freq.update(text.split())
    
    vocab_to_idx = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
```
**توضیح:** ایجاد vocabulary:
- `min_freq=2`: کلمات با حداقل 2 تکرار
- اولویت‌دهی به کلمات موجود در Word2Vec

```python
def create_word2vec_embedding_matrix(vocab_to_idx, word2vec_model, embedding_dim=300):
    vocab_size = len(vocab_to_idx)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    words_found = 0
    for word, idx in vocab_to_idx.items():
        if word in ['<PAD>', '<UNK>']:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
        else:
            try:
                embedding_matrix[idx] = word2vec_model[word]
                words_found += 1
            except KeyError:
                embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
```
**توضیح:** ایجاد ماتریس embedding:
- استفاده از Word2Vec برای کلمات موجود
- مقداردهی تصادفی برای کلمات ناشناخته

---

## Cell 7: 1D CNN Model {#cell-7}

### ساختار مدل CNN:

```python
class CNN1D(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, embedding_dim=100, num_filters=100, 
                 filter_sizes=[2, 3, 4, 5], dropout=0.5):
```
**پارامترهای ورودی:**
- `vocab_size`: اندازه vocabulary
- `embedding_matrix`: ماتریس Word2Vec
- `embedding_dim=100`: بعد embedding (300 برای Word2Vec)
- `num_filters=100`: تعداد فیلترها برای هر اندازه
- `filter_sizes=[2, 3, 4, 5]`: اندازه‌های مختلف فیلتر
- `dropout=0.5`: نرخ dropout

```python
# Initialize embedding layer
self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

# Load pre-trained embeddings if available
if embedding_matrix is not None:
    print("Loading pre-trained Word2Vec embeddings...")
    self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
```
**توضیح لایه Embedding:**
- `padding_idx=0`: شاخص padding
- بارگذاری Word2Vec از پیش آموزش‌دیده

```python
# Multiple CNN filters with different sizes
self.convs = nn.ModuleList([
    nn.Conv1d(embedding_dim, num_filters, kernel_size=size, padding=size//2)
    for size in filter_sizes
])

# Batch normalization for each conv layer
self.conv_bns = nn.ModuleList([
    nn.BatchNorm1d(num_filters) for _ in filter_sizes
])
```
**توضیح لایه‌های CNN:**
- `Conv1d`: کانولوشن یک‌بعدی برای متن
- `kernel_size`: اندازه فیلتر (2، 3، 4، 5 کلمه)
- `BatchNorm1d`: تسریع همگرایی و پایداری

```python
# Enhanced classifier with multiple layers
total_filters = len(filter_sizes) * num_filters
self.dropout1 = nn.Dropout(dropout)
self.fc1 = nn.Linear(total_filters, total_filters // 2)
self.bn1 = nn.BatchNorm1d(total_filters // 2)
self.dropout2 = nn.Dropout(dropout)
self.fc2 = nn.Linear(total_filters // 2, 64)
self.fc3 = nn.Linear(64, 1)
```
**توضیح Classifier:**
- `total_filters = 4 × 100 = 400`: ترکیب همه فیلترها
- سه لایه کاملاً متصل با کاهش تدریجی ابعاد
- Batch normalization و dropout برای جلوگیری از overfitting

```python
def forward(self, x):
    # Embedding with dropout
    x = self.embedding(x)
    x = self.embedding_dropout(x)
    x = x.transpose(1, 2)
    
    # Multiple CNN filters with batch norm
    conv_outputs = []
    for conv, bn in zip(self.convs, self.conv_bns):
        conv_out = F.relu(bn(conv(x)))
        pooled = F.adaptive_max_pool1d(conv_out, 1)
        conv_outputs.append(pooled.squeeze(2))
    
    # Concatenate all conv outputs
    x = torch.cat(conv_outputs, dim=1)
```
**توضیح Forward Pass:**
1. Embedding + dropout
2. تبدیل ابعاد برای Conv1d
3. اعمال فیلترهای مختلف
4. Max pooling برای استخراج ویژگی مهم
5. ترکیب خروجی‌ها

---

## Cell 8: LSTM Model {#cell-8}

### ساختار مدل LSTM:

```python
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, embedding_dim=128, hidden_dim=128, 
                 num_layers=2, dropout=0.5):
```
**پارامترهای LSTM:**
- `hidden_dim=128`: تعداد نورون‌های مخفی
- `num_layers=2`: تعداد لایه‌های LSTM
- `bidirectional=True`: LSTM دوطرفه

```python
# Enhanced LSTM with layer normalization
self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                   batch_first=True, dropout=dropout if num_layers > 1 else 0, 
                   bidirectional=True)
```
**دلایل انتخاب:**
- **Bidirectional**: دیدن کل متن (گذشته و آینده)
- **Multi-layer**: پیچیدگی بیشتر برای یادگیری الگوهای پیچیده
- **Dropout**: فقط بین لایه‌ها (نه داخل لایه)

```python
# Attention mechanism
self.attention = nn.Linear(hidden_dim * 2, 1)
```
**مکانیزم Attention:**
- `hidden_dim * 2`: بخاطر bidirectional بودن
- توجه انتخابی به بخش‌های مختلف متن

```python
def forward(self, x):
    # LSTM with attention
    lstm_out, (hidden, cell) = self.lstm(x)
    
    # Simple attention mechanism
    attention_weights = F.softmax(self.attention(lstm_out), dim=1)
    attended_output = torch.sum(attention_weights * lstm_out, dim=1)
```
**Attention Forward:**
1. محاسبه وزن‌های attention
2. میانگین وزن‌دار خروجی‌های LSTM
3. تمرکز بر بخش‌های مهم متن

---

## Cell 9: Hybrid CNN-LSTM Model {#cell-9}

### ساختار مدل ترکیبی:

```python
class HybridCNNLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, embedding_dim=128, num_filters=100, 
                 filter_sizes=[2, 3, 4, 5], hidden_dim=128, num_layers=2, dropout=0.5):
```

**دلیل ترکیب CNN + LSTM:**
- **CNN**: استخراج ویژگی‌های محلی (الگوهای n-gram)
- **LSTM**: درک روابط دوربرد و ترتیب

```python
# Multiple CNN filters with different sizes - FIXED padding
self.convs = nn.ModuleList([
    nn.Conv1d(embedding_dim, num_filters, kernel_size=size, padding='same')
    for size in filter_sizes
])
```
**تغییر کلیدی:** `padding='same'` برای حفظ طول ورودی

```python
# Enhanced LSTM
total_filters = len(filter_sizes) * num_filters
self.lstm = nn.LSTM(
    total_filters, 
    hidden_dim, 
    num_layers=num_layers,
    batch_first=True, 
    dropout=dropout if num_layers > 1 else 0,
    bidirectional=True
)
```
**جریان داده:**
1. Embedding → CNN (استخراج ویژگی)
2. CNN output → LSTM (درک ترتیب)
3. LSTM → Attention → Classification

```python
def forward(self, x):
    # Enhanced embedding
    x = self.embedding(x)
    x = self.embedding_dropout(x)
    x = x.transpose(1, 2)
    
    # Multiple CNN filters with batch normalization - FIXED
    conv_outputs = []
    for conv, bn in zip(self.convs, self.conv_bns):
        conv_out = F.relu(bn(conv(x)))
        # Use adaptive pooling to ensure consistent output size
        pooled = F.adaptive_avg_pool1d(conv_out, x.size(2))
        conv_outputs.append(pooled)
    
    # Concatenate all conv outputs - now they have the same size
    x = torch.cat(conv_outputs, dim=1)
    x = x.transpose(1, 2)
    
    # LSTM with attention
    lstm_out, (hidden, cell) = self.lstm(x)
```

---

## Cell 10: Training functions {#cell-10}

```python
def train_model(model, train_loader, test_loader, optimizer, criterion, 
                scheduler=None, num_epochs=25, patience=7, model_name="Model"):
```

**پارامترهای آموزش:**
- `num_epochs=25`: حداکثر تعداد epoch
- `patience=7`: صبر برای early stopping

```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
**Gradient Clipping:** جلوگیری از exploding gradients

```python
# Early stopping
if val_accuracy > best_val_acc:
    best_val_acc = val_accuracy
    patience_counter = 0
    best_model_state = model.state_dict().copy()
else:
    patience_counter += 1
    
if patience_counter >= patience:
    print(f"Early stopping at epoch {epoch+1} (best val acc: {best_val_acc:.4f})")
    break
```
**Early Stopping:** جلوگیری از overfitting

---

## Cell 11: Train all models {#cell-11}

### تنظیمات Optimizer:

```python
# Define optimizers with weight decay
cnn_optimizer = optim.AdamW(cnn_model.parameters(), lr=0.001, weight_decay=1e-4)
lstm_optimizer = optim.AdamW(lstm_model.parameters(), lr=0.001, weight_decay=1e-4)
hybrid_optimizer = optim.AdamW(hybrid_model.parameters(), lr=0.001, weight_decay=1e-4)
```

**توضیح Hyperparameters:**

#### AdamW Optimizer:
- **`lr=0.001`**: نرخ یادگیری
  - **دلیل انتخاب**: متعادل بین سرعت و پایداری
  - کم‌تر: همگرایی کندتر اما پایدارتر
  - بیش‌تر: سرعت بالا اما ناپایداری

- **`weight_decay=1e-4`**: وزن تنظیم‌ساز L2
  - **دلیل**: جلوگیری از overfitting
  - **مقدار**: کوچک تا تأثیر منفی بر یادگیری نداشته باشد

#### مزایای AdamW نسبت به Adam:
- تفکیک weight decay از gradient
- بهبود تعمیم‌پذیری
- همگرایی بهتر

```python
# Define learning rate schedulers
cnn_scheduler = ReduceLROnPlateau(cnn_optimizer, mode='min', factor=0.5, patience=3, verbose=True)
lstm_scheduler = ReduceLROnPlateau(lstm_optimizer, mode='min', factor=0.5, patience=3, verbose=True)
hybrid_scheduler = ReduceLROnPlateau(hybrid_optimizer, mode='min', factor=0.5, patience=3, verbose=True)
```

**ReduceLROnPlateau Parameters:**
- **`mode='min'`**: کاهش LR وقتی loss کاهش نیابد
- **`factor=0.5`**: نصف کردن نرخ یادگیری
- **`patience=3`**: صبر 3 epoch قبل از کاهش
- **`verbose=True`**: نمایش پیام‌ها

**دلیل انتخاب:**
- تطبیق خودکار نرخ یادگیری
- جلوگیری از گیر کردن در local minima
- بهبود همگرایی در مراحل آخر

```python
# Define loss functions
cnn_criterion = nn.BCELoss()
lstm_criterion = nn.BCELoss()
hybrid_criterion = nn.BCELoss()
```

**Binary Cross Entropy Loss:**
- مناسب برای طبقه‌بندی دودویی
- خروجی sigmoid + BCE
- فرمول: `-[y*log(p) + (1-y)*log(1-p)]`

---

## Cell 12: Results comparison and visualization {#cell-12}

### نتایج نهایی مدل‌ها:

```python
print("FINAL MODEL PERFORMANCE COMPARISON")
print(f"CNN Model Accuracy:        {cnn_accuracy:.4f}")
print(f"LSTM Model Accuracy:       {lstm_accuracy:.4f}")
print(f"Hybrid Model Accuracy:     {hybrid_accuracy:.4f}")
```

## نتایج عملکرد مدل‌ها

### دقت مدل‌ها (بر اساس کد):

| مدل | دقت تست | توضیحات |
|-----|----------|----------|
| **CNN** | `{cnn_accuracy:.4f}` | سریع، ویژگی‌های محلی |
| **LSTM** | `{lstm_accuracy:.4f}` | روابط دوربرد، attention |
| **Hybrid** | `{hybrid_accuracy:.4f}` | ترکیب CNN + LSTM |

### تحلیل نتایج:

#### مزایای هر مدل:

**CNN Model:**
- ✅ سرعت آموزش بالا
- ✅ استخراج الگوهای محلی (n-grams)
- ✅ پارامترهای کمتر
- ❌ عدم درک روابط دوربرد

**LSTM Model:**
- ✅ درک روابط دوربرد
- ✅ مکانیزم attention
- ✅ bidirectional processing
- ❌ آموزش کندتر
- ❌ مستعد vanishing gradient

**Hybrid Model:**
- ✅ ترکیب مزایای CNN و LSTM
- ✅ استخراج ویژگی محلی + روابط دوربرد
- ✅ عملکرد متعادل
- ❌ پیچیدگی بالا
- ❌ پارامترهای بیشتر

### Classification Reports:

```python
print("Detailed Classification Reports:")
print("\nCNN Model:")
print(classification_report(cnn_targets, cnn_preds, labels=[0, 1], target_names=['Negative', 'Positive']))

print("\nLSTM Model:")
print(classification_report(lstm_targets, lstm_preds, labels=[0, 1], target_names=['Negative', 'Positive']))

print("\nHybrid Model:")
print(classification_report(hybrid_targets, hybrid_preds, labels=[0, 1], target_names=['Negative', 'Positive']))
```

**معیارهای ارزیابی:**
- **Precision**: دقت پیش‌بینی مثبت
- **Recall**: پوشش کلاس‌ها
- **F1-Score**: میانگین هارمونیک precision و recall
- **Support**: تعداد نمونه‌های هر کلاس

## خلاصه و نتیجه‌گیری

### بهترین تنظیمات:

1. **Embedding**: Word2Vec 300-dimensional
2. **Optimizer**: AdamW با weight decay
3. **Scheduler**: ReduceLROnPlateau
4. **Regularization**: Dropout + Batch Normalization
5. **Early Stopping**: Patience = 7

### توصیه‌ها:

- **برای سرعت**: CNN Model
- **برای دقت**: LSTM یا Hybrid
- **برای تعادل**: Hybrid Model
- **برای منابع محدود**: CNN Model
