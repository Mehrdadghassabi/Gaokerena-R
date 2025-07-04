# تحلیل ساختار Transformer و تفاوت Pre-LN با Post-LN

## مقدمه

مقاله "Attention Is All You Need" معماری Transformer را معرفی کرد که انقلابی در پردازش زبان طبیعی ایجاد کرده است. یکی از جنبه‌های مهم این معماری، نحوه اعمال Layer Normalization است.

## ساختار کلی Transformer

### کامپوننت‌های اصلی:
1. **Multi-Head Attention**: مکانیزم توجه چندگانه
2. **Position-wise Feed-Forward Networks**: شبکه‌های پیش‌خور
3. **Layer Normalization**: نرمال‌سازی لایه
4. **Residual Connections**: اتصالات باقی‌مانده

## تفاوت Post-LN و Pre-LN

### Post-LN (ساختار اصلی Transformer):

```
x → Multi-Head Attention → Add & Norm → Feed Forward → Add & Norm → output
```

#### فرمول Post-LN:
```
# برای بلوک Attention:
x₁ = LayerNorm(x + MultiHeadAttention(x))

# برای بلوک Feed-Forward:
x₂ = LayerNorm(x₁ + FeedForward(x₁))
```

### Pre-LN (ساختار بهبود یافته):

```
x → Layer Norm → Multi-Head Attention → Add → Layer Norm → Feed Forward → Add → output
```

#### فرمول Pre-LN:
```
# برای بلوک Attention:
x₁ = x + MultiHeadAttention(LayerNorm(x))

# برای بلوک Feed-Forward:
x₂ = x₁ + FeedForward(LayerNorm(x₁))
```

## تحلیل تأثیرات

### 1. تأثیر بر Gradient Flow

#### Post-LN:
- **مشکل Gradient Vanishing/Exploding**: در شبکه‌های عمیق
- **ناپایداری آموزش**: به دلیل قرارگیری LayerNorm بعد از residual connection

#### فرمول گرادیان در Post-LN:
```
∂L/∂x = ∂L/∂(LN(x + f(x))) × ∂(LN(x + f(x)))/∂x
      = ∂L/∂(LN(x + f(x))) × ∂LN/∂(x + f(x)) × (1 + ∂f(x)/∂x)
```

#### Pre-LN:
- **بهبود Gradient Flow**: گرادیان مستقیماً از طریق residual connection عبور می‌کند
- **پایداری بیشتر**: کمتر تحت تأثیر تغییرات شدید قرار می‌گیرد

#### فرمول گرادیان در Pre-LN:
```
∂L/∂x = ∂L/∂(x + f(LN(x))) × ∂(x + f(LN(x)))/∂x
      = ∂L/∂(x + f(LN(x))) × (1 + ∂f(LN(x))/∂x × ∂LN(x)/∂x)
```

### 2. مزایای Pre-LN نسبت به Post-LN:

#### پایداری آموزش:
- **Gradient Norm**: در Pre-LN کمتر تغییر می‌کند
- **Learning Rate**: حساسیت کمتر به انتخاب نرخ یادگیری
- **Warm-up**: نیاز کمتر به warm-up مرحله

#### فرمول تحلیل گرادیان:
```
# Pre-LN gradient norm:
||∇L||_Pre-LN ≈ ||∇L||_residual × (1 + bounded_term)

# Post-LN gradient norm:
||∇L||_Post-LN = ||∇L||_LN × |∂LN/∂input| × (1 + ∂f/∂x)
```

### 3. تأثیر عملی:

#### کیفیت آموزش:
- **همگرایی سریع‌تر**: Pre-LN معمولاً سریع‌تر همگرا می‌شود
- **دقت بالاتر**: در برخی وظایف عملکرد بهتری دارد
- **مقاوم‌تر**: در برابر تغییرات hyperparameter

#### محاسبات:
```python
# Post-LN implementation
def post_ln_block(x, attention_fn, ffn_fn):
    # Attention sublayer
    attn_out = attention_fn(x)
    x = layer_norm(x + attn_out)
    
    # Feed-forward sublayer  
    ffn_out = ffn_fn(x)
    x = layer_norm(x + ffn_out)
    return x

# Pre-LN implementation
def pre_ln_block(x, attention_fn, ffn_fn):
    # Attention sublayer
    norm_x = layer_norm(x)
    attn_out = attention_fn(norm_x)
    x = x + attn_out
    
    # Feed-forward sublayer
    norm_x = layer_norm(x)
    ffn_out = ffn_fn(norm_x)
    x = x + ffn_out
    return x
```

## تحلیل ریاضی عمیق‌تر

### تأثیر بر Gradient Scale:

#### Post-LN:
```
∂L/∂x_i = Σ_j ∂L/∂y_j × ∂LN(x + f(x))_j/∂x_i

where: ∂LN(z)_j/∂z_i = (δ_ij - μ_j)/σ × (1 + ∂f/∂x_i)
```

#### Pre-LN:
```
∂L/∂x_i = ∂L/∂y_i + Σ_j ∂L/∂y_j × ∂f(LN(x))_j/∂LN(x)_k × ∂LN(x)_k/∂x_i
```

### مزیت کلیدی Pre-LN:
- **مسیر مستقیم گرادیان**: `∂L/∂y_i` مستقیماً به ورودی منتقل می‌شود
- **تنظیم خودکار**: LayerNorm قبل از عملیات‌های غیرخطی اعمال می‌شود

## نتیجه‌گیری

### Pre-LN برتری دارد زیرا:

1. **پایداری آموزش**: گرادیان‌ها پایدارتر هستند
2. **سرعت همگرایی**: آموزش سریع‌تر و مؤثرتر
3. **مقاومت**: کمتر به تنظیمات hyperparameter وابسته است
4. **عملکرد**: در بسیاری از وظایف نتایج بهتری ارائه می‌دهد

### تأثیر ریاضی کلیدی:
```
Pre-LN: ∇_x L = ∇_direct + ∇_through_normalization
Post-LN: ∇_x L = ∇_through_normalization_only × scaling_factor
```

این تفاوت باعث می‌شود Pre-LN مسیر مستقیم‌تری برای انتشار گرادیان فراهم کند و در نتیجه آموزش پایدارتر و مؤثرتری داشته باشد.
