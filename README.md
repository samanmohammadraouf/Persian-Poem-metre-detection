# Persian Poem Metre Detection (وزن عروضی)

A lightweight repo for detecting the **metre** of a Persian hemistich.  
Given a verse as input, the model outputs a canonical metre sequence (e.g., `مفاعیلن مفاعیلن مفاعیلن فعولن`).

<p align="center">
<img width="415" height="215" alt="image" align="center" src="https://github.com/user-attachments/assets/2292d97e-2b3b-4f7d-b8f5-34c57d22f403" />
</p>

---

## Approaches evaluated

We implemented and compared four approaches:

1. **Multiclass classification with ParsBERT ([CLS])**  
2. **Seq2Seq with RNN (Bi-LSTM encoder) using ParsBERT tokenizer/embeddings**  
3. **Seq2Seq with a Transformer (encoder–decoder) using ParsBERT embeddings**  
4. **Fine-tuning mT5 (mt5-small) as a conditional generation model**

---

## 1) Classification (ParsBERT + [CLS])

- **Task:** 48-class classification over all metre labels.  
- **Tokenizer/encoder:** HooshvareLab ParsBERT; we prepend/use **[CLS]** and classify from the [CLS] representation.  
- **Imbalance handling:** class-weighted cross-entropy.  
- **Result (val):** macro-F1 ≈ **0.73**
<p align="center">
<img width="629" height="581" alt="image" align="center" src="https://github.com/user-attachments/assets/b95708dc-3b87-46ae-a867-a60f1962d89b" />
</p>

---

## 2) Seq2Seq (RNN)

- **Setup:** ParsBERT for input embeddings → **Bi-LSTM encoder** + attention decoder; custom space-based tokenizer for metre tokens.  
- **Result (val):**  
  - **BLEU ≈ 0.620**  
  - **ROUGE-1 F1 ≈ 0.764**, **ROUGE-2 F1 ≈ 0.668**  
  - Token-level: **Acc ≈ 0.759**, **F1 ≈ 0.411**

<p align="center">
<img width="482" height="596" alt="image" align="center" src="https://github.com/user-attachments/assets/98fea681-12d3-4a92-87ae-0224a2cd658a" />
</p>
---

## 3) Seq2Seq (Transformer)

- **Setup:** ParsBERT embeddings → linear projection + positional encoding → **nn.Transformer** encoder–decoder; learned output embedding & linear head.  
- **Result (val):**  
  - **BLEU ≈ 0.246**  
  - **ROUGE-1 F1 ≈ 0.447**, **ROUGE-2 F1 ≈ 0.284**

---

## 4) Fine-tuning mT5 (mt5-small)

- **Setup:** `google/mt5-small` tokenizer & model for conditional generation; custom label tokenizer with EOS for metre output space.  
- **Result (val):**  
  - **BLEU ≈ 0.857**  
  - **ROUGE-1 F1 ≈ 0.912**, **ROUGE-2 F1 ≈ 0.874**  
  - Reported **accuracy > 90%**

---

## Comparison (validation)

| Approach              | Type              | Key Metrics |
|-----------------------|------------------|-------------|
| ParsBERT [CLS]        | 48-class classifier | macro-F1 ~ **0.73** |
| RNN (Bi-LSTM)         | Seq2Seq          | **BLEU ~0.62**, **ROUGE-1 F1 ~0.764**, **ROUGE-2 F1 ~0.668**; token **Acc ~0.759** |
| Transformer           | Seq2Seq          | **BLEU ~0.246**, **ROUGE-1 F1 ~0.447**, **ROUGE-2 F1 ~0.284** |
| mT5 (fine-tuned)      | Seq2Seq (pretrained) | **BLEU ~0.857**, **ROUGE-1 F1 ~0.912**, **ROUGE-2 F1 ~0.874**, **Acc > 90%** |

---

## Conclusion

- **mT5 fine-tuning** clearly dominates other methods on sequence-level metrics (BLEU/ROUGE) and achieves **>90% accuracy**, making it the recommended production path.  
- **RNN Seq2Seq** is a strong non-pretrained baseline and notably better than the vanilla Transformer under our settings.  
- **ParsBERT [CLS] classifier** gives a reasonable 48-class baseline (macro-F1 ~0.73) but lacks token-level structure needed for fine-grained metre sequences.

---
If you’re reproducing results, mind the dataset’s class imbalance and keep **class-weighted loss** for the classifier baseline.
