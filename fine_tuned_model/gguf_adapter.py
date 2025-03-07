from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class GGUFAdapter:
    def __init__(self, model_name):
        # Tải mô hình và tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, text):
        # Tiền xử lý đầu vào văn bản
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Dự đoán từ mô hình
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Xử lý kết quả (có thể cần xác định cách xử lý tùy vào loại mô hình)
        logits = outputs.logits
        predicted_class = logits.argmax(dim=-1).item()
        
        return predicted_class
