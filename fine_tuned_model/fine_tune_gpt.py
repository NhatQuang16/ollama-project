from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from safetensors.torch import load_file
from datasets import Dataset
from gguf_adapter import GGUFAdapter
import torch
import pandas as pd

model_data = load_file("model.safetensors")
print(model_data.keys())

# Đọc file JSONL bằng pandas
df = pd.read_json("baotang_hanoi.jsonl", lines=True)

# Chuyển đổi DataFrame thành Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Đường dẫn tới mô hình và tokenizer
model_path = "C:/ollama-project/fine_tuned_model/checkpoint-3/model.safetensors"

# Kiểm tra dữ liệu
print(dataset)

# Tải mô hình và tokenizer
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Tải mô hình
model = AutoModelForCausalLM.from_pretrained(model_path, from_tf=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Đặt pad_token cho tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Tokenize dữ liệu
def tokenize_function(examples):
    # Kết hợp các cột thành chuỗi
    text = [instr + " " + inp + " " + outp for instr, inp, outp in zip(examples['instruction'], examples['input'], examples['output'])]
    
    # Token hóa dữ liệu với padding và truncation
    return tokenizer(text, padding="max_length", truncation=True, max_length=512)

# Tokenize dataset với batched=True và tránh lỗi chiều dài không khớp
def map_function(examples):
    result = tokenize_function(examples)
    # Đảm bảo rằng mọi cột đều có chiều dài giống nhau sau khi token hóa
    result["labels"] = result["input_ids"]
    return result

# Tokenize dataset với batched=True
tokenized_datasets = dataset.map(map_function, batched=True)

# Kiểm tra kết quả
print(tokenized_datasets)

# Thiết lập các tham số huấn luyện
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
)

# Fine-tune mô hình
trainer.train()

model.save_pretrained("C:/ollama-project/fine_tuned_model")
tokenizer.save_pretrained("C:/ollama-project/fine_tuned_model")

def main():
    # Khởi tạo adapter với tên mô hình GGUF
    adapter = GGUFAdapter("bert-base-uncased")  # Tên mô hình có thể thay đổi tùy vào mô hình bạn muốn sử dụng
    
    # Văn bản để phân loại hoặc xử lý
    text = "Hà Nội có nhiều bảo tàng hấp dẫn."
    
    # Sử dụng adapter để dự đoán (có thể là phân loại văn bản)
    prediction = adapter.predict(text)
    
    # In kết quả dự đoán (có thể là lớp dự đoán, hoặc bạn có thể mở rộng xử lý tùy theo mô hình)
    print(f"Class predicted: {prediction}")
