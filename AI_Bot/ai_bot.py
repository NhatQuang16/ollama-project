import pandas as pd

def load_csv(file_path):
    # Đọc file CSV
    df = pd.read_csv(file_path, encoding="utf-8")

    # Chuẩn hóa tên cột để tránh lỗi KeyError
    df.columns = df.columns.str.strip()

    return df

# Test đọc file
data = load_csv("bao_tang_ha_noi_khong_dau.csv")
print("Dữ liệu đã đọc:", data[:5])  # In 5 dòng đầu

def search_info(question, data):
    for text in data:
        if any(word.lower() in text.lower() for word in question.split()):  
            return text  # Trả về dòng đầu tiên khớp nhất
    return "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong dữ liệu."

# 📌 Tìm bảo tàng có giá vé rẻ nhất
def find_cheapest_museum(df):
    df["Gia Ve"] = df["Gia Ve"].replace("[^0-9]", "", regex=True).astype(float)
    min_price = df["Gia Ve"].min()
    cheapest_museums = df[df["Gia Ve"] == min_price]
    return cheapest_museums.to_dict(orient="records")

# 📌 Tìm kiếm thông tin phù hợp
def search_info(question, df):
    question_lower = question.lower()

    if "giá vé rẻ nhất" in question_lower or "vé rẻ nhất" in question_lower:
        cheapest_museums = find_cheapest_museum(df)
        if cheapest_museums:
            return "\n".join([f"{m['Ten Bao Tang']} - Giá vé: {m['Gia Ve']} VND" for m in cheapest_museums])
        return "Không tìm thấy bảo tàng nào."

    for _, row in df.iterrows():
        if any(word.lower() in row["Ten Bao Tang"].lower() for word in question.split()):
            return f"{row['Ten Bao Tang']} - Địa chỉ: {row['Dia Chi']} - Giờ mở cửa: {row['Gio Mo Cua']} - Giá vé: {row['Gia Ve']}"

    return "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong dữ liệu."

import ollama

# Load dữ liệu CSV
df = load_csv("bao_tang_ha_noi_khong_dau.csv")

while True:
    user_input = input("Bạn: ")
    if user_input.lower() == "exit":
        break

    # Tìm thông tin phù hợp từ file CSV
    relevant_info = search_info(user_input, df)

    #ưu tiên trả lời bằng tiếng Việt
    messages = [
        {"role": "system", "content": "Bạn là một trợ lý thông minh chuyên trả lời câu hỏi về bảo tàng ở Hà Nội. Luôn trả lời bằng tiếng Việt một cách rõ ràng, dễ hiểu và đầy đủ."},
        {"role": "user", "content": f"Câu hỏi: {user_input}\n\nThông tin từ dữ liệu:\n{relevant_info}\n\nHãy trả lời bằng tiếng Việt một cách tự nhiên và dễ hiểu."}
    ]

    model_name = "gemma"
    response = ollama.chat(model="gemma", messages=messages)

    # Hiển thị phản hồi từ bot
    if "message" in response and "content" in response["message"]:
        print(f"Bot: {response['message']['content']}")
    else:
        print("Bot: Xin lỗi, tôi không thể tìm thấy câu trả lời.")

