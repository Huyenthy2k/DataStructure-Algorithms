import spacy

# Tải mô hình tiếng Anh đã được huấn luyện
nlp = spacy.load("en_core_web_sm")

text = """
Peter is a student in New York. He loves visiting the Statue of Liberty with his friend, Mary. 
Mary also lives in New York. They often talk about Python programming. What a city New York is!
"""

# Xử lý văn bản
doc = nlp(text)

# Trích xuất các thực thể có tên
print("Các thực thể có tên tìm được bằng spaCy (NER):")

# Đếm số lần xuất hiện của mỗi thực thể
entity_count = {}
for ent in doc.ents:
    entity_key = f"{ent.text} ({ent.label_})"
    if entity_key in entity_count:
        entity_count[entity_key] += 1
    else:
        entity_count[entity_key] = 1

# In ra kết quả với số lần xuất hiện
for entity, count in entity_count.items():
    print(f"- '{entity}' : xuất hiện {count} lần")

"""
Các thực thể có tên tìm được bằng spaCy (NER):
- 'New York (GPE)' : xuất hiện 3 lần
- 'Statue of Liberty (GPE)' : xuất hiện 1 lần
- 'Mary (PERSON)' : xuất hiện 1 lần
- 'Python (NORP)' : xuất hiện 1 lần
"""
"""
Cơ chế hoạt động: NER là bước nhảy vọt từ việc hiểu cấu trúc ngữ pháp sang hiểu ngữ nghĩa. Nó cũng chủ yếu dựa vào Học máy, nhưng ở một mức độ phức tạp hơn nhiều so với POS Tagging.

*Huấn luyện trên dữ liệu đã gán nhãn:
- NER cần một tập dữ liệu khổng lồ, nơi con người đã gán nhãn sẵn cho các thực thể.
- Ví dụ: "Apple Inc. was founded by Steve Jobs in Cupertino." 
Nhãn: [Apple Inc.]ORG was founded by [Steve Jobs]PERSON in [Cupertino]GPE.

*Học các đặc trưng (Feature Learning): Mô hình không chỉ học về từ loại mà còn học về hàng trăm, hàng nghìn đặc trưng khác từ ngữ cảnh:
- Đặc trưng về từ: Từ đó có viết hoa không? Có phải toàn bộ viết hoa không (như NASA)? Từ đó có chứa số không?
- Đặc trưng về ngữ cảnh:
    - Từ đứng trước nó là gì? (Ví dụ: từ "Mr." là một dấu hiệu mạnh cho thấy từ tiếp theo là PERSON).
    - Từ đứng sau nó là gì? (Ví dụ: từ "Inc." là dấu hiệu của ORG).
    - Từ loại của từ hiện tại và các từ xung quanh là gì?

- Đặc trưng về hình thái: Từ đó kết thúc bằng "-ville" (thường là địa danh), hay "-son" (thường là tên người)?
Dựa trên tri thức ngoài: Hệ thống có thể được tích hợp một danh sách lớn các địa danh, tên công ty, tên người nổi tiếng đã biết.

Dự đoán:
- Khi nhận một câu mới, hệ thống NER hiện đại (thường dùng các mô hình như Conditional Random Fields - CRF hoặc mạng nơ-ron hồi quy như LSTM, BiLSTM) sẽ xem xét toàn bộ câu.
- Nó tính toán xác suất cho từng từ thuộc về một loại thực thể nào đó (ví dụ: B-PERSON cho từ bắt đầu tên người, I-PERSON cho từ bên trong tên người, O cho từ không phải thực thể).

Cuối cùng, nó đưa ra chuỗi nhãn có xác suất cao nhất cho cả câu, cho phép nó nhận diện chính xác các thực thể, kể cả những cái nó chưa từng thấy trước đây, dựa vào các đặc trưng đã học được.
"""