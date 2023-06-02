import os
import pandas as pd
from pdf_struct.feature_extractor import TextContractFeatureExtractor
from pdf_struct import loader
from pdf_struct.core import transition_labels
from pdf_struct import feature_extractor
from pdf_struct.core.export import to_tree, to_paragraphs
from pdf_struct.core.predictor import train_classifiers, \
    predict_with_classifiers
from pdf_struct.export.hocr import export_result
import tqdm
import pickle

FILE_TYPE = 'docx'

def select_model(model_path:str):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def process_file(file_path):
    try:
        document = loader.modules[FILE_TYPE].load_document(file_path, None, None)
        document = TextContractFeatureExtractor.append_features_to_document(document)
        pred = predict_with_classifiers(clf, clf_ptr, [document])[0]
        return (file_path, to_paragraphs(pred))
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

# Load classifiers
clf = select_model('clf.pkl')
clf_ptr = select_model('clf_ptr.pkl')

# Đường dẫn tới thư mục chứa các file dữ liệu
data_dir = 'data'

# Khởi tạo danh sách kết quả
results = []

# Duyệt qua tất cả các file trong thư mục
for filename in tqdm.tqdm(os.listdir(data_dir)):
    # Lấy đường dẫn đầy đủ đến tệp
    file_path = os.path.join(data_dir, filename)

    # Nếu tệp là tệp văn bản
    if filename.endswith('.docx') or filename.endswith('.txt'):
        # Xử lý tệp và thêm kết quả vào danh sách
        result = process_file(file_path)
        if result is not None:
            results.append(result)

df = pd.DataFrame(results, columns=['filename', 'prediction'])

# Loại bỏ phần data_dir trong phần filename
df['filename'] = df['filename'].apply(lambda x: os.path.basename(x))

# Ghi kết quả vào tệp CSV
df.to_csv('datasets/results.csv', index=False)
