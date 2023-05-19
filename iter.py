import tqdm

from pdf_struct import loader
from pdf_struct.core import transition_labels, feature_extractor
from pdf_struct.core.export import to_paragraphs
from pdf_struct.core.predictor import train_classifiers, predict_with_classifiers
from pdf_struct.feature_extractor import TextContractFeatureExtractor

annos = transition_labels.load_annos('datasets/anno')

FILE_TYPE = 'docx'
documents = loader.modules[FILE_TYPE].load_from_directory('datasets/raw', annos)
assert len(documents) > 0
feature_extractor_cls = feature_extractor.feature_extractors['TextContractFeatureExtractor']
documents = [feature_extractor_cls.append_features_to_document(document)
                 for document in tqdm.tqdm(documents)]

clf, clf_ptr = train_classifiers(documents)

'''with open('clf.pkl', 'wb') as file:
    pickle.dump(clf, file)

with open('clf_ptr.pkl', 'wb') as file:
    pickle.dump(clf_ptr, file)'''

# Now make predictions
document = loader.modules[FILE_TYPE].load_document('datasets/raw/LeHuuLoi_10120764_124201_DoAn2.docx', None, None)

document = TextContractFeatureExtractor.append_features_to_document(document)

pred = predict_with_classifiers(clf, clf_ptr, [document])[0]


'''# đường dẫn tới thư mục chứa các file dữ liệu
data_dir = 'datasets/raw'

# khởi tạo danh sách kết quả
results = []

# duyệt qua tất cả các file trong thư mục
for filename in os.listdir(data_dir):
    # lấy đường dẫn đầy đủ đến tệp
    file_path = os.path.join(data_dir, filename)

    # nếu tệp là tệp văn bản
    if filename.endswith('.docx') or filename.endswith('.txt'):
        # load tài liệu và trích xuất đặc trưng
        document = loader.modules[FILE_TYPE].load_document(file_path, None, None)
        document = TextContractFeatureExtractor.append_features_to_document(document)

        # dự đoán lớp của tài liệu
        pred = predict_with_classifiers(clf, clf_ptr, [document])[0]

        # thêm kết quả vào danh sách
        results.append((filename, to_paragraphs(pred)))

# tạo DataFrame từ danh sách kết quả
df = pd.DataFrame(results, columns=['filename', 'prediction'])

# ghi kết quả vào tệp CSV
df.to_csv('datasets/results.csv', index=False)'''

print(to_paragraphs(pred))

