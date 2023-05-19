import pickle
from pdf_struct import loader
from pdf_struct.core.export import to_paragraphs
from pdf_struct.core.predictor import predict_with_classifiers
from pdf_struct.feature_extractor import TextContractFeatureExtractor


def select_model(model_path:str):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

clf = select_model('clf.pkl')

clf_ptr = select_model('clf_ptr.pkl')



def extract_docx(file, FILE_TYPE = 'docx'):
    document = loader.modules[FILE_TYPE].load_document(file, None, None)

    document = TextContractFeatureExtractor.append_features_to_document(document)

    pred = predict_with_classifiers(clf, clf_ptr, [document])[0]

    return to_paragraphs(pred)