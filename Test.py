import pandas as pd
from sklearn import metrics

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


modelPath = './model_saved/resnet_v2_152_1000/'  # 모델이 저장된 경로
weight = 'model-239-0.963908-0.911765.h5'  # 학습된 모델의 파일이름
testPath = './data/test/'  # 테스트 이미지 폴더

model = load_model(modelPath + weight)

model.summary()

test_datagen = ImageDataGenerator(rescale=1. / 255,
                                  featurewise_std_normalization=True)
test_gen = test_datagen.flow_from_directory(
    directory=testPath,
    target_size=(224, 224),
    shuffle=False)

# model로 test set 추론
cls_pred = model.predict_generator(test_gen, steps=1, verbose=1)
cls_test = test_gen.classes
cls_pred_argmax = cls_pred.argmax(axis=1)

# 결과 산출 및 저장
report = metrics.classification_report(y_true=cls_test, y_pred=cls_pred_argmax, output_dict=True)
report = pd.DataFrame(report).transpose()
report.to_csv(f'./output/report_test_{weight[:-3]}.csv', index=True, encoding='cp949')