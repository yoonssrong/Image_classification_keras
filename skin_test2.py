from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


modelPath = './model_saved/abdomen_erythema/resnet_v2_152_1000/'  # 모델이 저장된 경로
weight = 'model-225-0.892045-0.937500.h5'  # 학습된 모델의 파일이름
testPath = './data/re_test/'

model = load_model(modelPath + weight)

datagen_test = ImageDataGenerator(rescale=1./255)

batch_size = 256

generator_test = datagen_test.flow_from_directory(directory=testPath,
                                                  target_size=(224, 224),
                                                  batch_size=batch_size,
                                                  shuffle=False
                                                  )


# model로 test set 추론
cls_pred = model.predict_generator(generator_test, steps=1, verbose=1)

cls_test = generator_test.classes
cls_pred_classes = cls_pred.argsort(axis=1)

idx_cls = {v:k for k,v in generator_test.class_indices.items()}

print(cls_test)
print(cls_pred_classes)
