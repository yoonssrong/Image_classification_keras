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

print(len(test_gen))

x, y = next(test_gen)
print('acc :', model.evaluate(x, y))
