from ops3 import *

diagnosis = 'excoriation'

train_path = './data/abdomen_{}/'.format(diagnosis) #경로 마지막에 반드시 '/'를 기입해야합니다.
model_name_list = ['densenet_169', 'vgg_16', 'vgg_19']

epoch = 600

if __name__ == '__main__':
    for model_name in model_name_list:
        fine_tunning = Fine_tunning(train_path=train_path,
                                    model_name=model_name,
                                    epoch=epoch)
        history = fine_tunning.training()
        fine_tunning.save_accuracy(history)
