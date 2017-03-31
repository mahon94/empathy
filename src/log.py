from keras.models import  Sequential
import cPickle as pickle
import json
from keras.models import model_from_json
import time
import sys

endtime = time.asctime(time.localtime(time.time()))


def save_model(model, json_string, dirpath='../data/results/'):
    with open(dirpath + endtime +'.json', 'w') as f:
        f.write(json_string)
    model.save_weights(dirpath + endtime + ".h5")


def save_config(config, dirpath='../data/results/'):
    with open(dirpath + 'config_log.txt', 'a') as f:
        f.write(endtime + '\n')
        f.write(str(config) + '\n')


def save_result(starttime, batch_size, nb_epoch, model, modelParams, train_acc, val_acc, test_acc,
                history = '', dirpath='../data/results/'):
    with open(dirpath + endtime +'_result_log.txt', 'w') as f:
            f.write(starttime + '_' + endtime + '\n')
            f.write('      batch size: ' + str(batch_size) + ', epoches: ' + str(nb_epoch) + '\n')
            # f.write('         summary: ' + str(modelSummary) + '\n')
            f.write('number of params: ' + str(modelParams) + '\n')
            f.write('       train acc: ' + str(train_acc) + '\n')
            f.write('  validation acc: ' + str(val_acc) + '\n')
            f.write('        test acc: ' + str(test_acc) + '\n')
            f.write('         history: ' + str(history) + '\n')
            orig_stdout = sys.stdout
            sys.stdout = f
            print(model.summary())
            sys.stdout = orig_stdout


