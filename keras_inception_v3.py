import os
import json
import arrow
import pickle
import numpy as np
from operator import itemgetter
from keras import Input
from keras.engine import Model
from keras.layers import Dense, noise
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold


sample_num = '100k'
training_type = 'softmax'       # linear, softmax, sda
training_img_path = "dataset/training_pics_%s" % sample_num
testing_img_path = "dataset/testing_pics"
results_save_folder = 'result-%s-%s' % (sample_num, training_type)
game_name_mapping_json = 'dataset/game_mapping.json'
training_feature_Extraction = results_save_folder + "/training_feature_extraction_%s" % sample_num
model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')


if not os.path.exists(results_save_folder):
    os.mkdir(results_save_folder)


def label_to_category(labels, type='training'):
    if type == 'training':
        return {i: key for key, i in enumerate(set(labels))}
    elif type == 'testing':
        return {key: i for key, i in enumerate(set(labels))}


def feature_transferring(img_file_path, img_target_size):
    print("Proccessing: %s" % img_file_path)
    try:
        img = image.load_img(img_file_path, target_size=img_target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return model.predict(x)
    except OSError:
        print("File Error: ", img_file_path)
        os.remove(img_file_path)
        return None


def feature_extracting(img_target_size):
    path = training_feature_Extraction + "/" + "x".join([str(i) for i in img_target_size])
    if not os.path.exists(training_feature_Extraction):
        os.mkdir(training_feature_Extraction)
    if os.path.exists(path) and os.path.exists(path + '/features.dat') and os.path.exists(path + '/labels.dat'):
        print("Loading Image Features...")
        features = pickle.load(open(path + '/features.dat', 'rb'))
        labels = pickle.load(open(path + '/labels.dat', 'rb'))
    else:
        labels = []
        features = np.array([]).reshape(-1, 2048)
        start_time = arrow.now().timestamp
        features_per_images = path + "/per_images"
        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(features_per_images)
        for label in [i for i in os.listdir(training_img_path) if list(i)[0] != '.']:
            for file in [i for i in os.listdir(training_img_path + "/" + label) if list(i)[0] != '.']:
                feature_file_name = features_per_images + "/%s-%s.dat" % (label, file)

                if file[:1] != '.':
                    if os.path.exists(feature_file_name):
                        print("Loading ", feature_file_name, " ...")
                        preds = pickle.load(open(feature_file_name, 'rb'))
                    else:
                        img_file_path = training_img_path + "/" + label + "/" + file
                        preds = feature_transferring(img_file_path=img_file_path, img_target_size=img_target_size)
                        pickle.dump(preds, open(feature_file_name, 'wb'))
                    if preds is not None:
                        features = np.vstack([features, preds])
                        labels.append(label)

        spent_time = "Spent Time: %s mins." % ((arrow.now().timestamp - start_time)/60)
        print("Saving features...")
        pickle.dump(features, open(path + '/features.dat', 'wb'))
        pickle.dump(labels, open(path + '/labels.dat', 'wb'))

        with open('%s/metrics.log' % path, 'a') as f:
            f.write(spent_time)

    return features, labels


def svm_training(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=10)
    print("Classifying...")
    clf = svm.SVC(C=10, kernel=training_type, decision_function_shape='ovr')
    for key, x in enumerate(X_train):
        print("Nan: ", key, x) if np.isnan(np.sum(x)) else None
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    return clf, accuracy


def cv_training(features, labels):
    print("Classifying...")
    tmp_accuracy = []
    clf = svm.SVC(C=10, kernel=training_type, decision_function_shape='ovr')
    kf = KFold(n_splits=10, shuffle=True)
    for train, test in kf.split(labels):
        clf.fit(features[train], labels[train])
        y_pred = clf.predict(features[test])
        tmp_accuracy.append(accuracy_score(labels[test], y_pred) * 100)
    accuracy = sum(tmp_accuracy)/len(tmp_accuracy)
    return clf, accuracy


def softmax_training(features, labels):
    int_labels = label_to_category(labels=labels, type='training')
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=10)
    labels = to_categorical([int_labels[i] for i in labels])
    y_train = to_categorical([int_labels[i] for i in y_train])
    y_test = to_categorical([int_labels[i] for i in y_test])

    finetuning = Sequential()
    finetuning.add(Dense(1024, input_shape=(X_train.shape[1],), activation='sigmoid'))
    finetuning.add(Dense(242, input_shape=(1024,), activation='softmax'))
    finetuning.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    finetuning.fit(features, labels, epochs=30, batch_size=128, validation_split=0.8)
    accuracy = finetuning.evaluate(features, labels, batch_size=256)[1] * 100
    return finetuning, accuracy


def sda_training(features, labels):
    encoder_dims = [1600, 1024, 768]
    stacked_encoder = []
    int_labels = label_to_category(labels=labels, type='training')
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=10)
    y_train = to_categorical([int_labels[i] for i in y_train])
    y_test = to_categorical([int_labels[i] for i in y_test])

    for encoder_dim in encoder_dims:
        input_dim = X_train.shape[1]
        input_img = Input(shape=(input_dim,))
        n_layer = noise.GaussianNoise(0.3)(input_img)
        encode = Dense(encoder_dim, activation='sigmoid')(n_layer)
        decode = Dense(input_dim, activation='sigmoid')(encode)

        ae = Model(input_img, decode)
        ae.compile(optimizer='adam', loss='mape')
        ae.fit(X_train, X_train, epochs=10, batch_size=32, shuffle=True, validation_data=(X_test, X_test))

        encoder = Model(input_img, encode)
        X_train = encoder.predict(X_train)
        X_test = encoder.predict(X_test)
        stacked_encoder.append(encoder.layers[-1])


def training(img_target_size, save_folder):
    program_start_time = arrow.now().timestamp
    features, labels = feature_extracting(img_target_size=img_target_size)
    training_time = arrow.now().timestamp

    clf, accuracy = {
        'linear': svm_training,
        'cv': cv_training,
        'softmax': softmax_training
    }[training_type](features=features, labels=labels)

    img_size_msg = "Target size:\t%s" % str(img_target_size)
    kernal_msg = "Kernal:\t%s" % training_type.upper()
    accuracy_str = "Accuracy:\t{0:0.1f}%".format(accuracy)
    training_spent_time = "Training spent Time:\t%s mins." % ((arrow.now().timestamp - training_time)/60)
    total_spent_time = "Total spent Time:\t%s mins." % ((arrow.now().timestamp - program_start_time)/60)

    os.mkdir(save_folder)
    if training_type == 'linear':
        pickle.dump(clf, open('%s/%s-model.clf' % (save_folder, training_type), 'wb'))
    elif training_type == 'softmax':
        clf.save('%s/%s-model.h5' % (save_folder, training_type))
        del clf
    with open('%s/metrics.tsv' % save_folder, 'w') as f:
        f.write(img_size_msg + "\n" + kernal_msg + "\n" + accuracy_str + "\n" +
                training_spent_time + "\n" + total_spent_time)


def predicting(img_target_size, save_folder):
    print()
    print("Loading %s model..." % training_type.upper())
    result = []
    labels_int = {}
    mapping = json.load(open(game_name_mapping_json))
    get_category = lambda game_name: [category for gname, category in mapping if gname == game_name]
    start_time = arrow.now().timestamp
    if training_type == 'linear':
        clf = pickle.load(open('%s/%s-model.clf' % (save_folder, training_type), 'rb'))
    elif training_type == 'softmax':
        clf = load_model('%s/%s-model.h5' % (save_folder, training_type))

    for i in os.listdir(testing_img_path):
        if i[:1] != '.':
            gname = i.split('-')[0]
            fname = '-'.join(i.split('-')[1:])
            if training_type == 'linear':
                pred = clf.predict(feature_transferring(testing_img_path + '/' + i, img_target_size))[0]
                prob = 1
            elif training_type == 'softmax':
                training_labels_path = training_feature_Extraction + "/" + "x".join([str(i) for i in img_target_size])
                training_labels = pickle.load(open(training_labels_path + '/labels.dat', 'rb'))
                labels_int = label_to_category(labels=training_labels, type='testing')
                pickle.dump(labels_int, open(save_folder+'/testing_labels.dat', 'wb'))

                img_feature = feature_transferring(testing_img_path + '/' + i, img_target_size)
                pred_feature = clf.predict(img_feature)
                pred = labels_int[np.argmax(pred_feature)]
                prob = np.max(pred_feature)
            result.append([gname, fname, pred, prob])

    result = sorted(result, key=itemgetter(0, 2))
    result = "\n".join(["%s\t%s\t%s\t%s\t%s\t%s" % (key, i[0], i[1], i[2], i[2] in get_category(i[0]), i[3])
                        for key, i in enumerate(result)])

    spent_time = "Spent Time: %s mins." % ((arrow.now().timestamp - start_time) / 60)

    with open(save_folder+"/testing_results.tsv", 'w') as f:
        f.write(result + "\n\n" + spent_time)
    

if __name__ == '__main__':
    for i in range(5, 6):
        img_target_size = (int(128*i), int(128*i))
        save_folder = "%s/result-%s-%s-%s" % (
            results_save_folder,
            arrow.now().format("YYYYMMDD-HHmm"),
            sample_num,
            "x".join([str(i) for i in img_target_size]))
        training(img_target_size=img_target_size, save_folder=save_folder)
        predicting(img_target_size=img_target_size, save_folder=save_folder)

    

