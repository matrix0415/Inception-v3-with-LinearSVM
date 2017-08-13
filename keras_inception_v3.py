import os
import json
import arrow
import pickle
import numpy as np
from operator import itemgetter
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sample_num = '10k'
training_img_path = "dataset/training_pics_%s" % sample_num
testing_img_path = "dataset/testing_pics"
results_save_folder = 'result-%s' % sample_num
game_name_mapping_json = 'dataset/game_mapping.json'
training_feature_Extraction = results_save_folder + "/training_feature_extraction_%s" % sample_num
model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')
SVMKernal = 'linear'

if not os.path.exists(results_save_folder):
    os.mkdir(results_save_folder)


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


def training(img_target_size, save_folder):
    program_start_time = arrow.now().timestamp
    features, labels = feature_extracting(img_target_size=img_target_size)
    training_time = arrow.now().timestamp
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    print("Classifying...")
    clf = svm.SVC(C=10, kernel=SVMKernal, decision_function_shape='ovr')
    for key, x in enumerate(X_train):
        print("Nan: ", key, x) if np.isnan(np.sum(x)) else None

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    img_size_msg = "Target size:\t%s" % str(img_target_size)
    kernal_msg = "SVM Kernal:\t%s" % SVMKernal
    accuracy_str = "Accuracy:\t{0:0.1f}%".format(accuracy_score(y_test, y_pred) * 100)
    training_spent_time = "Training spent Time:\t%s mins." % ((arrow.now().timestamp - training_time)/60)
    total_spent_time = "Total spent Time:\t%s mins." % ((arrow.now().timestamp - program_start_time)/60)

    os.mkdir(save_folder)
    pickle.dump(clf, open('%s/model.clf' % save_folder, 'wb'))
    with open('%s/metrics.tsv' % save_folder, 'w') as f:
        f.write(img_size_msg + "\n" + kernal_msg + "\n" + accuracy_str + "\n" +
                training_spent_time + "\n" + total_spent_time)


def predicting(img_target_size, save_folder):
    print("Loading SVM model...")
    mapping = json.load(open(game_name_mapping_json))
    get_category = lambda game_name: [category for gname, category in mapping if gname == game_name]
    start_time = arrow.now().timestamp
    clf = pickle.load(open('%s/model.clf' % save_folder, 'rb'))
    result = sorted([[i.split('-')[0],
                      '-'.join(i.split('-')[1:]),
                      clf.predict(feature_transferring(testing_img_path + '/' + i, img_target_size))[0]]
                     for i in os.listdir(testing_img_path) if i[:1] != '.'], key=itemgetter(0, 2))
    result = "\n".join(["%s\t%s\t%s\t%s\t%s" % (key, i[0], i[1], i[2], i[2] in get_category(i[0]))
                        for key, i in enumerate(result)])
    spent_time = "Spent Time: %s mins." % ((arrow.now().timestamp - start_time) / 60)

    with open(save_folder+"/testing_results.tsv", 'w') as f:
        f.write(result + "\n\n" + spent_time)
    

if __name__ == '__main__':
    for i in range(1, 11):
        img_target_size = (int(128*i), int(128*i))
        save_folder = "%s/result-%s-%s-%s" % (
            results_save_folder,
            arrow.now().format("YYYYMMDD-HHmm"),
            sample_num,
            "x".join([str(i) for i in img_target_size]))
        training(img_target_size=img_target_size, save_folder=save_folder)
        predicting(img_target_size=img_target_size, save_folder=save_folder)
    

