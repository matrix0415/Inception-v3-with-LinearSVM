import os
import cv2
import json
import arrow
import shutil
import random
import requests
from PIL import Image
from glob import glob
from optparse import OptionParser
from google_play_lib import GooglePlaySearch, ualist

cdn_url_prefix = 'https://cdn.soocii.me'
training_pics_save_path = 'dataset/training_pics'
testing_pics_save_path = 'dataset/testing_pics'

# parser = argparse.ArgumentParser()
# parser.add_argument('download-testing-photos')
# parser.add_argument('download-training-gp-photos')
# parser.add_argument('download-sampling-streaming')
# parser.add_argument('use-sampling-streaming')

# download_pepper_raw_records_from_s3()
# data = download_photo_testing_data_from_s3()
# search_google_play_for_images(keywords=set([i[1] for i in data]))
# for i in [20]:
#     sampling_photos_from_video(sampling=10000*i)
# move_videoclip_pics_to_training()
# subsampling_pics(from_folder='dataset/training_pics_100k', sampling_k=30)
# generate_reports(folder='result-10k-softmax')


def video_frame_capture(video_file, img_saving_folder, sampling_number):
    count = 0
    success = True
    vidcap = cv2.VideoCapture(video_file)
    framerate = vidcap.get(cv2.CAP_PROP_FPS)
    capture = round(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)/sampling_number)
    if capture > framerate*4:
        capture = round(framerate*4)
    if not os.path.exists(img_saving_folder):
        os.mkdir(img_saving_folder)
    while success:
        success, image = vidcap.read()
        if count > framerate * 2 and count % capture == 0:
            print('Read a new frame %s: ' % count, success)
            cv2.imwrite(img_saving_folder + "/frame-%d.jpg" % count, image)
        count += 1


def img_write(uri, save_location):
    try:
        print("Downloading: ", uri)
        headers = {'User-Agent': ualist[random.randint(0, len(ualist) - 1)]}
        response = requests.get(uri, headers=headers, stream=True)
        with open(save_location, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        del response
    except Exception as e:
        print("Error: ", e)


def search_google_play_for_images(keywords):
    for row in GooglePlaySearch(keywords=keywords):
        row['app_name']['name'] = row['app_name']['name'].replace('/', '')
        folder_name = row['app_name']['name'] + "_" + row['pkg_name']
        if not os.path.exists(os.path.join(training_pics_save_path, folder_name)):
            os.mkdir(os.path.join(training_pics_save_path, folder_name))
            for nb, img in enumerate(row['images']):
                print("Gathering %s_%s.jpg" % (row['app_name']['name'], nb))
                path = os.path.join(training_pics_save_path, folder_name, "%s_%s.jpg" % (row['app_name']['name'], nb))
                if not os.path.exists(path):
                    img_write(img, path)


def download_pepper_raw_records_from_s3():
    os.system("rm dataset/pepper_posted_*.tsv")
    for target in ['posted_streaming_table', 'posted_photo_table', 'posted_video_table']:
        raw_dataset_tmp = target + "_tmp"
        output_file = "dataset/pepper_" + target + "_%s.tsv" % arrow.utcnow().format("YYYYMMDD")
        if not os.path.exists(raw_dataset_tmp):
            os.mkdir(raw_dataset_tmp)
        os.system('aws s3 cp s3://soocii-table/soocii_pepper/%s.tsv %s --recursive' % (target, raw_dataset_tmp))
        daily_records = "\n".join([
            open(os.path.join(raw_dataset_tmp, daypath, target + ".tsv"), encoding='utf-8').read()
            for daypath in os.listdir(raw_dataset_tmp)])
        rs = [[i.split('\t')[0], i.split('\t')[7]] for i in daily_records.split('\n')
              if i and i.split('\t')[7] != 'null']
        rs = "\n".join(["\t".join(i) for i in sorted(rs, key=lambda x: x[1])])
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(rs)
        shutil.rmtree(raw_dataset_tmp)


def download_photo_testing_data_from_s3():
    photo_records = 'dataset/pepper_posted_photo_table_%s.tsv' % arrow.utcnow().format("YYYYMMDD")
    if not os.path.exists(testing_pics_save_path):
        os.mkdir(testing_pics_save_path)
    if not os.path.exists(photo_records):
        download_pepper_raw_records_from_s3()
    data = [i.split('\t') for i in open(photo_records, encoding='utf-8').read().split('\n')]
    for photoID, gameName in data:
        print(gameName, photoID)
        save_path = testing_pics_save_path + "/" + gameName.replace('/', '-') + "-" + photoID + ".jpg"
        if not os.path.exists(save_path):
            img_write(uri=cdn_url_prefix + "/" + photoID, save_location=save_path)
    return data


def move_videoclip_pics_to_training():
    mapping = json.load(open('dataset/game_mapping.json'))
    training_folder = 'dataset/training_pics'
    video_clip_folder = 'dataset/training_pics_video_sampling_200000'
    for game_name, target_folder in mapping:
        if not os.path.exists(os.path.join(training_folder, target_folder)):
            os.mkdir(os.path.join(training_folder, target_folder))
        print()
        print("Processing %s:%s..." % (game_name, target_folder))
        for i in os.listdir(video_clip_folder):
            print(game_name, i)
            if i != '.DS_Store' and i.split('-')[0] == game_name:
                for f in os.listdir(os.path.join(video_clip_folder, i)):
                    print("Moving ", os.path.join(video_clip_folder, i), " ...")
                    shutil.move(os.path.join(video_clip_folder, i, f),
                                os.path.join(training_folder, target_folder, i + '-' + f))
                os.rmdir(os.path.join(video_clip_folder, i))


def sampling_photos_from_video(sampling=10000):
    video_tmp_folder = "dataset/video_tmp"
    sampling_pics_save_folder = training_pics_save_path + "_video_sampling_" + str(sampling)
    photos_games_list = set([i.split('\t')[1] for i in
                             open(glob('dataset/pepper_posted_photo_table*')[0], encoding='utf-8').read().split('\n')])
    if not os.path.exists(video_tmp_folder):
        os.mkdir(video_tmp_folder)
    if not os.path.exists(sampling_pics_save_folder):
        os.mkdir(sampling_pics_save_folder)
    if not glob('dataset/pepper_posted_streaming_table*'):
        download_pepper_raw_records_from_s3()
    if not glob('dataset/pepper_posted_video_table*'):
        download_pepper_raw_records_from_s3()

    video_list = open(glob('dataset/pepper_posted_streaming_table*')[0], encoding='utf-8').read()+"\n"
    video_list += open(glob('dataset/pepper_posted_video_table*')[0], encoding='utf-8').read()
    video_list = [i.split('\t') for i in video_list.split('\n') if i]
    pics_per_video = round(sampling / sum([1 for vid, gname in video_list if gname in photos_games_list]))

    for video, gname in video_list:
        if gname in photos_games_list and gname != 'Soocii':
            gname = gname.replace('/', '')
            pics_save_folder_per_video = sampling_pics_save_folder+"/%s-%s" % (gname, video)
            if not os.path.exists(pics_save_folder_per_video):
                print("Gethering: ", gname, video, "...")
                save_video_file = video_tmp_folder + "/" + video + ".mp4"
                cli_command = 'aws s3 cp s3://pepper-prod-backend-media-%s/%s %s' % (
                                                                "-".join(video.split('-')[1:]), video, save_video_file)
                os.system(cli_command)
                if not os.path.exists(save_video_file):
                    img_write(uri=cdn_url_prefix+"/"+video, save_location=save_video_file)
                if os.path.exists(save_video_file):
                    try:
                        video_frame_capture(video_file=save_video_file,
                                            img_saving_folder=pics_save_folder_per_video,
                                            sampling_number=pics_per_video)
                    except:
                        img_write(uri=cdn_url_prefix+"/"+video, save_location=save_video_file)
                        video_frame_capture(video_file=save_video_file,
                                            img_saving_folder=pics_save_folder_per_video,
                                            sampling_number=pics_per_video)
                    os.remove(save_video_file)
    shutil.rmtree(video_tmp_folder)
    move_videoclip_pics_to_training()


def subsampling_pics(from_folder, sampling_k):
    increase_ratio = 1.5 + ((sampling_k / 10) - 1) * 0.8
    destination_folder = 'dataset/training_pics_%sk' % sampling_k
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)
    sample_total_num = [1 for fo in os.listdir(from_folder) if fo != '.DS_Store'
                        for _ in os.listdir(os.path.join(from_folder, fo))].__len__()
    if sampling_k > sample_total_num/1000:
        raise ValueError("Sampling %sk > Total Sample num." % sampling_k)
    sampling = sampling_k * 1000
    category = [c for c in os.listdir(from_folder)]
    current_total_sampling = 0
    for c_num, c in enumerate(category):
        if c != '.DS_Store':
            print("Processing ", c_num, "/", len(category), current_total_sampling, c, "...")
            current_sampling = round(sampling / (len(category) - (c_num - 1)) * increase_ratio)
            c_path = os.path.join(from_folder, c)
            save_c_path = os.path.join(destination_folder, c)
            if not os.path.exists(save_c_path):
                os.mkdir(save_c_path)
            if [_ for _ in os.listdir(c_path)].__len__() > current_sampling:
                choose_pics = [i for i in random.sample(os.listdir(c_path), current_sampling)]
            else:
                choose_pics = [i for i in os.listdir(c_path)]
            current_total_sampling += [shutil.copyfile(os.path.join(c_path, pics), os.path.join(save_c_path, pics))
                                       for pics in choose_pics].__len__()


def generate_reports(folder):
    rs = []

    for i in os.listdir(folder):
        if i.split('-')[0] == 'result':
            records = [c.split('\t')[:6]
                       for c in open(os.path.join(folder, i, 'testing_results.tsv')).read().split('\n')[:-2]]
            valid_records = [[key, name, pic, fo, r]
                             for key, name, pic, fo, r, prob in records
                             if name != 'Soocii' and name != '(unknown)' and float(prob)>0.9]
            title = "-".join(i.split('-')[3:])
            accuracy = sum([1 for k in valid_records if k[4] == 'True']) / len(valid_records)
            details = {i.split('\t')[0]: i.split('\t')[1]
                       for i in open(os.path.join(folder, i, 'metrics.tsv')).read().split('\n')}
            rs.append([title, accuracy, len(valid_records)/len(records), len(valid_records)]) # , details])

    for i in sorted(rs, key=lambda x: x[0]):
        print(i)
        print()
    return rs


def demo(url):
    import pickle
    import numpy as np
    import skvideo.io
    from keras.models import load_model
    from keras.preprocessing import image
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    from keras_inception_v3 import label_to_category

    save_path = 'demo'
    datasize = '10k'
    model = 'softmax'
    img_target_size = (384, 384)

    result_path = 'result-%s-%s' % (datasize, model)
    save_tmp_file = os.path.join(save_path, 'demo.mp4')
    model_path = glob(os.path.join(
        result_path, 'result-*-%s-%s' % (datasize, 'x'.join([str(i) for i in img_target_size])), '*.h5'
    ))[0]
    print(os.path.join(
        result_path,
        'training_feature_extraction_%s' % datasize,
        'x'.join([str(i) for i in img_target_size]),
        'labels.dat'
    ))
    label_path = glob(os.path.join(
        result_path,
        'training_feature_extraction_%s' % datasize,
        'x'.join([str(i) for i in img_target_size]),
        'labels.dat'
    ))[0]

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # img_write(url, save_location=save_tmp_file)

    if os.path.exists(save_tmp_file):
        try:
            Image.open(save_tmp_file)
        except IOError:
            print("It's not an image file.")
            video = skvideo.io.LibAVReader('demo/demo.mp4')
            for key, frame in enumerate(video.nextFrame()):
                if key % 30 == 0:
                    filename = "%s/demo-%s.jpg" % (save_path, key)
                    print("Saving ", filename)
                    skvideo.io.vwrite(filename, frame)
            os.remove(save_tmp_file)

        imgs = [image.load_img(os.path.join(save_path, f), target_size=img_target_size) for f in os.listdir(save_path) if f != '.DS_Store']
        x = np.array([image.img_to_array(img) for img in imgs])
        x = preprocess_input(x)
        model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')
        img_features = model.predict(x)
        clf = load_model(model_path)
        pred = clf.predict(img_features)
        avg_prob = np.mean(pred, axis=0)
        cate = np.argsort(avg_prob)[-3:]
        label = label_to_category(labels=pickle.load(open(label_path, 'rb')), type='testing')
        print({label[i]: avg_prob[i] for i in cate})
    

if __name__ == '__main__':
    # download_pepper_raw_records_from_s3()
    # data = download_photo_testing_data_from_s3()
    # search_google_play_for_images(keywords=set([i[1] for i in data]))
    # for i in [20]:
    #     sampling_photos_from_video(sampling=10000*i)
    # move_videoclip_pics_to_training()
    # subsampling_pics(from_folder='dataset/training_pics_100k', sampling_k=30)
    # generate_reports(folder='result-10k-softmax')
    demo(url='http://cdn.soocii.me/0025eccdbc684118adc2acbff7f4c8db.mp4')
