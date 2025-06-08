# 中川さんのコード？ 一応残しておくが必要なさそう

import os
import re
import numpy as np

IMG_EXTENSION = ['.jpg', 'png']

BASSOON = '1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0'
CELLO = '0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0'
CLARINET = '0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0'
DOUBLE_BASS = '0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0'
FLUTE = '0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0'
HORN = '0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0'
OBOE = '0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0'
SAX = '0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0'
TROMBONE = '0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0'
TRUMPET = '0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0'
TUBA = '0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0'
VIOLA = '0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0'
VIOLIN = '0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1'

re_digits = re.compile(r'(\d+)')


def embedded_numbers(s):
    pieces = re_digits.split(s)
    pieces[1::2] = map(int, pieces[1::2])
    return pieces


def sort_string(lst):
    return sorted(lst, key=embedded_numbers)


def add_label(path):
    if path.upper().__contains__('BASSOON'):
        return BASSOON
    elif path.upper().__contains__('CELLO'):
        return CELLO
    elif path.upper().__contains__('CLARINET'):
        return CLARINET
    elif path.upper().__contains__('DOUBLE_BASS'):
        return DOUBLE_BASS
    elif path.upper().__contains__('FLUTE'):
        return FLUTE
    elif path.upper().__contains__('HORN'):
        return HORN
    elif path.upper().__contains__('OBOE'):
        return OBOE
    elif path.upper().__contains__('SAX'):
        return SAX
    elif path.upper().__contains__('TROMBONE'):
        return TROMBONE
    elif path.upper().__contains__('TRUMPET'):
        return TRUMPET
    elif path.upper().__contains__('TUBA'):
        return TUBA
    elif path.upper().__contains__('VIOLA'):
        return VIOLA
    elif path.upper().__contains__('VIOLIN'):
        return VIOLIN
    else:
        print("LABEL ERROR")
        return False


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSION)


def make_dataset(dir, max_dataset_size=float("inf")):
    assert os.path.isfile(dir), '%s is not a valid txt file' % dir
    audio = []
    label = []
    image1 = []
    image2 = []
    image3 = []
    image4 = []

    with open(dir, 'r') as f:
        for line in f.readlines():
            line_list = line.split('||')
            audio.append(line_list[0])
            label.append(np.fromstring(line_list[1], dtype=float, sep=','))
            image1.append(line_list[2])
            image2.append(line_list[3])
            image3.append(line_list[4])
            image4.append(line_list[5].split('\n')[0])
    return {'audio': audio[:min(max_dataset_size, len(audio))],
            'label': label[:min(max_dataset_size, len(label))],
            'image1': image1[:min(max_dataset_size, len(image1))],
            'image2': image2[:min(max_dataset_size, len(image2))],
            'image3': image3[:min(max_dataset_size, len(image3))],
            'image4': image4[:min(max_dataset_size, len(image4))]}


def prepare_data(dir):
    global i, j
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    filepaths = []
    num = 10000
    num_end = 19900
    second = 100
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sort_string(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)

                #########concate使う時に数が合わなくなるから修正(6/20)
                if "spec" in path:
                    if "train" in path:
                        if "trombone05" in path:
                            path_name_a = "../../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/spec_t_05/train_05/trombone/trombone05_"
                            path_name_b =".png"
                            filepaths.append(path_name_a + str(num) + path_name_b)
                            num = num + 100
                        if num > num_end:
                            break
                            
                    elif "test" in path:
                        if "trombone02" in path:
                            path_name_a = "../../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/spec_t_05/test/trombone/trombone02_"
                            path_name_b =".png"
                            filepaths.append(path_name_a + str(num) + path_name_b)
                            num = num + 100
                        if num > num_end:
                            break
                            
                    
                if "img" in path:
                    if "train" in path:
                        if "trombone05" in path:
                            path_name_a = "../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img_t_05/train_05/trombone/trombone05_"
                            path_name_b =".jpg"
                            filepaths.append(path_name_a + str(num+second) + path_name_b)
                            num = num + 100
                        if num > num_end:
                            break
                            
                    elif "test" in path:
                        if "trombone02" in path:
                            path_name_a = "../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img_t_05/test/trombone/trombone02_"
                            path_name_b =".jpg"
                            filepaths.append(path_name_a + str(num+second) + path_name_b)
                            num = num + 100
                        if num > num_end:
                            break
                       
                #########################################################
    return filepaths

def prepare_label(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    filepaths = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sort_string(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                filepaths.append(add_label(path))
    return filepaths

def prepare_all(audio, image, trainOrtesr):
    audios = prepare_data(audio)
    images = prepare_data(image)
    labels = prepare_label(audio)
    size = len(audios)
    
    with open('/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/' + trainOrtesr + '_zurashi_t_05_01_data.txt', 'w+') as f:
        count = 0
        for laudio, limage, llabel in zip(audios, images, labels):
            f.write(laudio + '||')
            f.write(llabel + '||')
            f.write(limage)
            count += 1
            if count < size:
                f.write('\n')


if __name__ == '__main__':

    ################violin　オンリー###############
    train_audio = "../../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/spec_t_05/train_05/trombone"
    train_image = "../../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img_t_05/train_05/trombone"
    #prepare_all(train_audio, train_image, 'train')

    test_audio = "../../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/spec_t_05/test"
    test_image = "../../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img_t_05/test"
    prepare_all(test_audio, test_image, 'test')

     
    ################violin_con　オンリー###############
    """ train_audio = "/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/spec_con/train"
    train_image = "/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img/train"
    prepare_all(train_audio, train_image, 'train') """

    """ test_audio = "/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/spec_con/test"
    test_image = "/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img/test"
    prepare_all(test_audio, test_image, 'test') """

    ################楽器ミックス###############
    """ train_audio = "../../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/spec_m/train"
    train_image = "../../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img_m/train"
    prepare_all(train_audio, train_image, 'train')


    test_audio = "../../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/spec/validation"
    test_image = "../../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/img/validation"
    prepare_all(test_audio, test_image, 'test') """
