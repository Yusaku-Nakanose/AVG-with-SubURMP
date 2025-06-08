# 卒論やSSIIでの実験(Sub-URMP)で使用

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


def prepare_data(dir, im_all):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    filepaths = []
    cnt = 0
    for root, _, fnames in sorted(os.walk(dir)):
        #print(len(fnames))
        for fname in sort_string(fnames):
            #print(fname)
            if is_image_file(fname):
                path = os.path.join(root, fname)
                if cnt < 0:
                    pass
                elif cnt > im_all:
                    pass
                else:
                    filepaths.append(path)
            else:
                print("koko")
            cnt = cnt + 1
    return filepaths

def prepare_imagedata(dir, im_all):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    filepaths1 = []
    filepaths2 = []
    filepaths3 = []
    filepaths4 = []
    cnt = 0
    #cnt_all = 1015
    for root, _, fnames in sorted(os.walk(dir)):
        #print(len(fnames))
        for fname in sort_string(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                if cnt <= 0:
                    pass
                elif cnt > im_all:
                    pass
                elif cnt == im_all-1: #1304
                    filepaths4.append(path)
                elif cnt == im_all-2: #1303
                    filepaths3.append(path)
                    filepaths4.append(path)
                elif cnt == im_all-3: #1302
                    filepaths2.append(path)
                    filepaths3.append(path)
                    filepaths4.append(path)
                elif cnt == 1:
                    filepaths1.append(path)
                elif cnt == 2:
                    filepaths1.append(path)
                    filepaths2.append(path)
                elif cnt == 3:
                    filepaths1.append(path)
                    filepaths2.append(path)
                    filepaths3.append(path)
                else:
                    filepaths1.append(path)
                    filepaths2.append(path)
                    filepaths3.append(path)
                    filepaths4.append(path)
                cnt = cnt + 1
                #filepaths.append(path)
    return filepaths1, filepaths2, filepaths3, filepaths4

def prepare_label(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    filepaths = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sort_string(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                filepaths.append(add_label(path))
    return filepaths

def prepare_all(audio, image, trainOrtesr, instrument, inst, im_all):
    im_all = im_all
    audios = prepare_data(audio, im_all)
    #print(audios)

    images1,images2,images3,images4 = prepare_imagedata(image, im_all)
    #print(images4)
    labels = prepare_label(audio)
    size = len(audios)
    with open('../../../../../mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/' + instrument + '_txt/' + trainOrtesr + '_' + inst + '_full_3d_data_.txt', 'w+') as f:
        count = 0
        for laudio, limage1,limage2,limage3,limage4, llabel in zip(audios, images1, images2, images3, images4, labels):
            f.write(laudio + '||')
            f.write(llabel + '||')
            f.write(limage1 + '||')
            f.write(limage2 + '||')
            f.write(limage3 + '||')
            f.write(limage4)
            count += 1
            if count < size:
                f.write('\n')


if __name__ == '__main__':

    
    instrument = "cello"
    inst = "c"
    num = "_02"
    im_all = 9800
    train_audio = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/spec_" + inst + num + "/train/"
    train_image = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img_" + inst + num + "/train/"
    prepare_all(train_audio, train_image, 'train', instrument, inst, im_all)

   
    # cargan/spec_  kaeru
    # cargan/img_   kaeru
    # with open cargan/   kaeru
    # test_audio = "../../../../mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/spec_d_00/test/"
    # test_image = "../../../../mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img_d_00/test/"

    
    # prepare_all(test_audio, test_image, 'test')
