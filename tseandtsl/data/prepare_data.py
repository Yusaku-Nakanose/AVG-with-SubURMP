# car-ganのオリジナルコード

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
    image = []

    with open(dir, 'r') as f:
        for line in f.readlines():
            line_list = line.split('||')
            audio.append(line_list[0])
            label.append(np.fromstring(line_list[1], dtype=float, sep=','))
            image.append(line_list[2].split('\n')[0])
    return {'audio': audio[:min(max_dataset_size, len(audio))],
            'label': label[:min(max_dataset_size, len(label))],
            'image': image[:min(max_dataset_size, len(image))]}


def prepare_data(dir, ins, instrument, max):
    global i, j
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    filepaths = []
    #num = 10000
    num = 500
    num_end = max
    second = 100
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sort_string(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)

                #########concate使う時に数が合わなくなるから修正(6/20)
                if "spec" in path:
                    if "train" in path:
                        path_name_a = "../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/spec_"+ ins + "/train/" + instrument + "_"
                        #path_name_a = "../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/spec/train/trumpet00_"
                        
                        path_name_b =".png"
                        filepaths.append(path_name_a + str(num) + path_name_b)
                        num = num + 100
                    if num > num_end:
                        break
                            
                    elif "test" in path:
                        path_name_a = "../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/spec_"+ ins + "/test/" + instrument + "_"
                        path_name_b =".png"
                        filepaths.append(path_name_a + str(num) + path_name_b)
                        num = num + 100
                    if num > num_end:
                        break
                            
                    
                if "img" in path:
                    if "train" in path:
                        path_name_a = "../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img_"+ ins + "/train/" + instrument + "_"
                        #path_name_a = "../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/img/train/trumpet00_"
                        path_name_b =".jpg"
                        filepaths.append(path_name_a + str(num) + path_name_b)
                        num = num + 100
                    if num > num_end:
                        break
                            
                    elif "test" in path:
                        path_name_a = "../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img_"+ ins + "/test/" + instrument + "_"
                        path_name_b =".jpg"
                        filepaths.append(path_name_a + str(num) + path_name_b)
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

def prepare_all(audio, image, trainOrtesr, ins, instrument, txt_name, max):
    audios = prepare_data(audio, ins, instrument, max)
    images = prepare_data(image, ins, instrument, max)
    labels = prepare_label(audio)
    size = len(audios)
    
    #この"00"はずらし秒数
    with open('/mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/' + txt_name + trainOrtesr + '_zurashi_'+ ins + '_00_data_kari.txt', 'w+') as f:
        count = 0
        for laudio, limage, llabel in zip(audios, images, labels):
            f.write(laudio + '||')
            f.write(llabel + '||')
            f.write(limage)
            count += 1
            if count < size:
                f.write('\n')


if __name__ == '__main__':
    #画像枚数も変更する
    ################violin　オンリー###############
    """ ins = "s_00"
    instrument = "sax00"
    txt_name = "sax_txt/"
    max = 101900
    train_audio = "../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/spec_" + ins + "/train"
    train_image = "../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img_" + ins + "/train"
    prepare_all(train_audio, train_image, 'train', ins, instrument, txt_name, max) """

    ins = "tu_04"
    instrument = "tuba03"
    txt_name = "tuba_txt/"
    max = 254400
    train_audio = "../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/spec_" + ins + "/train/"
    train_image = "../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img_" + ins + "/train/"
    prepare_all(train_audio, train_image, 'train', ins, instrument, txt_name, max)

    test_audio = "../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/spec_" + ins + "/test"
    test_image = "../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img_" + ins + "/test"
    #prepare_all(test_audio, test_image, 'test')
    #prepare_all(test_audio, test_image, 'test', ins, instrument, txt_name, max)

     
    