from PIL import Image
from .base_dataset import BaseDataset, get_params, get_transform
from data.prepare_data_3d import make_dataset
import torch


class SpectramDataset(BaseDataset):

    def __init__(self, opt):
        dataset = make_dataset(opt.data_dir, opt.max_dataset_size)
        self.audio_paths = dataset['audio']
        self.image_paths1 = dataset['image1']
        self.image_paths2 = dataset['image2']
        self.image_paths3 = dataset['image3']
        self.image_paths4 = dataset['image4']
        self.labels = dataset['label']
        self.opt = opt

    def __getitem__(self, index):
        ### input A (real audio)
        audio_path = self.audio_paths[index]
        #print(audio_path)
        #print(index)
        spectrogram = Image.open(audio_path).convert('RGB')
        params_spectrogram = get_params(self.opt, spectrogram.size)
        transform_image = get_transform(self.opt, params_spectrogram)
        audio_tensor = transform_image(spectrogram)

        ### input B (real images)
        image_path = self.image_paths1[index]
        image = Image.open(image_path).convert('RGB')
        params = get_params(self.opt, image.size)
        transform_image = get_transform(self.opt, params)
        image_tensor1 = transform_image(image)
        #print(image_path)

        """ print("----------------")
        print(self.image_paths2) """

        image_path = self.image_paths2[index]
        image = Image.open(image_path).convert('RGB')
        params = get_params(self.opt, image.size)
        transform_image = get_transform(self.opt, params)
        image_tensor2 = transform_image(image)
        #print(image_path)

        image_path = self.image_paths3[index]
        image = Image.open(image_path).convert('RGB')
        params = get_params(self.opt, image.size)
        transform_image = get_transform(self.opt, params)
        image_tensor3 = transform_image(image)

        image_path = self.image_paths4[index]
        image = Image.open(image_path).convert('RGB')
        params = get_params(self.opt, image.size)
        transform_image = get_transform(self.opt, params)
        image_tensor4 = transform_image(image)

        input_dict = {'audio': audio_tensor, 'image1': image_tensor1, 'image2': image_tensor2, 'image3': image_tensor3, 'image4': image_tensor4, 'path': audio_path}

        if not self.opt.no_label:
            label = self.labels[index]
            label_tensor = torch.FloatTensor(label)
            input_dict.__setitem__('label', label_tensor)

        return input_dict

    def __len__(self):
        return len(self.audio_paths)

    def name(self):
        return 'SpectramDataset'
