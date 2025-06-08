from .base_model import BaseModel
from . import networks
import torch

class TestFullAddModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        parser.set_defaults(dataset_mode='spectram')
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

        return parser

    def __init__(self, opt):

        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_audio', 'real_image1', 'real_image2', 'real_image3', 'real_image4', 'fake_image1_1', 'fake_image1_2', 'fake_image1_3', 'fake_image1_4','fake_image2_1', 'fake_image2_2','fake_image2_3','fake_image2_4']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G1', 'G2', 'C2']
        if not opt.no_label:
            self.netC2 = networks.define_C('C2', gpu_ids=self.gpu_ids)
            opt.input_nc = opt.input_nc + opt.label_nc
        self.netG1 = networks.define_G(opt.input_nc+1, opt.output_nc, opt.ngf, opt.netG1, opt.norm,
                                      not opt.no_dropout, not opt.no_attention, opt.init_type, opt.init_gain,
                                      gpu_ids=self.gpu_ids)
        self.netG2 = networks.define_G(opt.input_nc+1, opt.output_nc, opt.ngf, opt.netG2, opt.norm,
                                      not opt.no_dropout, not opt.no_attention, opt.init_type, opt.init_gain,
                                      gpu_ids=self.gpu_ids)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG1', self.netG1)  # store netG in self.
        setattr(self, 'netG2', self.netG2)  # store netG in self.
        setattr(self, 'netC2', self.netC2)
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        """ self.real_audio = input['audio'].to(self.device)
        self.real_image = input['image'].to(self.device)
        self.audio_label = input['label'].to(self.device)
        self.audio_paths = input['path'] """
        self.real_audio = input['audio'].to(self.device)
        self.real_image1 = input['image1'].to(self.device)
        self.real_image2 = input['image2'].to(self.device)
        self.real_image3 = input['image3'].to(self.device)
        self.real_image4 = input['image4'].to(self.device)
        self.audio_label = input['label'].to(self.device)
        self.audio_paths = input['path']

    def cat_label(self, input, label):
        # Replicate spatially and concatenate domain information.
        label = label.view(label.size(0), label.size(1), 1, 1)
        label = label.repeat(1, 1, input.size(2), input.size(3))
        input = torch.cat([input, label], dim=1)
        return input

    def cat_input(self, input, stage):
        if not self.opt.no_label and stage == 2:
            # input = torch.cat((input, self.fake_image1), 1)
            input = self.cat_label(input, self.res_label)
        if not self.opt.no_label and stage == 1:
            input = self.cat_label(input, self.audio_label)
        return input


    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        """ self.fake_image1 = self.netG1(self.cat_input(self.real_audio, 1)) #stage1
        self.image_label = self.netC2(self.fake_image1)
        self.res_label = self.audio_label - self.image_label
        fake_image = torch.cat((self.fake_image1, self.real_audio), 1)
        self.fake_image2 = self.netG2(self.cat_input(fake_image, 2)) #stage2 """
        

        #追加
        tsemask1 = torch.ones(128, 128)
        tsemask2 = torch.ones(128, 128)
        tsemask3 = torch.ones(128, 128)
        tsemask4 = torch.ones(128, 128)
        tsemask1[:,12:39] = 2 #13～39
        tsemask2[:,38:64] = 2
        tsemask3[:,63:89] = 2
        tsemask4[:,88:114] = 2
        tsemask1 = tsemask1.to(self.device)
        tsemask2 = tsemask2.to(self.device)
        tsemask3 = tsemask3.to(self.device)
        tsemask4 = tsemask4.to(self.device)


        #バッチ数分の数にする，1行目
        mask1 = torch.zeros(1, 1, 128, 128)
        mask2 = torch.zeros(1, 1,128, 128)
        mask3 = torch.zeros(1, 1,128, 128)
        mask4 = torch.zeros(1, 1,128, 128)
        mask1[:, :, :,12:39] = 1 #13～39
        mask2[:, :, :,38:64] = 1
        mask3[:, :, :,63:89] = 1
        mask4[:, :, :,88:114] = 1
        mask1 = mask1.to(self.device)
        mask2 = mask2.to(self.device)
        mask3 = mask3.to(self.device)
        mask4 = mask4.to(self.device)

        #追加
        self.fake_image1_1 = self.netG1(self.cat_input(torch.cat([self.real_audio*tsemask1, mask1], dim=1), 1))
        self.fake_image1_2 = self.netG1(self.cat_input(torch.cat([self.real_audio*tsemask2, mask2], dim=1), 1))
        self.fake_image1_3 = self.netG1(self.cat_input(torch.cat([self.real_audio*tsemask3, mask3], dim=1), 1))
        self.fake_image1_4 = self.netG1(self.cat_input(torch.cat([self.real_audio*tsemask4, mask4], dim=1), 1))


        #全種類ではなく一枚だけで分類するとふわっとしたラベルになるのかも？
        self.image_label = self.netC2(self.fake_image1_1)
        self.res_label = self.audio_label - self.image_label

        self.fake_image2_1 = self.netG2(self.cat_input(torch.cat([self.fake_image1_1, torch.zeros(1, 1, 128, 128).to(self.device)],dim=1), 2))
        self.fake_image2_2 = self.netG2(self.cat_input(torch.cat([self.fake_image1_2, torch.zeros(1, 1, 128, 128).to(self.device)],dim=1), 2))
        self.fake_image2_3 = self.netG2(self.cat_input(torch.cat([self.fake_image1_3, torch.zeros(1, 1, 128, 128).to(self.device)],dim=1), 2))
        self.fake_image2_4 = self.netG2(self.cat_input(torch.cat([self.fake_image1_4, torch.zeros(1, 1, 128, 128).to(self.device)],dim=1), 2))

    def optimize_parameters(self):
        """No optimization for test model."""
        pass
