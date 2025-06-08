from .base_model import BaseModel
from . import networks
import torch

class TestFullAddClapModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode spectram'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestFullAddClapModel cannot be used during training time'
        parser.set_defaults(dataset_mode='spectram')
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

        return parser

    def __init__(self, opt):

        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_audio', 'real_image1', 'real_image2', 'real_image3', 'real_image4', 
                           'fake_image1_1', 'fake_image1_2', 'fake_image1_3', 'fake_image1_4',
                           'fake_image2_1', 'fake_image2_2','fake_image2_3','fake_image2_4']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G1', 'G2']
        
        # Classifier (netC2)の初期化を安全に行う
        if not opt.no_label:
            try:
                self.netC2 = networks.define_C('C2', gpu_ids=self.gpu_ids)
                self.model_names.append('C2')
                opt.input_nc = opt.input_nc + opt.label_nc
            except Exception as e:
                print(f"Warning: Failed to initialize netC2: {e}")
                self.netC2 = None
        
        # CLAP特徴量を統合する層の設定
        clap_layers = ['encoder', 'bottleneck', 'decoder']  # 全層に統合
        
        # 適切な入力チャネル数を計算（学習コードと同じ17チャネル）
        effective_input_nc = 17
        
        # G1とG2をCLAP対応のジェネレーターに変更
        self.netG1 = networks.define_G(effective_input_nc, opt.output_nc, opt.ngf, 'unet_128_clap', opt.norm,
                                      not opt.no_dropout, not opt.no_attention, opt.init_type, opt.init_gain,
                                      gpu_ids=self.gpu_ids, clap_dim=512, clap_layers=clap_layers)
        self.netG2 = networks.define_G(effective_input_nc, opt.output_nc, opt.ngf, 'unet_128_clap', opt.norm,
                                      not opt.no_dropout, not opt.no_attention, opt.init_type, opt.init_gain,
                                      gpu_ids=self.gpu_ids, clap_dim=512, clap_layers=clap_layers)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG1', self.netG1)  # store netG1 in self.
        setattr(self, 'netG2', self.netG2)  # store netG2 in self.
        if hasattr(self, 'netC2') and self.netC2 is not None:
            setattr(self, 'netC2', self.netC2)  # store netC2 in self.
        
        # TimeSegmentHandlerの追加
        self.time_handler = TimeSegmentHandler(self.device)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.real_audio = input['audio'].to(self.device)
        self.real_image1 = input['image1'].to(self.device)
        self.real_image2 = input['image2'].to(self.device)
        self.real_image3 = input['image3'].to(self.device)
        self.real_image4 = input['image4'].to(self.device)
        self.audio_label = input['label'].to(self.device)
        self.audio_paths = input['path']
        
        # CLAP特徴量を受け取る
        self.clap_features = input['clap_features'].to(self.device)

    def cat_label(self, input, label):
        # Replicate spatially and concatenate domain information.
        label = label.view(label.size(0), label.size(1), 1, 1)
        label = label.repeat(1, 1, input.size(2), input.size(3))
        input = torch.cat([input, label], dim=1)
        return input

    def cat_input(self, input, stage, spec_num):
        """学習コードと同じロジックを使用"""
        if not self.opt.no_label and stage == 2:
            if spec_num == 1:
                input = self.cat_label(input, self.res_label1_1)
            elif spec_num == 2:
                input = self.cat_label(input, self.res_label1_2)
            elif spec_num == 3:
                input = self.cat_label(input, self.res_label1_3)
            elif spec_num == 4:
                input = self.cat_label(input, self.res_label1_4)
            else:
                input = self.cat_label(input, self.res_label1_1)
        if not self.opt.no_label and stage == 1:
            input = self.cat_label(input, self.audio_label)
        return input

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        batch_size = self.real_audio.shape[0]
        tse_masks, tsl_masks = self.time_handler.get_masks(batch_size)
        
        # Stage 1の生成とラベル処理をまとめて実行
        fake_images_1 = []
        for i in range(4):
            # 時間情報を考慮した画像生成
            input_data = torch.cat([self.real_audio*tse_masks[i], tsl_masks[i]], dim=1)
            
            # CLAP特徴量を渡して画像生成
            fake_image = self.netG1(self.cat_input(input_data, 1, i+1), self.clap_features)
            fake_images_1.append(fake_image)
            
            # ラベル処理（netC2が利用可能な場合のみ）
            if hasattr(self, 'netC2') and self.netC2 is not None:
                image_label = self.netC2(fake_image)
                setattr(self, f'image_label1_{i+1}', image_label)
                setattr(self, f'res_label1_{i+1}', self.audio_label - image_label)
            else:
                # netC2が利用できない場合のフォールバック
                setattr(self, f'res_label1_{i+1}', self.audio_label)
                
            setattr(self, f'fake_image1_{i+1}', fake_image)

        # Stage 2の生成
        zeros = torch.zeros(batch_size, 1, 128, 128).to(self.device)
        for i in range(4):
            input_data = torch.cat([fake_images_1[i], zeros], dim=1)
            
            # ここでもCLAP特徴量を渡す
            fake_image = self.netG2(self.cat_input(input_data, 2, i+1), self.clap_features)
            setattr(self, f'fake_image2_{i+1}', fake_image)

    def optimize_parameters(self):
        """No optimization for test model."""
        pass

class TimeSegmentHandler:
    """時間情報の埋め込みを管理するクラス（学習コードと同じ）"""
    def __init__(self, device):
        self.device = device
        self.segments = [
            (12, 39),  # segment 1
            (38, 64),  # segment 2
            (63, 89),  # segment 3
            (88, 114)  # segment 4
        ]

    def get_masks(self, batch_size):
        """TSEとTSLのマスクを一括生成"""
        tse_masks = []
        tsl_masks = []
        
        for start_idx, end_idx in self.segments:
            # TSE mask
            tse_mask = torch.ones(128, 128).to(self.device)
            tse_mask[:, start_idx:end_idx] = 2
            tse_masks.append(tse_mask)
            
            # TSL mask
            tsl_mask = torch.zeros(batch_size, 1, 128, 128).to(self.device)
            tsl_mask[:, :, :, start_idx:end_idx] = 1
            tsl_masks.append(tsl_mask)
            
        return tse_masks, tsl_masks