#satoshi
import torch
import itertools
from . import networks
from .base_model import BaseModel
import random

class Pix2PixconfulladdModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['G_GAN', 'G_L1', 'D_i_real', 'D_i_fake',  'D_v_real', 'D_v_fake']
        self.visual_names = ['real_audio', 'real_image1', 'real_image2', 'real_image3', 'real_image4', 'fake_image1_1', 'fake_image1_2', 'fake_image1_3', 'fake_image1_4','fake_image2_1', 'fake_image2_2','fake_image2_3','fake_image2_4']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        if self.isTrain:
            self.model_names = ['G1', 'G2', 'D_i', 'D_v']
        else:  # during test time, only load G
            self.model_names = ['G1', 'G2']
        if not opt.no_label:
            self.loss_names.append('C_label')
            opt.input_nc = opt.input_nc + opt.label_nc
            if self.isTrain:
                self.model_names.append('C2')
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        if not opt.no_label:
            self.netC2 = networks.define_C('C2', gpu_ids=self.gpu_ids)
        self.netG1 = networks.define_G(opt.input_nc+1, opt.output_nc, opt.ngf, opt.netG1, opt.norm,
                                      not opt.no_dropout, not opt.no_attention, opt.init_type, opt.init_gain,
                                      gpu_ids=self.gpu_ids)
        ##0711変更
        self.netG2 = networks.define_G(opt.input_nc+1, opt.output_nc, opt.ngf, opt.netG2, opt.norm,
                                      not opt.no_dropout, not opt.no_attention, opt.init_type, opt.init_gain,
                                      gpu_ids=self.gpu_ids)
        """ self.netG2 = networks.define_G_3d(opt.input_nc, opt.output_nc, opt.ngf, opt.netG2, opt.norm,
                                      not opt.no_dropout, not opt.no_attention, opt.init_type, opt.init_gain,
                                      gpu_ids=self.gpu_ids) """
        if self.isTrain:  # only defined during training time
            self.netD_i = networks.define_D_i(opt.input_nc + opt.output_nc+1, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, not opt.no_attention, opt.init_type, opt.init_gain,
                                          gpu_ids=self.gpu_ids)
            self.netD_v = networks.define_D_v(20, opt.ndf, opt.netD,
                                          opt.n_layers_D, 'batch3d', not opt.no_attention, opt.init_type, opt.init_gain,
                                          gpu_ids=self.gpu_ids)
            """ print("#############")
            print(self.netD_v) """

        if self.isTrain:  # only defined during training time
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionL1 = torch.nn.L1Loss()
            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG1.parameters(), self.netG2.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_i = torch.optim.Adam(self.netD_i.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_v = torch.optim.Adam(self.netD_v.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_i)
            self.optimizers.append(self.optimizer_D_v)


            if not opt.no_label:
                self.criterionCEN = torch.nn.CrossEntropyLoss()
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

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

    def cat_label(self, input, label):
        # Replicate spatially and concatenate domain information.
        label = label.view(label.size(0), label.size(1), 1, 1)
        label = label.repeat(1, 1, input.size(2), input.size(3))
        input = torch.cat([input, label], dim=1)
        return input

    def cat_input(self, input, stage, spec_num):
        if not self.opt.no_label and stage == 2:
            # input = torch.cat((input, self.fake_image1), 1)
            # より細かく条件分け
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
            fake_image = self.netG1(self.cat_input(input_data, 1, i+1))
            fake_images_1.append(fake_image)
            
            # ラベル処理
            image_label = self.netC2(fake_image)
            setattr(self, f'fake_image1_{i+1}', fake_image)
            setattr(self, f'image_label1_{i+1}', image_label)
            setattr(self, f'res_label1_{i+1}', self.audio_label - image_label)

        # Stage 2の生成
        zeros = torch.zeros(batch_size, 1, 128, 128).to(self.device)
        for i in range(4):
            input_data = torch.cat([fake_images_1[i], zeros], dim=1)
            fake_image = self.netG2(self.cat_input(input_data, 2, i+1))
            setattr(self, f'fake_image2_{i+1}', fake_image)


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        batch_size = self.real_audio.shape[0]
        tse_masks, tsl_masks = self.time_handler.get_masks(batch_size)
        
        # Stage 1の損失計算
        loss_fake1 = []
        for i in range(4):
            fake_image = getattr(self, f'fake_image1_{i+1}')
            input_data = torch.cat([self.real_audio*tse_masks[i], tsl_masks[i]], dim=1)
            fake_combined = torch.cat([input_data, fake_image], 1)
            pred_fake = self.netD_i(self.cat_input(fake_combined, 2, i+1).detach())
            loss_fake1.append(self.criterionGAN(pred_fake, False))
        self.loss_D_fake1_1 = sum(loss_fake1) / 4

        # Stage 2の損失計算
        loss_fake2 = []
        for i in range(4):
            fake_image = getattr(self, f'fake_image2_{i+1}')
            input_data = torch.cat([self.real_audio*tse_masks[i], tsl_masks[i]], dim=1)
            fake_combined = torch.cat([input_data, fake_image], 1)
            pred_fake = self.netD_i(self.cat_input(fake_combined, 2, i+1).detach())
            loss_fake2.append(self.criterionGAN(pred_fake, False))
        self.loss_D_fake2_1 = sum(loss_fake2) / 4

        # Real画像の損失計算
        real_images = [self.real_image1, self.real_image2, self.real_image3, self.real_image4]
        real_loss = []
        for i in range(4):
            input_data = torch.cat([self.real_audio*tse_masks[i], tsl_masks[i]], dim=1)
            real_combined = torch.cat([input_data, real_images[i]], 1)
            pred_real = self.netD_i(self.cat_input(real_combined, 2, i+1))
            real_loss.append(self.criterionGAN(pred_real, True))
        self.loss_D_i_real = sum(real_loss) / 4

        # 最終的な損失計算と逆伝播
        self.loss_D_i_fake = self.loss_D_fake1_1 + self.loss_D_fake2_1
        self.loss_D = (self.loss_D_i_fake + self.loss_D_i_real) * 0.5
        self.loss_D.backward(retain_graph=True)

    def backward_D_v(self):
        """Calculate GAN loss for the video discriminator"""
        batch_size = self.real_audio.shape[0]
        tse_masks, tsl_masks = self.time_handler.get_masks(batch_size)
        
        # 時系列の順序パターン
        sequence_patterns = [
            [1,2,4,3],[1,3,2,4],[1,3,4,2],[1,4,2,3],[1,4,3,2],
            [2,1,3,4],[2,1,4,3],[2,3,1,4],[2,3,4,1],[2,4,1,3],
            [2,4,3,1],[3,1,2,4],[3,1,4,2],[3,2,1,4],[3,2,4,1],
            [3,4,1,2],[3,4,2,1],[4,1,2,3],[4,1,3,2],[4,2,1,3],
            [4,2,3,1],[4,3,1,2],[4,3,2,1]
        ]
        random_order = random.choice(sequence_patterns)
        
        # Stage 1の生成画像のシーケンス
        stage1_images = [self.fake_image1_1, self.fake_image1_2, self.fake_image1_3, self.fake_image1_4]
        stage2_images = [self.fake_image2_1, self.fake_image2_2, self.fake_image2_3, self.fake_image2_4]

        # Stage 1のFakeシーケンス（シャッフル）の損失
        fake_sequence_stage1 = []
        for idx in random_order:
            i = idx - 1
            embedded_audio = self.real_audio * tse_masks[i]
            input_combined = torch.cat([stage1_images[i], torch.cat([embedded_audio, tsl_masks[i]], dim=1)], dim=1)
            fake_sequence_stage1.append(self.cat_input(input_combined, 2, idx))
        fake_sequence_stage1 = torch.stack(fake_sequence_stage1, 2)
        loss_D_v_fake_stage1 = self.criterionGAN(self.netD_v(fake_sequence_stage1.detach()), False)

        # Stage 2のFakeシーケンス（シャッフル）の損失
        fake_sequence_stage2 = []
        for idx in random_order:
            i = idx - 1
            embedded_audio = self.real_audio * tse_masks[i]
            input_combined = torch.cat([stage2_images[i], torch.cat([embedded_audio, tsl_masks[i]], dim=1)], dim=1)
            fake_sequence_stage2.append(self.cat_input(input_combined, 2, idx))
        fake_sequence_stage2 = torch.stack(fake_sequence_stage2, 2)
        loss_D_v_fake_stage2 = self.criterionGAN(self.netD_v(fake_sequence_stage2.detach()), False)

        # Stage 1のRealシーケンス（正しい順序）損失
        real_sequence_stage1 = []
        for i in range(4):
            embedded_audio = self.real_audio * tse_masks[i]
            input_combined = torch.cat([stage1_images[i], torch.cat([embedded_audio, tsl_masks[i]], dim=1)], dim=1)
            real_sequence_stage1.append(self.cat_input(input_combined, 2, i+1))
        real_sequence_stage1 = torch.stack(real_sequence_stage1, 2)
        loss_D_v_real_stage1 = self.criterionGAN(self.netD_v(real_sequence_stage1.detach()), True)

        # Stage 2のRealシーケンス（正しい順序）損失
        real_sequence_stage2 = []
        for i in range(4):
            embedded_audio = self.real_audio * tse_masks[i]
            input_combined = torch.cat([stage2_images[i], torch.cat([embedded_audio, tsl_masks[i]], dim=1)], dim=1)
            real_sequence_stage2.append(self.cat_input(input_combined, 2, i+1))
        real_sequence_stage2 = torch.stack(real_sequence_stage2, 2)
        loss_D_v_real_stage2 = self.criterionGAN(self.netD_v(real_sequence_stage2.detach()), True)

        # 最終的な損失計算
        self.loss_D_v_fake = loss_D_v_fake_stage1 + loss_D_v_fake_stage2  # 間違った順序の損失
        self.loss_D_v_real = loss_D_v_real_stage1 + loss_D_v_real_stage2  # 正しい順序の損失
        self.loss_D_v = (self.loss_D_v_fake + self.loss_D_v_real) * 0.5
        self.loss_D_v.backward(retain_graph=True)

    def backward_C(self):
        self.image_label2 = self.netC2(self.fake_image2_1)
        index = torch.max(self.audio_label, 1)[1]
        self.loss_C_label = (self.criterionCEN(self.image_label1_1, index) + self.criterionCEN(self.image_label1_2, index) + self.criterionCEN(self.image_label1_3, index) + self.criterionCEN(self.image_label1_4, index) + self.criterionCEN(self.image_label2, index)) * 0.2
        self.loss_C_label.backward(retain_graph=True)

    def backward_G(self):
        """Calculate GAN loss for the generator"""
        batch_size = self.real_audio.shape[0]
        tse_masks, tsl_masks = self.time_handler.get_masks(batch_size)
        
        # Stage 1の損失計算
        loss_gan_stage1 = []
        loss_l1_stage1 = []
        stage1_images = [self.fake_image1_1, self.fake_image1_2, self.fake_image1_3, self.fake_image1_4]
        real_images = [self.real_image1, self.real_image2, self.real_image3, self.real_image4]
        
        for i in range(4):
            # GAN loss
            embedded_audio = self.real_audio * tse_masks[i]
            fake_combined = torch.cat([embedded_audio, tsl_masks[i]], dim=1)
            fake_input = torch.cat([fake_combined, stage1_images[i]], 1)
            pred_fake = self.netD_i(self.cat_input(fake_input, 1, i+1))
            loss_gan_stage1.append(self.criterionGAN(pred_fake, True))
            
            # L1 loss
            loss_l1_stage1.append(self.criterionL1(stage1_images[i], real_images[i]))

        # Stage 2の損失計算
        loss_gan_stage2 = []
        loss_l1_stage2 = []
        stage2_images = [self.fake_image2_1, self.fake_image2_2, self.fake_image2_3, self.fake_image2_4]
        
        for i in range(4):
            # GAN loss
            embedded_audio = self.real_audio * tse_masks[i]
            fake_combined = torch.cat([embedded_audio, tsl_masks[i]], dim=1)
            fake_input = torch.cat([fake_combined, stage2_images[i]], 1)
            pred_fake = self.netD_i(self.cat_input(fake_input, 1, i+1))
            loss_gan_stage2.append(self.criterionGAN(pred_fake, True))
            
            # L1 loss
            loss_l1_stage2.append(self.criterionL1(stage2_images[i], real_images[i]))

        # 損失の集計
        self.loss_G_L1_1 = sum(loss_l1_stage1) / 4 * self.opt.lambda_L1
        loss_G_L2_1 = sum(loss_l1_stage2) / 4 * self.opt.lambda_L1
        self.loss_G_L1 = self.loss_G_L1_1 + loss_G_L2_1

        loss_G_GAN1_1 = sum(loss_gan_stage1) / 4
        loss_G_GAN2_1 = sum(loss_gan_stage2) / 4
        self.loss_G_GAN = loss_G_GAN1_1 + loss_G_GAN2_1

        # 最終的な損失計算と逆伝播
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward(retain_graph=True)

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()  # first call forward to calculate intermediate results
        self.set_requires_grad(self.netC2, False)
        # update D
        self.set_requires_grad(self.netD_i, True)  # enable backprop for D
        self.optimizer_D_i.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D_i.step()  # update D's weights

        self.set_requires_grad(self.netD_v, True)  # enable backprop for D
        self.optimizer_D_v.zero_grad()  # set D's gradients to zero
        self.backward_D_v()  # calculate gradients for D
        self.optimizer_D_v.step()  # update D's weights

        # update G
        self.set_requires_grad(self.netD_i, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD_v, False)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_C()
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
        
class TimeSegmentHandler:
    """時間情報の埋め込みを管理するクラス"""
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

# networks.py に追加する新しいコード

class CLAPAdapter(nn.Module):
    """CLAP特徴量をネットワークの中間層に統合するためのアダプターモジュール"""
    
    def __init__(self, clap_dim=512, out_dim=256, spatial_size=(16, 16), use_bn=True):
        """
        Parameters:
            clap_dim (int): CLAP特徴量の次元数（デフォルト: 512）
            out_dim (int): 出力特徴量の次元数
            spatial_size (tuple): 出力特徴マップの空間サイズ (height, width)
            use_bn (bool): バッチ正規化を使用するかどうか
        """
        super(CLAPAdapter, self).__init__()
        
        # 全結合層でCLAP特徴量の次元を変換
        self.fc_layers = nn.Sequential(
            nn.Linear(clap_dim, out_dim),
            nn.ReLU(True)
        )
        
        if use_bn:
            self.fc_layers.add_module('bn', nn.BatchNorm1d(out_dim))
        
        self.spatial_size = spatial_size
        
    def forward(self, clap_features):
        """
        CLAP特徴量を受け取り、指定された空間サイズの特徴マップに変換
        
        Parameters:
            clap_features (tensor): [B, clap_dim] 形式のCLAP特徴量
            
        Returns:
            tensor: [B, out_dim, H, W] 形式の特徴マップ
        """
        batch_size = clap_features.size(0)
        
        # 特徴量の次元変換
        x = self.fc_layers(clap_features)
        
        # 空間次元の追加と拡張
        x = x.view(batch_size, -1, 1, 1)
        x = x.expand(-1, -1, self.spatial_size[0], self.spatial_size[1])
        
        return x


# 異なる統合ポイント用のCLAPアダプター生成関数
def define_clap_adapters(clap_dim=512, gpu_ids=[]):
    """
    Generatorの各層に対応するCLAPアダプターを生成
    
    Parameters:
        clap_dim (int): CLAP特徴量の次元数
        gpu_ids (list): GPUのID
        
    Returns:
        dict: 各統合ポイントに対応するアダプターのディクショナリ
    """
    adapters = {
        # エンコーダ中間層用
        'encoder': CLAPAdapter(clap_dim=clap_dim, out_dim=256, spatial_size=(16, 16)),
        
        # ボトルネック層用
        'bottleneck': CLAPAdapter(clap_dim=clap_dim, out_dim=512, spatial_size=(1, 1)),
        
        # デコーダ中間層用
        'decoder': CLAPAdapter(clap_dim=clap_dim, out_dim=256, spatial_size=(16, 16))
    }
    
    # GPUへの転送
    if len(gpu_ids) > 0:
        for key in adapters:
            adapters[key].to(gpu_ids[0])
            adapters[key] = torch.nn.DataParallel(adapters[key], gpu_ids)
    
    return adapters