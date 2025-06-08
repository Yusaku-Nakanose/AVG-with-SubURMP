#satoshi
import torch
import itertools
from . import networks
from .base_model import BaseModel
import random

class Pix2PixconfulladdclapModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.loss_names = ['G_GAN', 'G_L1', 'D_i_real', 'D_i_fake',  'D_v_real', 'D_v_fake']
        self.visual_names = ['real_audio', 'real_image1', 'real_image2', 'real_image3', 'real_image4', 'fake_image1_1', 'fake_image1_2', 'fake_image1_3', 'fake_image1_4','fake_image2_1', 'fake_image2_2','fake_image2_3','fake_image2_4']

        # specify the models you want to save to the disk.
        if self.isTrain:
            self.model_names = ['G1', 'G2', 'D_i', 'D_v']
        else:  # during test time, only load G
            self.model_names = ['G1', 'G2']
            
        # Classifier (netC2)の初期化を安全に行う
        try:
            self.netC2 = networks.define_C('C2', gpu_ids=self.gpu_ids)
            if self.isTrain and not opt.no_label:
                self.model_names.append('C2')
                self.loss_names.append('C_label')
        except Exception as e:
            print(f"Warning: Failed to initialize netC2: {e}")
            self.netC2 = None
        
        # ラベル情報を含めたチャネル数の計算
        if not opt.no_label:
            opt.input_nc = opt.input_nc + opt.label_nc
            
        # CLAP特徴量を統合する層の設定
        clap_layers = ['encoder', 'bottleneck', 'decoder']  # 全層に統合
        
        # 適切な入力チャネル数を計算
        # テスト結果から17チャネルが必要と判断
        effective_input_nc = 17
        
        # G1とG2をCLAP対応のジェネレーターに変更
        self.netG1 = networks.define_G(effective_input_nc, opt.output_nc, opt.ngf, 'unet_128_clap', opt.norm,
                                    not opt.no_dropout, not opt.no_attention, opt.init_type, opt.init_gain,
                                    gpu_ids=self.gpu_ids, clap_dim=512, clap_layers=clap_layers)
        self.netG2 = networks.define_G(effective_input_nc, opt.output_nc, opt.ngf, 'unet_128_clap', opt.norm,
                                    not opt.no_dropout, not opt.no_attention, opt.init_type, opt.init_gain,
                                    gpu_ids=self.gpu_ids, clap_dim=512, clap_layers=clap_layers)

        # 判別器にCLAP特徴量を使用する設定
        use_clap_discriminator = True
        
        if self.isTrain:  # only defined during training time
            self.netD_i = networks.define_D_i(opt.input_nc + opt.output_nc+1, opt.ndf, opt.netD,
                                        opt.n_layers_D, opt.norm, not opt.no_attention, opt.init_type, opt.init_gain,
                                        gpu_ids=self.gpu_ids, use_clap=use_clap_discriminator, clap_dim=512)
            self.netD_v = networks.define_D_v(20, opt.ndf, opt.netD,
                                        opt.n_layers_D, 'batch3d', not opt.no_attention, opt.init_type, opt.init_gain,
                                        gpu_ids=self.gpu_ids, use_clap=use_clap_discriminator, clap_dim=512, T=4)

        if self.isTrain:
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
        
        # CLAP特徴量を受け取る
        self.clap_features = input['clap_features'].to(self.device)

    def cat_label(self, input, label):
        # Replicate spatially and concatenate domain information.
        label = label.view(label.size(0), label.size(1), 1, 1)
        label = label.repeat(1, 1, input.size(2), input.size(3))
        input = torch.cat([input, label], dim=1)
        return input

    def cat_input(self, input, stage, spec_num):
        # デバッグ出力を追加
        # print(f"Input shape before cat_input: {input.shape}")
        
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
        
        # デバッグ出力を追加
        # print(f"Input shape after cat_input: {input.shape}")
        
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
            
            # ラベル処理
            image_label = self.netC2(fake_image)
            setattr(self, f'fake_image1_{i+1}', fake_image)
            setattr(self, f'image_label1_{i+1}', image_label)
            setattr(self, f'res_label1_{i+1}', self.audio_label - image_label)

        # Stage 2の生成
        zeros = torch.zeros(batch_size, 1, 128, 128).to(self.device)
        for i in range(4):
            input_data = torch.cat([fake_images_1[i], zeros], dim=1)
            
            # ここでもCLAP特徴量を渡す
            fake_image = self.netG2(self.cat_input(input_data, 2, i+1), self.clap_features)
            setattr(self, f'fake_image2_{i+1}', fake_image)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        batch_size = self.real_audio.shape[0]
        tse_masks, tsl_masks = self.time_handler.get_masks(batch_size)
        
        # Stage 1の損失計算
        loss_fake1 = []
        for i in range(4):
            fake_image = getattr(self, f'fake_image1_{i+1}')
            # 音声×TSEマスク と TSLマスク を結合
            input_data = torch.cat([self.real_audio * tse_masks[i], tsl_masks[i]], dim=1)
            # マスク付き音声＋偽画像 を結合
            fake_combined = torch.cat([input_data, fake_image], dim=1)
            # CLAP特徴量も渡して判別器へ（detachで勾配を停止）
            pred_fake = self.netD_i(
                self.cat_input(fake_combined, 2, i+1).detach(),
                self.clap_features
            )
            loss_fake1.append(self.criterionGAN(pred_fake, False))
        self.loss_D_fake1_1 = sum(loss_fake1) / 4

        # Stage 2の損失計算（CLAP特徴量を渡す）
        loss_fake2 = []
        for i in range(4):
            fake_image = getattr(self, f'fake_image2_{i+1}')
            input_data = torch.cat([self.real_audio * tse_masks[i], tsl_masks[i]], dim=1)
            fake_combined = torch.cat([input_data, fake_image], dim=1)
            pred_fake = self.netD_i(
                self.cat_input(fake_combined, 2, i+1).detach(),
                self.clap_features
            )
            loss_fake2.append(self.criterionGAN(pred_fake, False))
        self.loss_D_fake2_1 = sum(loss_fake2) / 4

        # Real画像の損失計算（CLAP特徴量を渡す）
        real_images = [self.real_image1, self.real_image2, self.real_image3, self.real_image4]
        real_loss = []
        for i, real_img in enumerate(real_images):
            input_data = torch.cat([self.real_audio * tse_masks[i], tsl_masks[i]], dim=1)
            real_combined = torch.cat([input_data, real_img], dim=1)
            pred_real = self.netD_i(
                self.cat_input(real_combined, 2, i+1),
                self.clap_features
            )
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
        
        # 時系列の順序パターン（24通り）
        sequence_patterns = [
            [1,2,4,3],[1,3,2,4],[1,3,4,2],[1,4,2,3],[1,4,3,2],
            [2,1,3,4],[2,1,4,3],[2,3,1,4],[2,3,4,1],[2,4,1,3],
            [2,4,3,1],[3,1,2,4],[3,1,4,2],[3,2,1,4],[3,2,4,1],
            [3,4,1,2],[3,4,2,1],[4,1,2,3],[4,1,3,2],[4,2,1,3],
            [4,2,3,1],[4,3,1,2],[4,3,2,1]
        ]
        random_order = random.choice(sequence_patterns)
        
        # Stage 1 / Stage 2 の生成画像リスト
        stage1_images = [self.fake_image1_1, self.fake_image1_2, self.fake_image1_3, self.fake_image1_4]
        stage2_images = [self.fake_image2_1, self.fake_image2_2, self.fake_image2_3, self.fake_image2_4]

        # --- Fake シーケンス（シャッフル順）の損失 ---
        # Stage 1
        fake_sequence_stage1 = []
        for idx in random_order:
            i = idx - 1
            embedded_audio = self.real_audio * tse_masks[i]
            input_combined = torch.cat(
                [stage1_images[i], torch.cat([embedded_audio, tsl_masks[i]], dim=1)],
                dim=1
            )
            fake_sequence_stage1.append(self.cat_input(input_combined, 2, idx))
        fake_sequence_stage1 = torch.stack(fake_sequence_stage1, dim=2)
        # CLAP特徴量を判別器に渡して「偽」として損失計算
        loss_D_v_fake_stage1 = self.criterionGAN(
            self.netD_v(fake_sequence_stage1.detach(), self.clap_features),
            False
        )
        
        # Stage 2
        fake_sequence_stage2 = []
        for idx in random_order:
            i = idx - 1
            embedded_audio = self.real_audio * tse_masks[i]
            input_combined = torch.cat(
                [stage2_images[i], torch.cat([embedded_audio, tsl_masks[i]], dim=1)],
                dim=1
            )
            fake_sequence_stage2.append(self.cat_input(input_combined, 2, idx))
        fake_sequence_stage2 = torch.stack(fake_sequence_stage2, dim=2)
        loss_D_v_fake_stage2 = self.criterionGAN(
            self.netD_v(fake_sequence_stage2.detach(), self.clap_features),
            False
        )

        # --- Real シーケンス（正しい順）の損失 ---
        # Stage 1
        real_sequence_stage1 = []
        for i in range(4):
            embedded_audio = self.real_audio * tse_masks[i]
            input_combined = torch.cat(
                [stage1_images[i], torch.cat([embedded_audio, tsl_masks[i]], dim=1)],
                dim=1
            )
            real_sequence_stage1.append(self.cat_input(input_combined, 2, i+1))
        real_sequence_stage1 = torch.stack(real_sequence_stage1, dim=2)
        loss_D_v_real_stage1 = self.criterionGAN(
            self.netD_v(real_sequence_stage1.detach(), self.clap_features),
            True
        )
        
        # Stage 2
        real_sequence_stage2 = []
        for i in range(4):
            embedded_audio = self.real_audio * tse_masks[i]
            input_combined = torch.cat(
                [stage2_images[i], torch.cat([embedded_audio, tsl_masks[i]], dim=1)],
                dim=1
            )
            real_sequence_stage2.append(self.cat_input(input_combined, 2, i+1))
        real_sequence_stage2 = torch.stack(real_sequence_stage2, dim=2)
        loss_D_v_real_stage2 = self.criterionGAN(
            self.netD_v(real_sequence_stage2.detach(), self.clap_features),
            True
        )

        # --- 最終的な損失と逆伝播 ---
        self.loss_D_v_fake = loss_D_v_fake_stage1 + loss_D_v_fake_stage2  # Fake（シャッフル順）
        self.loss_D_v_real = loss_D_v_real_stage1 + loss_D_v_real_stage2  # Real（正しい順）
        self.loss_D_v = 0.5 * (self.loss_D_v_fake + self.loss_D_v_real)
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