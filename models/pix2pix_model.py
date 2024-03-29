import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from torch.autograd import Variable
from . import networks
import pdb
import numpy as np
from skimage.measure import compare_ssim as ssim
import pytorch_ssim
class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(which_model_netG='unet_256')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.opt = opt
	# specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'G_rmse', 'G_rel', 'G_t_1', 'G_t_2', 'G_t_3']
        self.loss_names_plt = ['G_GAN', 'G_L1', 'D_real', 'D_fake']#, 'G_scale', 'G_hu', 'G_berHu', 'G_rmse', 'G_rel', 'G_t_1', 'G_t_2', 'G_t_3']
        if(self.opt.loss2==1):
            self.loss_names.append('G_scale')
            self.loss_names_plt.append('G_scale')
        elif(self.opt.loss2==2):
            self.loss_names.append('G_hu')
            self.loss_names_plt.append('G_hu')
        elif(self.opt.loss2==3):
            self.loss_names.append('G_berHu')
            self.loss_names_plt.append('G_berHu')
        elif(self.opt.loss2==4):
            self.loss_names.append('G_L2')
            self.loss_names_plt.append('G_L2')
        #self.loss_names = ['G_GAN', 'G_L1', 'G_scale', 'G_hu', 'G_rev', 'G_rmse', 'G_abs', 'G_loss_sqrel', 'G_t_1', 'G_t_2', 'G_t_3']
	# specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt, opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        self.upsample = torch.nn.UpsamplingBilinear2d(size = (460, 345))
        self.criterionL1 = torch.nn.L1Loss()
        
        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            if(torch.__version__ == '0.3.0.post4'):
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        # self.set_ratio(opt.loadSize)
    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        if(torch.__version__ == '0.3.0.post4'):
            input_A = input['A' if AtoB else 'B']
            input_B = input['B' if AtoB else 'A']
            if len(self.gpu_ids) > 0:
                input_A = input_A.cuda(self.gpu_ids[0], async=True)
                input_B = input_B.cuda(self.gpu_ids[0], async=True)
            self.input_A = input_A
            self.input_B = input_B
        else:
            self.real_A = input['A' if AtoB else 'B'].to(self.device)
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def forward(self):
        if(torch.__version__ == '0.3.0.post4'):
            self.real_A = Variable(self.input_A)
            self.fake_B = self.netG(self.real_A)
            self.real_B = Variable(self.input_B)
        else:            
            self.fake_B = self.netG(self.real_A)

    def test_forward(self):
        if(torch.__version__ == '0.3.0.post4'):
            self.real_A = Variable(self.input_A, volatile=True)
            self.fake_B = self.netG(self.real_A)
            self.real_B = Variable(self.input_B, volatile=True)
        else:
            self.fake_B = self.netG(self.real_A)
        
        # self.real_A = self.upsample(self.real_A)
        # self.real_B = self.upsample(self.real_B)
        self.fake_B = self.upsample(self.fake_B)
    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    # def log_func(self, depth):

    #     if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
    #         scalar = Variable(torch.cuda.FloatTensor([1.0 / 255.0]))
    #         depth = torch.max(depth, scalar.expand_as(depth))
    #     else:
    #         scalar = torch.FloatTensor([1.0 / 255.0])
    #         depth = torch.max(depth, scalar.expand_as(depth))
    #         return 0.179581 * torch.log(depth) + 1

# def LogDepth(depth):
#     depth = np.maximum(depth, 1.0 / 255.0)
# 	return 0.179581 * np.log(depth) + 1

# def ScaleInvariantMeanSquaredError(output, gt):
#     output = LogDepth(output / 10.0) * 10.0 # division is because he had passed inputs multiplied by 10
# 	gt = LogDepth(gt / 10.0) * 10.0
# 	d = output - gt
# 	diff = np.mean(d * d)

# 	relDiff = (d.sum() * d.sum()) / float(d.size * d.size)
# 	return diff - relDiff


    # def scale_loss(self):
    #     d = 10*self.log_func((self.fake_B+1.0)/2) - 10*self.log_func((self.real_B+1.0)/2) # see util.util tensor2im
    #     t1 = d - d.mean()
    #     t2 = (t1*t1).mean()
    #     return t2

    def RootMeanSquaredError(self, imgGT=None, imgOut=None):
        
        if imgGT is not None:
            d = (imgOut - imgGT)*10.0/2
        else: 
            d = (self.fake_B - self.real_B)*80.0/2 # see util.util tensor2im, 10 is to scale to meters, not needed in cases its canceled out
       
        diff = torch.sqrt(torch.mean(d * d))
        return diff

    def AbsoluteRelativeDifference(self, imgGT=None, imgOut=None):
        if imgGT is not None:
            output = (imgOut + 1.0 + 1e-5)/2
            gt = (imgGT + 1.0 + 1e-5)/2
        else:
            output = (self.fake_B + 1.0)/2
            gt = (self.real_B + 1.0)/2
        diff = torch.mean(torch.abs(output - gt) / gt)
        return diff

    def SquaredRelativeDifference(self):
        output = (self.fake_B + 1.0)/2
        gt = (self.real_B + 1.0)/2
        d = output - gt
        diff = torch.mean((d * d) / gt)
        return diff

    def Threshold(self, threshold, imgGT=None, imgOut=None):
        if imgGT is not None:
            gt, output = (imgGT.data.cpu().numpy()+ 1.0)/2, (imgOut.data.cpu().numpy()+ 1.0)/2
        else:
            gt, output = (self.fake_B.data.cpu().numpy() + 1.0)/2, (self.real_B.data.cpu().numpy() + 1.0)/2

        output = np.maximum(output, 1.0 / 255.0) # remove elements = 0
        gt = np.maximum(gt, 1.0 / 255.0)
        output = output*10.0 #conversion to meters "needed"
        gt = gt*10.0
        bb = np.maximum(output / gt, gt / output)
        kk = np.argwhere(np.maximum(output / gt, gt / output) < threshold)
        withinThresholdCount = len(kk)
        return (withinThresholdCount / float(gt.size))


    # def LogDepth(depth):
    #     depth = np.maximum(depth, 1.0 / 255.0)	
    # 	return 0.179581 * np.log(depth) + 1

    # def RootMeanSquaredErrorLog(output, gt):
    #     d = LogDepth(output / 10.0) * 10.0 - LogDepth(gt / 10.0) * 10.0
    #     diff = np.sqrt(np.mean(d * d))
    #     return diff
    def hu_loss(self):
        output = (self.fake_B + 1.0)*10.0/2
        gt = (self.real_B + 1.0)*10.0/2
        absAlpha = torch.abs(output - gt)
        delta = torch.max(absAlpha)*.2
        diffAlpha = torch.min(absAlpha, delta)
        error = diffAlpha*( 2*absAlpha - diffAlpha)*.5
        return error.mean()

    def berHu_loss(self):
        output = (self.fake_B + 1.0)*10.0/2
        gt = (self.real_B + 1.0)*10.0/2

        absAlpha = torch.abs(output - gt)
        delta = torch.max(absAlpha)*.2
        diffAlpha = torch.max(absAlpha, delta)
        error = torch.abs(diffAlpha*( 2*absAlpha - diffAlpha)*.5)
        return error.mean()

    def SSIM(self):
        output = (self.fake_B + 1.0)/2
        gt = (self.real_B + 1.0)/2
        ssim_loss = pytorch_ssim.SSIM(window_size = 11)
        error = ssim_loss(gt, output).data.cpu().numpy()#ssim(gt, output, data_range=1, multichannel=True)
        return error

    def L1(self):
        return self.criterionL1(self.fake_B, self.real_B)

    def L2(self):
        return torch.nn.functional.mse_loss(self.fake_B, self.real_B)

    def findEvalLosses(self):
        # self.real_A = self.upsample(self.real_A)
        # self.real_B = self.upsample(self.real_B)
        # self.fake_B = self.upsample(self.fake_B)
        # --------- eval losses ------ no lambda ------------------------
        self.loss_G_rmse = np.array(self.RootMeanSquaredError().data)
        self.loss_G_rel = np.array(self.AbsoluteRelativeDifference().data)
        # self.loss_G_loss_sqrel = np.array(self.SquaredRelativeDifference().data)
        self.loss_G_t_1 = self.Threshold(1.25)
        self.loss_G_t_2 = self.Threshold(1.25*1.25)
        self.loss_G_t_3 = self.Threshold(1.25*1.25*1.25)
        self.loss_ssim = self.SSIM()
        self.evalLosses = [self.loss_G_rmse, self.loss_G_rel, self.loss_G_t_1, self.loss_G_t_2, self.loss_G_t_3, self.loss_ssim]
        self.evalLossesNames = ['RMSE', 'Rel', 'Thresh1', 'Thresh2', 'Thresh3', 'SSIM']
        # -----------------------------------------------------------------

    def findCustomLosses(self, imgOut, imgGT):
        try:
            a, b = self.real_B, self.fake_B
        except:
            a, b = 0, 0
        imgGT = 2*Variable(torch.from_numpy(imgGT)).float()/255.0 - 1
        imgOut = 2*Variable(torch.from_numpy(imgOut)).float()/255.0 - 1
        if torch.cuda.is_available():
            imgGT, imgOut = imgGT.cuda(), imgOut.cuda() 
        loss_G_rmse = np.array(self.RootMeanSquaredError(imgGT, imgOut).data)
        loss_G_rel = np.array(self.AbsoluteRelativeDifference(imgGT, imgOut).data)
        loss_G_t_1 = self.Threshold(1.25, imgGT, imgOut)
        loss_G_t_2 = self.Threshold(1.25*1.25, imgGT, imgOut)
        loss_G_t_3 = self.Threshold(1.25*1.25*1.25, imgGT, imgOut)
        loss_ssim = self.ssim()

        return [loss_G_rmse, loss_G_rel, loss_G_t_1, loss_G_t_2, loss_G_t_3, loss_ssim], ['RMSE', 'Rel', 'Thresh1', 'Thresh2', 'Thresh3', 'SSIM']

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.L1() * self.opt.lambda_L1
        self.loss_G_L2 = self.L2() * self.opt.lambda_L1
        # self.loss_G_scale = self.scale_loss() * self.opt.lambda_L1
        self.loss_G_hu = self.hu_loss() * self.opt.lambda_L1
        self.loss_G_berHu = self.berHu_loss() * self.opt.lambda_L1

        self.findEvalLosses()

        if(self.opt.loss2==0):
            self.loss_G = self.loss_G_L2 + self.loss_G_GAN
        elif(self.opt.loss2==1):
            print("******* issue in log_func, not resolved*************")
            exit()
            # self.loss_G = self.loss_G_scale + self.loss_G_GAN
        elif(self.opt.loss2==2):
            self.loss_G = self.loss_G_hu + self.loss_G_GAN
        elif(self.opt.loss2==3):
            self.loss_G = self.loss_G_berHu + self.loss_G_GAN
        elif(self.opt.loss2==4):
            self.loss_G = self.loss_G_L1 + self.loss_G_GAN 
        self.loss_G.backward()


    def optimize_parameters(self):
        self.forward()
        # # update D
        if self.opt.which_model_netG != 'Gen_depth':
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
        else:
            try:
                self.loss_names.remove('D_real')
                self.loss_names_plt.remove('D_real')
                self.loss_names.remove('D_fake')
                self.loss_names_plt.remove('D_fake')
            except:
                a = 0
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
