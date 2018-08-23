import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset,get_transform
from data.image_folder import make_dataset
from PIL import Image
import random

class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')
        self.transform = get_transform(opt)


    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h)).resize((self.opt.loadSize_1, self.opt.loadSize_2), Image.BICUBIC)
        B = AB.crop((w2, 0, w, h)).resize((self.opt.loadSize_1, self.opt.loadSize_2), Image.BICUBIC)
        if self.opt.isTrain and not self.opt.no_flip:
            # print("="*1000)
            transform = transforms.RandomHorizontalFlip()
            seed = random.randint(0,2**32)
            random.seed(seed)
            A = transform(A)
            random.seed(seed)
            B = transform(B)
            # transform = transforms.Compose([transforms.RandomHorizontalFlip()])
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        # print(A.shape)
        # print(B.shape)
        w_offset = random.randint(0, max(0, self.opt.loadSize_1 - self.opt.fineSize_1 - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize_2 - self.opt.fineSize_2 - 1))

        # print(w_offset)
        # print(h_offset)

        A = A[:, h_offset:h_offset + self.opt.fineSize_2, w_offset:w_offset + self.opt.fineSize_1]
        B = B[:, h_offset:h_offset + self.opt.fineSize_2, w_offset:w_offset + self.opt.fineSize_1]
        # print(A.shape)
        # print(B.shape)
        # print(self.opt.fineSize_1)
        # print(self.opt.fineSize_2)
        # print("+===================")
        # exit()
        A = 2*(A) - 1#transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = 2*(B) - 1#transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
        # print("Amax {}".format(A.max()))
        # print("Amean {}".format(A.mean()))
        # print("Amin {}".format(A.min()))
        # print("Bmax {}".format(B.max()))
        # print("Bmean {}".format(B.mean()))
        # print("Bmin {}".format(B.min()))
        # exit()
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
