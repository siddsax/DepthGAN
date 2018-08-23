from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=1000, help='how many test images to run')
        parser.add_argument('--continue_train', type=int, default=1, help='continue training: load the latest model')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
	parser.set_defaults(model='test')
        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize_1=parser.get_default('fineSize_1'))
        parser.set_defaults(loadSize_2=parser.get_default('fineSize_2'))
        
        self.isTrain = False
        return parser
