import os
import numpy as np
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import torch
def updateLosses(model, arr=None, div=None, file=None):
    if div is None:
        if arr is None:
            arr = np.zeros(len(model.evalLosses))
        for i, loss in enumerate(model.evalLosses):
            arr[i] += loss*1.0
        return arr
    else:
        prints = "DP used: {} ".format(div)
        for i, loss in enumerate(model.evalLosses):
            prints += model.evalLossesNames[i] + " : " + str(arr[i]/div) + "  "
        print(prints)
        if file is not None:
            file.write( prints + '\n')

def test(opt, model, file=None):
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        model.findEvalLosses()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        # if i % 5 == 0 and i !=0:
        #     print('processing (%04d)-th image... %s' % (i, img_path))
        if(torch.__version__ != '0.3.0.post4'):
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        arr = updateLosses(model) if i==0  else updateLosses(model, arr)
        # i = 5 if a > 7 else 0

    webpage.save()
    updateLosses(model, arr, i, file=file)

if __name__ == '__main__':

    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    model = create_model(opt)
    model.setup(opt)
    # f = open
    test(opt, model)


# DP used: 20 RMSE : 2.1986363351345064  Abs : 0.893079374730587  Thresh1 : 0.21322943793402777  Thresh2 : 0.41924327256944444  Thresh3 : 0.6183082139756945  
# DP used: 20 RMSE : 0.8820810839533806  Rel : 0.2345905341207981  Thresh1 : 0.61558837890625  Thresh2 : 0.8547732204861113  Thresh3 : 0.9602861870659722  
