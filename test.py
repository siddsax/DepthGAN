import os
import numpy as np
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import torch
from util.visualizer import Visualizer

def updateLosses(model, phase, arr=None, div=None, file=None):
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
        if file is not None and phase == 'test':
            file.write( prints + '\n')
        return arr/div

def test(opt, model, file=None):
    oldDid = opt.display_id
    oldNF = opt.no_flip
    opt.display_id = oldDid + 2
    phases = ['test']#, 'train']
    opt.no_flip = True
    visualizer = Visualizer(opt)
    a, b = opt.loadSize_1, opt.loadSize_2
    opt.loadSize_1, opt.loadSize_2 = opt.fineSize_1, opt.fineSize_2
    model.opt = opt
    for phase in phases:
        opt.phase = phase
        if(phase == 'test'):
            opt.how_many = 1000
        else:
            opt.how_many = 20
        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        # create website
        web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
        # model.eval()
        div = 0.0
        for i, data in enumerate(dataset):
            if i >= opt.how_many:
                break
            div +=1.0
            model.set_input(data)
            model.test()
            model.findEvalLosses()
            visualizer.display_current_results(model.get_current_visuals(), 0, 0)
            if(torch.__version__ != '0.3.0.post4'):
                visuals = model.get_current_visuals()
                img_path = model.get_image_paths()
                save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
            arr = updateLosses(model, phase) if i==0  else updateLosses(model, phase, arr=arr)
            # i = 5 if a > 7 else 0

        webpage.save()
        if(phase == 'test'):
            arrT = updateLosses(model, phase, arr, div, file=file)
        else:
            updateLosses(model, phase, arr, div, file=file)
    opt.phase = 'train'
    opt.display_id = oldDid
    opt.no_flip = oldNF
    opt.loadSize_1, opt.loadSize_2 = a, b
    model.opt = opt
    return arrT
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


# SOTA: 653 RMSE : 0.573  Rel : 0.127  Thresh1 : 0.811  Thresh2 : 0.953  Thresh3 : 0.988
# DP used: 653 RMSE : 0.9059211268998215  Rel : 0.2531337386330912  Thresh1 : 0.4879080608861876  Thresh2 : 0.8350951587624426  Thresh3 : 0.9522043177455556
# DP used: 654.0 RMSE : 0.8170174763335729  Rel : 0.2387121403561364  Thresh1 : 0.5894500307270428  Thresh2 : 0.8661375858100785  Thresh3 : 0.958658743005118

