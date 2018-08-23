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
    phases = ['test', 'train']
    opt.no_flip = True
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
        for i, data in enumerate(dataset):
            if i >= opt.how_many:
                break
            model.set_input(data)
            model.test()
            model.findEvalLosses()
            if(torch.__version__ != '0.3.0.post4'):
                save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
            arr = updateLosses(model) if i==0  else updateLosses(model, arr)
            # i = 5 if a > 7 else 0

        webpage.save()
        updateLosses(model, arr, i, file=file)
    opt.no_flip = False
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
# DP used: 653 RMSE : 0.8662465251811066  Rel : 0.2496004984592862  Thresh1 : 0.5208516598737661  Thresh2 : 0.8526720607745238  Thresh3 : 0.9569496537747787  

