import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from test import test
import pdb
def train(opt, model):
    model.opt = opt
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    visualizer = Visualizer(opt)
    total_steps = 0
    f = open('test_acc_' + opt.name, 'w')
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            iter_start_time = time.time()

            model.set_input(data)
            model.optimize_parameters()
            iter_data_time = time.time()
            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                _, losses_plt = model.get_current_losses()
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses_plt)
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save_networks('latest')
            # if total_steps % opt.test_freq == 0 and (opt.niter + opt.niter_decay + 1 - opt.epoch_count - epoch):
            #     test(opt, model=model, file=f)
            #     f.close()
            #     f = open('test_acc_' + opt.name, 'a')
            #     model.train()

        if epoch % opt.save_epoch_freq == 0:
            model.save_networks('latest')
            model.save_networks(epoch)
            model.update_learning_rate()
            # test(opt, model=model, file=f)
            # model.train()
            # f.close()
            # f = open('test_acc_' + opt.name, 'a')
            print("++++ End of {} Epochs ++++".format(epoch))
    return model

if __name__ == '__main__':

  opt = TrainOptions().parse()
  model = create_model(opt)
  model.setup(opt)
  train(opt, model)

# DP used: 63.0 RMSE : 0.29453890167531516  Rel : 0.07184203459866463  Thresh1 : 0.9658638013944005  Thresh2 : 0.9857821111662256  Thresh3 : 0.992957675402337
