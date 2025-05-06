from __future__ import print_function
import torch
import torch.nn as nn
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer, throughput_ratio

class NormalNN(nn.Module):
    '''
    Normal Neural Network for classification
    '''
    def __init__(self, agent_config):
        super(NormalNN, self).__init__()
        self.log = print
        self.config = agent_config
        self.model = self.create_model()
        self.criterion_fn = nn.CrossEntropyLoss()
        if agent_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        self.init_optimizer()
        self.reset_optimizer = False

    def init_optimizer(self):
        optimizer_arg = {'params':self.model.parameters(),
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr= self.config['lr'], betas=(0.9, 0.999), eps=1e-8,
                                          weight_decay=0, amsgrad=False)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['schedule'],
                                                              gamma=0.1)

    def create_model(self):
        cfg = self.config

        # Define the backbone of model
        if cfg['model_type'] == "fusion":
            lidar_model = models.lidarnet.LidarNet([20,200,10])
            coord_model = models.coordnet.CoordNet(2)
            model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](lidar_model, coord_model)
        else:
            model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']]()

        # Redefine the task-dependent function
        def new_logits(self, x):
            outputs = {}
            for task, func in self.last.items():
                outputs[task] = func(x)
            return outputs

        return model




    def forward(self, input_lidar, input_coord):
        return self.model.forward(input_lidar, input_coord)

    def predict(self, input_lidar, input_coord):
        self.model.eval()
        out = self.forward(input_lidar, input_coord)
        out = out.detach()
        return out

    def valid(self, dataloader, task):
        batch_timer = Timer()
        batch_timer.tic()
        losses = AverageMeter()
        top1_acc = AverageMeter()

        orig_mode = self.training
        self.eval()
        for i, (input_coord, input_lidar, target) in enumerate(dataloader):

            if self.gpu:
                with torch.no_grad():
                    input_coord = input_coord.cuda()
                    input_lidar = input_lidar.cuda()
                    target = target.cuda()
            output = self.predict(input_lidar, input_coord)
            loss = self.criterion_fn(output, target)
            losses.update(loss, input_coord.size(0))

            top1_acc = accumulate_acc1(output, target, top1_acc)

        return {
            'loss': losses.avg,
            'top1_acc': top1_acc.avg,
        }


    def validation(self, dataloader, task):
        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        batch_timer.tic()
        losses = AverageMeter()
        top1_acc = AverageMeter()
        top2_acc = AverageMeter()
        top5_acc = AverageMeter()
        top10_acc = AverageMeter()
        top1_th = AverageMeter()
        top2_th = AverageMeter()
        top5_th = AverageMeter()
        top10_th = AverageMeter()

        orig_mode = self.training
        self.eval()
        for i, (input_coord, input_lidar, target) in enumerate(dataloader):

            if self.gpu:
                with torch.no_grad():
                    input_coord = input_coord.cuda()
                    input_lidar = input_lidar.cuda()
                    target = target.cuda()
            output = self.predict(input_lidar, input_coord)
            loss = self.criterion_fn(output, target)
            losses.update(loss, input_coord.size(0))

            # Summarize the performance of all tasks, or 1 task, depends on dataloader.
            # Calculated by total number of Raymobtime.
            top1_acc, top2_acc, top5_acc, top10_acc = accumulate_acc(output, target, top1_acc, top2_acc, top5_acc, top10_acc)
            top1_th, top2_th, top5_th, top10_th = accumulate_th(output, target, top1_th, top2_th, top5_th, top10_th)

        self.train(orig_mode)

        self.log(' * Val{i} Acc-top1 {acc1.avg:.3f}, Total time {time:.2f}'
              .format(i=task, acc1=top1_acc,time=batch_timer.toc()))
        return {
            'loss': losses.avg,
            'top1_acc': top1_acc.avg,
            'top2_acc': top2_acc.avg,
            'top5_acc': top5_acc.avg,
            'top10_acc': top10_acc.avg,
            'top1_th': top1_th.avg,
            'top2_th': top2_th.avg,
            'top5_th': top5_th.avg,
            'top10_th': top10_th.avg
        }

    def criterion(self, preds, targets, **kwargs):
        # The inputs and targets could come from single task or a mix of tasks
        # The network always makes the predictions with all its heads
        # The criterion will match the head and task to calculate the loss.
            loss = self.criterion_fn(preds, targets)
            return loss

    def update_model(self, input_lidars, input_coords, targets):
        out = self.forward(input_lidars, input_coords)
        loss = self.criterion(out, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach(), out

    def learn_batch(self, train_name, train_loader, val_loader_task=None, test_loader_task1=None, test_loader_task2 = None, modelname=None):
        epoch_data = {
            'val_task1_loss': [], 'val_task1_top1_acc': [],
            'val_task2_loss': [], 'val_task2_top1_acc': [],
            'test_task1_loss': [], 'test_task1_top1_acc': [], 'test_task1_top2_acc': [],
            'test_task1_top5_acc': [], 'test_task1_top10_acc': [],
            'test_task2_loss': [], 'test_task2_top1_acc': [], 'test_task2_top2_acc': [],
            'test_task2_top5_acc': [], 'test_task2_top10_acc': [],
            'test_task1_top1_th': [], 'test_task1_top2_th': [], 'test_task1_top5_th': [],
            'test_task1_top10_th': [], 'test_task2_top1_th': [], 'test_task2_top2_th': [],
            'test_task2_top5_th': [], 'test_task2_top10_th': [],
            'val_task1_epoch' : [-1], 'val_task2_epoch' : [-1]
        }

        best_val_loss_task1 = float('inf')
        best_val_loss_task2 = float('inf')

        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()

        for epoch in range(self.config['schedule'][-1]):
            data_timer = Timer()
            batch_timer = Timer()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1_acc = AverageMeter()

            # Config the model and optimizer
            self.log('Epoch:{0}'.format(epoch))
            self.model.train()
            for param_group in self.optimizer.param_groups:
                self.log('LR:',param_group['lr'])

            # Learning with mini-batch
            data_timer.tic()
            batch_timer.tic()

            for i, (input_coord, input_lidar, target) in enumerate(train_loader):

                data_time.update(data_timer.toc())  # measure Raymobtime loading time

                if self.gpu:
                    input_coord = input_coord.cuda()
                    input_lidar = input_lidar.cuda()
                    target = target.cuda()

                loss, output = self.update_model(input_lidar, input_coord, target)  #
                input_coord = input_coord.detach()
                input_lidar = input_lidar.detach()
                target = target.detach()

                # measure accuracy and record loss
                top1_acc = accumulate_acc1(output, target, top1_acc)
                # top1_th, top2_th, top5_th, top10_th = accumulate_th(output, target, top1_th, top2_th, top5_th, top10_th)
                losses.update(loss, input_coord.size(0))

                batch_time.update(batch_timer.toc())  # measure elapsed time
                data_timer.toc()


            self.log(' * Train top1-Acc {acc.avg:.3f}'.format(acc=top1_acc))


            # Evaluate the performance of current task
            if train_name == 1:
                valid_metrics_task1 = self.valid(val_loader_task, 1)
                test_metrics_task1 = self.validation(test_loader_task1, 1)
                save_metrics(epoch_data, valid_metrics_task1, 'val_task1')
                save_metrics(epoch_data, test_metrics_task1, 'test_task1')
                if valid_metrics_task1['loss'] < best_val_loss_task1:
                    epoch_data['val_task1_epoch'][0] = epoch
            else:
                valid_metrics_task2 = self.valid(val_loader_task, 2)
                test_metrics_task1 = self.validation(test_loader_task1, 1)
                test_metrics_task2 = self.validation(test_loader_task2, 2)
                save_metrics(epoch_data, valid_metrics_task2, 'val_task2')
                save_metrics(epoch_data, test_metrics_task1, 'test_task1')
                save_metrics(epoch_data, test_metrics_task2, 'test_task2')
                if valid_metrics_task2['loss'] < best_val_loss_task2:
                    epoch_data['val_task2_epoch'][0] = epoch

            self.scheduler.step(epoch)
        if train_name == 2:
            torch.save(self.model, modelname)


        return epoch_data

    def learn_stream(self, data, label):
        assert False,'No implementation yet'

    def save_metrics(epoch_data, metrics, task_name):
        for key, value in metrics.items():
            epoch_data[f'{key}_{task_name}'].append(value)


    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())

    def save_model(self, filename):
        model_state = self.model.state_dict()
        if isinstance(self.model,torch.nn.DataParallel):
            # Get rid of 'module' before the name of states
            model_state = self.model.module.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        print('=> Saving model to:', filename)
        torch.save(model_state, filename + '.pth')
        print('=> Save Done')

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self


def accumulate_acc1(output, target, top1):
    top1.update(accuracy(output, target), len(target))
    return top1


def accumulate_acc(output, target, top1, top2, top5, top10):
        metrics = accuracy(output, target, (1,2,5,10))
        top1.update(metrics[0], len(target))
        top2.update(metrics[1], len(target))
        top5.update(metrics[2], len(target))
        top10.update(metrics[3], len(target))
        return top1, top2, top5, top10

def accumulate_th(output, target, top1, top2, top5, top10 ):
      metrics = throughput_ratio(output, target)
      top1.update(metrics[0], len(target))
      top2.update(metrics[1], len(target))
      top5.update(metrics[2], len(target))
      top10.update(metrics[3], len(target))
      return top1, top2, top5, top10


def save_metrics(epoch_data, metrics, task_name):
    for key, value in metrics.items():
        epoch_data[f'{task_name}_{key}'].append(value)
