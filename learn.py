import os
import sys
import argparse
import torch
import numpy as np
from dataloaders.SplitRay import SplitRay
import agents
from utils.metric import save_results_to_npz

def update_combined_metrics(combined_metrics, epoch_data):

    for key in epoch_data:
        if (key == 'val_task2_epoch' or key == 'val_task1_epoch') and epoch_data[key][0] == -1:
            continue
        if (key == 'training_time'):
            continue
        combined_metrics[key].extend(epoch_data[key])


def run(args, modelname):
    global combined_metrics
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    # Prepare dataloaders
    train_dataset_splits, val_dataset_splits,test_dataset_splits, task_output_space = SplitRay()

    # Prepare the Agent (model)
    agent_config = {'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay,'schedule': args.schedule,
                    'model_type':args.model_type, 'model_name': args.model_name,
                    'optimizer':args.optimizer,
                    'gpuid': args.gpuid,
                    'reg_coef':args.reg_coef}
    agent = agents.__dict__[args.agent_type].__dict__[args.agent_name](agent_config)
    print(agent.model)
    print('#parameter of model:',agent.count_parameter())

    # Decide split ordering
    task_names = [1,2]
    print('Task order:', task_names)

    # Feed Raymobtime to agent and evaluate agent's performance
    combined_metrics = {
        'val_task1_loss': [], 'val_task1_top1_acc': [],
        'val_task2_loss': [], 'val_task2_top1_acc': [],
        'test_task1_loss': [], 'test_task1_top1_acc': [], 'test_task1_top2_acc': [],
        'test_task1_top5_acc': [], 'test_task1_top10_acc': [],
        'test_task2_loss': [], 'test_task2_top1_acc': [], 'test_task2_top2_acc': [],
        'test_task2_top5_acc': [], 'test_task2_top10_acc': [],
        'test_task1_top1_th': [], 'test_task1_top2_th': [], 'test_task1_top5_th': [],
        'test_task1_top10_th': [], 'test_task2_top1_th': [], 'test_task2_top2_th': [],
        'test_task2_top5_th': [], 'test_task2_top10_th': [],
        'val_task1_epoch' : [], 'val_task2_epoch' : []
    }

    test_loader_task1 = torch.utils.data.DataLoader(test_dataset_splits[1],
                                                    batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.workers)
    test_loader_task2 = torch.utils.data.DataLoader(test_dataset_splits[2],
                                                    batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.workers)
    for i in range(len(task_names)):
        train_name = task_names[i]
        print('======================',train_name,'=======================')
        train_loader = torch.utils.data.DataLoader(train_dataset_splits[train_name],
                                                    batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader_task = torch.utils.data.DataLoader(val_dataset_splits[train_name],
                                                  batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        # Learn
        metrics = agent.learn_batch(train_name, train_loader, val_loader_task, test_loader_task1, test_loader_task2,modelname)
        update_combined_metrics(combined_metrics, metrics)
    return combined_metrics

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--model_type', type=str, default='fusion', help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--model_name', type=str, default='FusionNet', help="The name of actual model for the backbone")
    parser.add_argument('--agent_type', type=str, default='regularization', help="The type (filename) of agent")
    parser.add_argument('--agent_name', type=str, default='SI', help="The class name of agent")
    parser.add_argument('--optimizer', type=str, default='Adam', help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--dataroot', type=str, default='Raymobtime', help="The root folder of dataset or downloaded Raymobtime")
    parser.add_argument('--workers', type=int, default=0, help="#Thread for dataloader")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--schedule', nargs="+", type=int, default=[100], help="The epoch numbers")
    parser.add_argument('--reg_coef', nargs="+", type=float, default=[1100], help="The coefficient for regularization. Larger means less plasilicity. Give a list for hyperparameter search.")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")

    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    reg_coef_list = args.reg_coef
    avg_final_acc = {}

    # The for loops over hyper-paramerters or repeats
    for reg_coef in reg_coef_list:
        args.reg_coef = reg_coef
        avg_final_acc[reg_coef] = np.zeros(args.repeat)
        results_dict = {
            'val_task1_loss': [], 'val_task1_top1_acc': [],
            'val_task2_loss': [], 'val_task2_top1_acc': [],
            'test_task1_loss': [], 'test_task1_top1_acc': [], 'test_task1_top2_acc': [],
            'test_task1_top5_acc': [], 'test_task1_top10_acc': [],
            'test_task2_loss': [], 'test_task2_top1_acc': [], 'test_task2_top2_acc': [],
            'test_task2_top5_acc': [], 'test_task2_top10_acc': [],
            'test_task1_top1_th': [], 'test_task1_top2_th': [], 'test_task1_top5_th': [],
            'test_task1_top10_th': [], 'test_task2_top1_th': [], 'test_task2_top2_th': [],
            'test_task2_top5_th': [], 'test_task2_top10_th': [],
            'val_task1_epoch': [], 'val_task2_epoch': []
        }

        if (args.agent_type == 'regularization'):
            filename = os.path.join('outputs', args.agent_name, str(args.reg_coef), str(args.repeat) + '_results.npz')
        elif(args.agent_type == 'default'):
            filename = os.path.join('outputs_new', args.agent_type,  str(args.repeat) + '_results.npz')
        else:
            filename = os.path.join('outputs_new', args.agent_name, str(args.memory_size), str(args.repeat) + '_results.npz')

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        for r in range(args.repeat):
            round = r + 1
            print(f"round: {round}, regularization term：{reg_coef}")
            if (args.agent_type == 'regularization'):
                modelname = os.path.join('outputs', args.agent_name, str(args.reg_coef),
                                        str(r+1) + '_model.pth')
            elif (args.agent_type == 'default'):
                modelname = os.path.join('outputs', args.agent_type, str(r+1) + '_model.pth')
            else:
                modelname = os.path.join('outputs', args.agent_name, str(args.memory_size),
                                        str(r+1) + '_model.pth')

            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # Run the experiment
            metrics = run(args, modelname)

        for key in results_dict:
            results_dict[key] = np.array(results_dict[key])

        save_results_to_npz(results_dict, filename)

