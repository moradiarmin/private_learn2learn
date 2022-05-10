#!/usr/bin/env python3

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import argparse
import random
import pickle

import numpy as np
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import os

import learn2learn as l2l
from PIL import Image

from torch.utils.tensorboard import SummaryWriter

class Net(nn.Module):
    def __init__(self, ways=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, ways)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1)
    acc = (predictions == targets).sum().float()
    acc /= len(targets)
    return acc.item()


def get_sample_from_data(dataset, from_calss):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=200)
    for batch_idx, (data, label) in enumerate(dataloader):
        idx = label == from_calss
        idx = np.where(idx)[0]
        rand_idx = random.choice(idx)
        sample_x = data[rand_idx]
        sample_y = label[rand_idx]
        sample_x = torch.unsqueeze(sample_x, dim=0)
        sample_y = torch.unsqueeze(sample_y, dim=0)
        return sample_x, sample_y


def plot_arr(arr, color, save_addr):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(arr, color + 'o')
    plt.ylabel('loss on target')
    plt.xlabel('epoch')
    plt.savefig(save_addr)

def save_arr(arr,save_addr):
    os.makedirs(os.path.dirname(save_addr), exist_ok=True)
    with open(save_addr,'wb') as f:
        pickle.dump(arr,f)

def gradient_ascent_on_data(maml, attack_data_x, attack_data_y):
    for i in range(1):
        output = maml(attack_data_x)
        error = - F.nll_loss(output, attack_data_y)

        opt = torch.optim.Adam(maml.parameters(), lr=4e-3)
        opt.zero_grad()
        error.backward()
        opt.step()

    output = maml(attack_data_x)
    error = F.nll_loss(output, attack_data_y)

    return maml, error


def show_image(img_tensor):
    img_arr = img_tensor.data.cpu().numpy()[0][0]
    img_arr = np.interp(img_arr, (img_arr.min(), img_arr.max()), (0, 255))
    img = Image.fromarray(img_arr)
    # img.save('my.png')
    img.show()


def choose_attack_data(dataset, member, task=None, query=True, shots=None, ways=None):
    if not member:
        target_class = random.randint(0, ways-1)
        target_x, target_y = get_sample_from_data(dataset, from_calss=target_class)
    else:
        train_task = dataset[task]
        data, labels = train_task
        # data = data.to(device)
        # labels = labels.to(device)
        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        adaptation_indices[np.arange(shots * ways) * 2] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
        data_idx = random.randint(0, len(evaluation_labels)-1)
#        print("#################" , len(evaluation_labels) , len(adaptation_labels))
        if query:
            target_x = evaluation_data[data_idx]
            target_y = evaluation_labels[data_idx]
        else:
            target_x = adaptation_data[data_idx]
            target_y = adaptation_labels[data_idx]

        target_x = torch.unsqueeze(target_x, dim=0)
        target_y = torch.unsqueeze(target_y, dim=0)

    return target_x, target_y


def run_attack(experiment_no=None, writer=None, is_dp=False, member=False, lr=0.005, maml_lr=0.01, iterations=1000, ways=5, shots=1, tps=1, fas=5, device=torch.device("cpu"),
         download_location='~/data', coef=0.0001, query=True, attack_iter=1, save_location=''):
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        lambda x: x.view(1, 28, 28),
    ])

    mnist_train = l2l.data.MetaDataset(MNIST(download_location,
                                             train=True,
                                             download=True,
                                             transform=transformations))

    train_tasks = l2l.data.TaskDataset(mnist_train,
                                       task_transforms=[
                                           l2l.data.transforms.NWays(mnist_train, ways),
                                           l2l.data.transforms.KShots(mnist_train, 2 * shots),
                                           l2l.data.transforms.LoadData(mnist_train),
                                           l2l.data.transforms.RemapLabels(mnist_train),
                                           l2l.data.transforms.ConsecutiveLabels(mnist_train),
                                       ],
                                       num_tasks=1000)

    model = Net(ways)
    model.to(device)
    if is_dp:
        # print('run private')
        meta_model = l2l.algorithms.MAML_DP(model, lr=maml_lr)
    else:
        meta_model = l2l.algorithms.MAML(model, lr=maml_lr)
    opt = optim.Adam(meta_model.parameters(), lr=lr)
    loss_func = nn.NLLLoss(reduction='mean')

    mnist_for_target = torchvision.datasets.MNIST(root="/tmp/mnist", train=False, download=True,
                                                  transform=transformations)

    if not member:
        attack_data = mnist_for_target
    else:
        attack_data = train_tasks
    attack_task = random.randint(0, tps-1)
    target_x, target_y = choose_attack_data(attack_data, member, task=attack_task, query=query, shots=shots, ways=ways)
    target_x = target_x.to(device)
    target_y = target_y.to(device)

    target_losses = []
    bad_losses = []
    task_losses = []

    for iteration in range(iterations):
        iteration_error = 0.0
        iteration_acc = 0.0

        for i in range(tps):

            learner = meta_model.clone()
            train_task = train_tasks[i]
            data, labels = train_task
            data = data.to(device)
            labels = labels.to(device)

            # Separate data into adaptation/evalutation sets
            adaptation_indices = np.zeros(data.size(0), dtype=bool)
            adaptation_indices[np.arange(shots * ways) * 2] = True
            evaluation_indices = torch.from_numpy(~adaptation_indices)
            adaptation_indices = torch.from_numpy(adaptation_indices)
            adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
            evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
            # get target and gradient ascent on that
            if iteration % attack_iter == attack_iter-1 and i == 0:
                predictions = learner(target_x)
                valid_error = loss_func(predictions, target_y)
                valid_error /= len(target_x)
                adv_error = valid_error

            # Fast Adaptation
            for step in range(fas):
                train_error = loss_func(learner(adaptation_data), adaptation_labels)
                learner.adapt(train_error)

            # Compute validation loss
            predictions = learner(evaluation_data)
            valid_error = loss_func(predictions, evaluation_labels)
            valid_error /= len(evaluation_data)
            valid_accuracy = accuracy(predictions, evaluation_labels)
            t_error = loss_func(learner(target_x), target_y)
#            print("good error", t_error)
#            print("all good error", valid_error)
            task_losses.append(valid_error.data.cpu().numpy())
            iteration_error += valid_error
            iteration_acc += valid_accuracy
            
            if iteration % attack_iter == attack_iter-1 and i == 0:
                if adv_error != 0:
                    coef =  (valid_error / adv_error) / (ways*shots)
                else:
                    coef = 1
                iteration_error -= adv_error*coef
                bad_losses.append((adv_error * coef).data.cpu().numpy())
                

        iteration_error /= (tps+1)
        iteration_acc /= tps
        # print('Loss : {:.3f} Acc : {:.3f}'.format(iteration_error.item(), iteration_acc), 'iteration:', iteration)
        # Take the meta-learning step
        opt.zero_grad()
        iteration_error.backward()
        opt.step()

        x_pred = meta_model(target_x)
        target_error = loss_func(x_pred, target_y)
        target_losses.append(target_error.data.cpu().numpy())

        # print("after meta update", target_error)
        writer.add_scalar(f'Iteration Accuracy {experiment_no}', iteration_acc, iteration)
        writer.add_scalar(f'Iteration Error {experiment_no} ', iteration_error.item(), iteration)
        writer.add_scalar(f'Target Error {experiment_no}', target_error.data.cpu().numpy(), iteration)
    save_arr(target_losses, f'{save_location}/target{experiment_no}.pkl')
    # save_arr(bad_losses, f'{save_location}/bad{experiment_no}.pkl')
    # save_arr(task_losses, f'{save_location}/task{experiment_no}.pkl')



def main(args, experiment_count=50):

    dp_dir = ['/nodp', '/dp']
    member_dir = ['/notmember', '/member']
    is_member_arr = [False]
    is_dp_arr = [False]

    for i in range(len(is_member_arr)):
        is_member = is_member_arr[i]
        is_dp = is_dp_arr[i]
        log_dir = dp_dir[is_dp] + member_dir[is_member]
        writer = SummaryWriter(log_dir='log' + log_dir)
        save_location = 'results' + log_dir

        for en in range(experiment_count):
            if en % 10 == 0:
                print (f'--------- {en}th iteration out of {experiment_count} experiments.' )

            run_attack(experiment_no=en,
                writer=writer,
                is_dp=is_dp,
                member=is_member,
                save_location=save_location,
                lr=args.lr,
                maml_lr=args.maml_lr,
                iterations=args.iterations,
                ways=args.ways,
                shots=args.shots,
                tps=args.tasks_per_step,
                fas=args.fast_adaption_steps,
                device=device,
                download_location=args.download_location,
                query=args.query,
                coef=args.coef,
                attack_iter=args.attack_iter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn2Learn MNIST Example')

    parser.add_argument('--member', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='is target member or not')


    parser.add_argument('--is-dp', default=False,type=lambda x: (str(x).lower() == 'true'),
                        help='should run dp')

    parser.add_argument('--coef', type=float, default=1 / 10000,
                        help='coefficient to multiply to grad ascent loss')

    parser.add_argument('--query', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='is target from query set or support set')

    parser.add_argument('--attack-iter', type=int, default=1, metavar='N',
                        help='number of iterations to attack')

    parser.add_argument('--ways', type=int, default=5, metavar='N',
                        help='number of ways (default: 5)')
    parser.add_argument('--shots', type=int, default=3, metavar='N',
                        help='number of shots (default: 1)')
    parser.add_argument('-tps', '--tasks-per-step', type=int, default=1, metavar='N',
                        help='tasks per step (default: 32)')
    parser.add_argument('-fas', '--fast-adaption-steps', type=int, default=1, metavar='N',
                        help='steps per fast adaption (default: 5)')

    parser.add_argument('--iterations', type=int, default=100, metavar='N',
                        help='number of iterations (default: 1000)')

    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--maml-lr', type=float, default=0.01, metavar='LR',
                        help='learning rate for MAML (default: 0.01)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--download-location', type=str, default="/tmp/mnist", metavar='S',
                        help='download location for train data (default : /tmp/mnist')

    parser.add_argument('--save-location', type=str, default="results_arr/", metavar='S',
                        help='save location for results')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

#    random.seed(args.seed)
#    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")

    main(args=args, experiment_count=70)
