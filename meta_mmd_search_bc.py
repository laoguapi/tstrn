from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader2
import shutil
# import ResNet_2 as models
import vggnet_meta as models
from Weight import Weight
from Config_bc import *
import time
from sklearn.metrics import confusion_matrix
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id

cuda = not no_cuda and torch.cuda.is_available()
#torch.manual_seed(seed)
#if cuda:
#    torch.cuda.manual_seed(seed)

nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
kwargs = {'num_workers': 4, 'pin_memory': False} if cuda else {}

source_loader = data_loader2.load_training(root_path, source_name, batch_size, kwargs)
# target_train_loader = data_loader2.load_training_target(root_path, target_name, batch_size, kwargs)
target_train_loader = data_loader2.load_training(root_path, target_name, batch_size, kwargs)
target_test_loader = data_loader2.load_testing(root_path, target_name, batch_size, kwargs)

len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_test_loader.dataset)
len_source_loader = len(source_loader)
len_target_loader = len(target_train_loader)

def get_parameters(model, name=False):
    params = []
    if name:
        for n, p in model.named_parameters():
            params.append([n, p])
    else:
        for n, p in model.named_parameters():
            params.append(p)

    return params


def train(epoch, model, paramlist):
    meta_lr = 10.0
    LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
    correct_s = 0
    print('learning rate{: .4f}'.format(LEARNING_RATE))
    if bottle_neck:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            # {'params': model.bottle.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)
    else:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
            ], lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)

    model.train()

    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len_source_loader
    for i in range(1, num_iter):
        for p in model.parameters():
            p.fast = None
        data_source, label_source = iter_source.next()
        data_target, _ = iter_target.next()
        if i % len_target_loader == 0:
            iter_target = iter(target_train_loader)
        if cuda:
            data_source, label_source = data_source.cuda(), label_source.cuda()
            data_target = data_target.cuda()
        data_source, label_source = Variable(data_source), Variable(label_source)
        data_target = Variable(data_target)

        # optimizer.zero_grad()

        # meta_train_1
        _, _, loss_mmd, _, _ = model(data_source, data_target, label_source, mode='bc')
        loss_mmd.requires_grad_()
        meta_grad = torch.autograd.grad(
            outputs=loss_mmd,
            inputs=get_parameters(model),
            create_graph=True,
            allow_unused=True
        )
        meta_grad = [g.detach() if g is not None else g for g in meta_grad]
        for k, param in enumerate(get_parameters(model, name=True)):
            if meta_grad[k] is not None:
                param[1].fast = param[1] - meta_lr * LEARNING_RATE * meta_grad[k]

        # # meta_train_2
        # _, _, _, loss_lmmd, _ = model(data_source, data_target, label_source, mode='bc')
        # loss_lmmd.requires_grad_()
        # meta_grad = torch.autograd.grad(
        #     outputs=loss_lmmd,
        #     inputs=get_parameters(model),
        #     create_graph=True,
        #     allow_unused=True
        # )
        # meta_grad = [g.detach() if g is not None else g for g in meta_grad]
        # for k, param in enumerate(get_parameters(model, name=True)):
        #     if meta_grad[k] is not None:
        #         param[1].fast = param[1] - meta_lr * LEARNING_RATE * meta_grad[k]

        # meta_test
        label_source_pred, label_target, _, _, _ = model(data_source, data_target, label_source, mode='bc')
        source_pred = label_source_pred.data.max(1)[1]
        loss_cls = F.cross_entropy(label_source_pred, label_source)
        lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
        loss = paramlist[0] * loss_cls + lambd * paramlist[1] * loss_mmd
        correct_s += source_pred.eq(label_source.data.view_as(source_pred)).cpu().sum()

        loss.backward()
        # if i % 4 == 0:
        #     optimizer.step()
        #     optimizer.zero_grad()
        optimizer.step()
        optimizer.zero_grad()

        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLcls_Loss: {:.6f}'
                  '\tMMD_Loss: {:.6f}'
                  '\tAccuracy: {}/{} ({:.2f}%)'.format(
                epoch, i * len(data_source), len_source_dataset,
                       100. * i / len_source_loader, loss.item(), loss_cls.item(),
                        loss_mmd.item(),
                        correct_s, len_source_dataset, 100. * correct_s / len_source_dataset))


# def train(epoch, model, paramlist):
#     meta_lr = 10.0
#     LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
#     correct_s = 0
#     print('learning rate{: .4f}'.format(LEARNING_RATE))
#     if bottle_neck:
#         optimizer = torch.optim.SGD([
#             {'params': model.feature_layers.parameters()},
#             # {'params': model.bottle.parameters(), 'lr': LEARNING_RATE},
#             {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
#         ], lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)
#     else:
#         optimizer = torch.optim.SGD([
#             {'params': model.feature_layers.parameters()},
#             {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
#             ], lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)
#
#     model.train()
#
#     iter_source = iter(source_loader)
#     iter_target = iter(target_train_loader)
#     num_iter = len_source_loader
#     for i in range(1, num_iter):
#         for p in model.parameters():
#             p.fast = None
#         data_source, label_source = iter_source.next()
#         data_target, _ = iter_target.next()
#         if i % len_target_loader == 0:
#             iter_target = iter(target_train_loader)
#         if cuda:
#             data_source, label_source = data_source.cuda(), label_source.cuda()
#             data_target = data_target.cuda()
#         data_source, label_source = Variable(data_source), Variable(label_source)
#         data_target = Variable(data_target)
#
#         # optimizer.zero_grad()
#
#         # meta_train_1
#         label_source_pred, label_target, _, _, _ = model(data_source, data_target, label_source, mode='bc')
#         source_pred = label_source_pred.data.max(1)[1]
#         loss_cls = F.cross_entropy(label_source_pred, label_source)
#         meta_grad = torch.autograd.grad(
#             outputs=loss_cls,
#             inputs=get_parameters(model),
#             create_graph=True,
#             allow_unused=True
#         )
#         meta_grad = [g.detach() if g is not None else g for g in meta_grad]
#         for k, param in enumerate(get_parameters(model, name=True)):
#             if meta_grad[k] is not None:
#                 param[1].fast = param[1] - meta_lr * LEARNING_RATE * meta_grad[k]
#
#
#         # meta_test
#         _, _, loss_mmd, _, _ = model(data_source, data_target, label_source, mode='bc')
#         loss_mmd.requires_grad_()
#         lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
#         loss = paramlist[0] * loss_cls + lambd * paramlist[1] * loss_mmd
#         correct_s += source_pred.eq(label_source.data.view_as(source_pred)).cpu().sum()
#
#         loss.backward()
#         # if i % 4 == 0:
#         #     optimizer.step()
#         #     optimizer.zero_grad()
#         optimizer.step()
#         optimizer.zero_grad()
#
#         if i % log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLcls_Loss: {:.6f}'
#                   '\tMMD_Loss: {:.6f}'
#                   '\tAccuracy: {}/{} ({:.2f}%)'.format(
#                 epoch, i * len(data_source), len_source_dataset,
#                        100. * i / len_source_loader, loss.item(), loss_cls.item(),
#                         loss_mmd.item(),
#                         correct_s, len_source_dataset, 100. * correct_s / len_source_dataset))


def test(model):
    model.eval()
    test_loss = 0
    correct = torch.tensor(0.0)
    pred_lab = []
    true_lab = []
    with torch.no_grad():
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            s_output, t_output, t_output1, t_output2 = model(data, data, target, mode='ec')
            test_loss += F.nll_loss(F.log_softmax(s_output, dim=1), target).item()  # sum up batch loss
            pred = s_output.data.max(1)[1]  # get the index of the max log-probability
            pred_lab += pred.cpu().tolist()
            true_lab += target.cpu().tolist()
            # print("target,", target.data.view_as(pred))
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len_target_dataset
        confusion_matrix_test = confusion_matrix(true_lab, pred_lab)
        uu = 0
        i = 0
        for i in range(5):
            u = confusion_matrix_test[i][i] / np.array(confusion_matrix_test[i]).sum()
            uu = uu + u
        uar = uu / 5
        war = correct / len_target_dataset
        print('uar {:.4f}\twar {:.4f}'.format(uar, war))
        print('confusion matrix: \n{}'.format(confusion_matrix_test))
    return confusion_matrix_test, uar, war


if __name__ == '__main__':
    f_name = './log_meta/{}{}-meta.txt'.format(source_name, target_name)
    f_para_name = './log_meta/param-{}{}.txt'.format(source_name, target_name)
    with open(f_para_name, "a+") as f:
        f.seek(0, 0)
        f_content = f.readlines()
        root_latest = f_content[0]
        root_best = f_content[1]
        paramlist = eval(f_content[2])
    print(root_best, root_latest, paramlist)

    model = models.DSAN(num_classes=class_num)
    # model.load_state_dict(torch.load("/data/wangjincen/DSAN3/DSAN/weight/BC/best-37.9.pth"), strict=False)
    w_confusion_matrix_test = []
    u_confusion_matrix_test = []
    uar = 0
    war = 0
    print(model)
    if cuda:
        model.cuda()
    time_start = time.time()

    for epoch in range(1, epochs + 1):
        # 返回了上一个epoch最后的Pt
        train(epoch, model, paramlist)
        t_confusion_matrix_test, t_uar, t_war = test(model)
        torch.save(model.state_dict(), root_latest)
        # if t_war > war:
        #     war = t_war
        #     uar = t_uar
        #     w_confusion_matrix_test = t_confusion_matrix_test
        #     shutil.copy(root_latest, root_best)
        if t_uar > uar:
            war = t_war
            uar = t_uar
            u_confusion_matrix_test = t_confusion_matrix_test
            shutil.copy(root_latest, root_best)
        print('source: {} to target: {}\n best UAR: {}\tUAR confusion_matrix: \n{}\n'
              'best WAR: {}\tWAR confusion_matrix: \n{} \n'.format(
              source_name, target_name, uar, u_confusion_matrix_test, war, u_confusion_matrix_test))
        # print('source: {} to target: {}\n best UAR: {}\t'
        #       'best WAR: {}\tconfusion_matrix: \n{} \n'.format(
        #     source_name, target_name, uar, war, w_confusion_matrix_test))
        end_time = time.time()
        print('cost time:', end_time - time_start)
    f_write = '{}\tbest_uar:{}\tbest_war:{}\ncmx:{}\n'.format(paramlist, uar, war, u_confusion_matrix_test)
    with open(f_name, "a+") as f:
        f.write(f_write + '\n')