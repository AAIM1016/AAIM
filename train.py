import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics
from AAIM_model import AAIM
from data import DataLoad
import random

def cacu_metric(output, y):
    predict = torch.argmax(output, dim=-1)
    ACC = torch.sum(predict == y)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    y = y.cpu()
    predict = predict.cpu()
    for i in range(len(y)):
        if y[i] == 1 and predict[i] == 1:
            TP += 1
        elif y[i] == 0 and predict[i] == 0:
            TN += 1
        elif y[i] == 0 and predict[i] == 1:
            FP += 1
        elif y[i] == 1 and predict[i] == 0:
            FN += 1
    return ACC / len(y), TP, TN, FP, FN, TP / (TP + FN), TN / (TN + FP)  

def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    if args.seed:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    all_best_ACC = np.zeros(5)
    all_best_SEN = np.zeros(5)
    all_best_SPE = np.zeros(5)
    all_best_AUC = np.zeros(5)
    all_best_epoch = np.zeros(5)
    print(args.dataset)
    for k in args.k_folds:    
        train_data=DataLoad(partition='train', dataset=args.dataset, fold=k)
        valid_data=DataLoad(partition='test', dataset=args.dataset, fold=k)
        num_train = len(train_data)
        num_valid = len(valid_data)
        sample, _ = train_data.__getitem__(1)  
        print(sample.shape)
        num_frame, num_point = sample.shape
        train_loader = DataLoader(train_data, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_data, num_workers=0, batch_size=num_valid, shuffle=False, drop_last=False)
        model = AAIM(num_point=num_point, GM_n=args.GM_n, WM_n=args.WM_n, config=args.config)
        model = model.cuda(args.gpu)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=20, verbose=True, threshold=0.00001)
        loss_F = nn.CrossEntropyLoss().to(args.gpu)
        Best_ACC_most = 0
        Best_ACC_mean = 0
        Best_auc = 0
        softmax = nn.Softmax(dim=1)
        relu = nn.ReLU()
        tanh = nn.Tanh()
        for epoch in range(args.end_epoch):
            if epoch < args.warm_up_epoch:
                lr = args.lr * (epoch + 1) / args.warm_up_epoch
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            model.train()
            total_loss = 0
            train_ACC = 0
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                x, target = data
                x = x.cuda(args.gpu)
                target = target.cuda(args.gpu)
                gm, wm, output = model(x)
                train_ACC += torch.sum((torch.argmax(output, dim=-1) == target))
                loss = loss_F(output, target)
                total_loss += loss
                loss.backward()
                if args.modulation_start <= epoch <= args.modulation_end: 
                    GM_weight_size = args.config[-1][1] * args.GM_n
                    fc_layer=nn.Sequential(nn.BatchNorm1d(64),nn.ReLU()).cuda(args.gpu)
                    out_GM = fc_layer(torch.mm(gm, torch.transpose(model.fc[0].weight[:, :GM_weight_size], 1, 0)) + model.fc[0].bias / 2)
                    out_GM=(torch.mm(out_GM, torch.transpose(model.fc[-1].weight, 0, 1)) + model.fc[-1].bias)
                    out_WM = fc_layer(torch.mm(wm, torch.transpose(model.fc[0].weight[:, GM_weight_size:],  1,0 ))+ model.fc[0].bias / 2)
                    out_WM=(torch.mm(out_WM, torch.transpose(model.fc[-1].weight, 0, 1)) + model.fc[-1].bias)
                    score_gm = sum([softmax(out_GM)[i][target[i]] for i in range(out_GM.size(0))])
                    score_wm = sum([softmax(out_WM)[i][target[i]] for i in range(out_WM.size(0))])
                    ratio_wm = score_wm / score_gm
                    ratio_gm = 1 / ratio_wm
                    if ratio_wm > 1:
                      coeff_wm = 1 - tanh(args.modu_coef * relu(ratio_wm))
                      coeff_gm = 1
                    else:
                      coeff_gm = 1 - tanh(args.modu_coef * relu(ratio_gm))
                      coeff_wm = 1
                    for name, parms in model.named_parameters():
                        if 'GM' in name:
                            parms.grad *= coeff_gm
                        if 'WM' in name:
                            parms.grad *= coeff_wm
                optimizer.step()
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(valid_loader):
                    x, target = data
                    x = x.cuda(args.gpu)
                    target = target.cuda(args.gpu)
                    _,_, output = model(x)
                ACC, TP, TN, FP, FN, sen, spe = cacu_metric(output, target)
                fpr, tpr, thresholds = metrics.roc_curve(target.cpu().detach().numpy(), F.softmax(output,dim=-1)[:, 1].cpu().detach().numpy(), pos_label=1)
                auc = metrics.auc(fpr, tpr)
            if args.lr_decay:
                scheduler.step(ACC)
            if epoch>0 and ACC >= Best_ACC_most:
                Best_ACC_most = ACC
                all_best_SEN[k] = TP / (TP + FN)
                all_best_SPE[k] = TN / (FP + TN)
                all_best_AUC[k] = auc
                all_best_ACC[k] = Best_ACC_most
                all_best_epoch[k]=epoch
            print('split:{} Epoch: {}  loss:{:.5f}  train_ACC:{:.5f} test_ACC:{:.5f} Best_ACC_most:{:.5f} '.format(k+1, epoch, total_loss / len(train_loader), train_ACC / num_train, ACC, Best_ACC_most))
            print('SEN: {:.5f}  SPE: {:.5f} auc:{:.5f}'.format(sen, spe, auc))
            print('TP:{}  TN:{}  FP:{}  FN:{} '.format(TP, TN, FP, FN))
    print('train:',num_train,'test:',num_valid)   
    print('ACC_{} Average: {:.5f}  std:{:.5f}'.format(np.round((all_best_ACC), 5), np.mean(all_best_ACC), (np.std(all_best_ACC))))
    print('SEN_{} Average: {:.5f} std: {:.5f}'.format(np.round((all_best_SEN),5), np.mean(all_best_SEN), (np.std(all_best_SEN))))
    print('SPE_{} Average: {:.5f} std: {:.5f}'.format(np.round((all_best_SPE),5), np.mean(all_best_SPE), (np.std(all_best_SPE))))
    print('AUC_{} Average: {:.5f} std: {:.5f}'.format(np.round((all_best_AUC),5), np.mean(all_best_AUC), (np.std(all_best_AUC))))

if __name__ == '__main__':
    import argparse 
    import ast
    parser = argparse.ArgumentParser()  
    parser.add_argument('--dataset', type=str, default='ADNI', help='Name of the dataset')
    parser.add_argument('--k_folds', nargs='+', type=int, default=[0, 1, 2, 3, 4], help='fold = 0,1,2,3,4')
    parser.add_argument('--GM_n', type=int, default=90, help='Number of GM ROIs')
    parser.add_argument('--WM_n', type=int, default=50, help='Number of WM ROIs')
    parser.add_argument('--modu_coef', type=float, default=0.4, help='modulation coefficient')
    parser.add_argument('--modulation_start', type=int, default=0, help='Start epoch for modulation')
    parser.add_argument('--modulation_end', type=int, default=200, help='End epoch for modulation')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--lr_decay', action='store_true', help='enable learning rate decay scheduler')
    parser.add_argument('--warm_up_epoch', type=int, default=5, help='warm up epoch')
    parser.add_argument('--gpu', type=str, default='cuda:0', help='choose gpu')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--end_epoch', type=int, default=200, help='Number of epoches')
    parser.add_argument('--config', type=ast.literal_eval, default='[[8, 8, 2, 6],[8, 8, 2, 6],[8, 8, 2, 6],]', help='Network configuration of each layer: in_channel, out_channel, num_head, temporal lag limit')
    args = parser.parse_args()
    print(args)
    main(args)


# python train.py --dataset 'ADNI' --GM_n 90 --WM_n 50 --config '[[8, 8, 2, 6],[8, 8, 2, 6],[8, 8, 2, 6]]' --modu_coef 0.4 --lr 0.1 --lr_decay --end_epoch 200 






