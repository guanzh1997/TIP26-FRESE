import argparse
import os, math, datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.KS import VADataset
from models.basic_model import AVNet
from utils.utils import setup_seed, weight_init
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import torch.nn.functional as F
from colorama import Fore, Style
from utils.utils import (
save_results_and_checkpoints,
get_modality_strength_masks,
FDFilter,
_init_proto_state_if_needed,
build_batch_prototypes,
_update_proto_bank_ema,
whmb_loss_with_protos,
paper_style_rol_block
)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', default=1, type=int)

    parser.add_argument('--data_root', default='/data/gzh/Imbalance/dataset/kinetics_sound', type=str)

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=1e-2, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='2', type=str, help='GPU ids')

    return parser.parse_args()

def train(args, epoch, net, device, train_dataloader, optimizer, scheduler, epoch_step_train, fd_filter):

    criterion = nn.CrossEntropyLoss()
    _total_loss = 0

    print('Start Training')
    pbar = tqdm(total=epoch_step_train, desc=f'Epoch {epoch + 1}/{args.epochs}', postfix=dict, mininterval=0.3, ncols=120)
    net.train()
    fd_filter.train()

    T = 2.0
    K_neg = 8  # 3 or 5
    margin_whmb = 0.2  # 0.1~0.3
    lambda_kd_logits = 0.5
    lambda_whmb = 0.2  # 0.1~0.3

    # ===== 全局原型 EMA 超参 =====
    EMA_BETA = 0.1
    EPS = 1e-6

    for step, bag in enumerate(train_dataloader):
        spec = bag[0].to(device)
        image = bag[1].to(device)
        label = bag[2].to(device)
        labels_idx = label

        optimizer.zero_grad()
        out_a, out_v, emb_a, emb_v = net(spec.unsqueeze(1).float(), image.float())

        # ===== 1. 原始分类损失 =====
        loss_a = criterion(out_a, labels_idx)
        loss_v = criterion(out_v, labels_idx)
        loss_f = criterion(out_a + out_v, labels_idx)

        # ===== 2. 动态判断强弱模态（样本级），logits 级别的强→弱蒸馏（KD） =====
        audio_stronger, video_stronger = get_modality_strength_masks(
            out_a.detach(),
            out_v.detach(),
            labels_idx)

        # log_p_a_T = F.log_softmax(out_a / T, dim=1)
        # log_p_v_T = F.log_softmax(out_v / T, dim=1)
        # p_a_T = F.softmax(out_a / T, dim=1)
        # p_v_T = F.softmax(out_v / T, dim=1)
        #
        # # 为了在一个 batch 内统一 KL，构造 student / teacher
        # student_logp = torch.empty_like(log_p_a_T)
        # teacher_prob = torch.empty_like(p_a_T)
        #
        # # 情况 A：audio 强 → video 学 audio
        # if audio_stronger.any():
        #     student_logp[audio_stronger] = log_p_v_T[audio_stronger]  # student: video
        #     teacher_prob[audio_stronger] = p_a_T[audio_stronger].detach()  # teacher: audio
        #
        # # 情况 B：video 强 → audio 学 video
        # if video_stronger.any():
        #     student_logp[video_stronger] = log_p_a_T[video_stronger]  # student: audio
        #     teacher_prob[video_stronger] = p_v_T[video_stronger].detach()  # teacher: video
        #
        # loss_kd_logits = F.kl_div(student_logp, teacher_prob, reduction="batchmean") * (T * T)

        # ===== 3. FD =====
        a_low, a_high = fd_filter(emb_a)
        v_low, v_high = fd_filter(emb_v)
        loss_rol = paper_style_rol_block(a_low, v_low, a_high, v_high)

        # ===== 初始化全局原型 bank（只做一次，挂到 net 上，跨 epoch 保持）=====
        # protp_bank_a/v: (6,512), proto_inited_a/v: (6,)
        _init_proto_state_if_needed(
            net=net,
            num_classes=out_a.shape[1],
            feat_dim=v_high.shape[1],
            device=device,
            dtype=v_high.dtype
        )

        # ===== 4. WHMB损失 =====
        loss_whmb_audio = torch.zeros((), device=device)
        loss_whmb_video = torch.zeros((), device=device)

        # 情况 A：audio 强 -> video 弱：约束 v_high
        if audio_stronger.any():
            m = audio_stronger
            protos_v_batch, valid_v = build_batch_prototypes(
                feats=v_high[m],
                labels=labels_idx[m],
                num_classes=out_a.shape[1],
                teacher_logits=out_a.detach()[m],
                T=2.0,
                eps=EPS,
                normalize=True
            )

            # net.proto_bank_v: 跨 batch、跨 epoch 一直保存的“全局每类中心向量”
            # protos_v_batch: 当前这一小撮样本里算出来的临时原型
            _update_proto_bank_ema(net.proto_bank_v, net.proto_inited_v, protos_v_batch, valid_v,
                                   beta=EMA_BETA, eps=EPS)

            # 在全局 bank 没覆盖所有类之前，先用 batch 原型（避免未初始化类影响 hard neg）
            if net.proto_inited_v.all():
                protos_for_loss = net.proto_bank_v
            else:
                protos_for_loss = protos_v_batch

            loss_whmb_audio = whmb_loss_with_protos(
                student_high=v_high[m],
                protos=protos_for_loss,
                teacher_logits=out_a.detach()[m],
                labels_idx=labels_idx[m],
                K=K_neg,
                margin=margin_whmb,
                eps=EPS
            )

        # 情况 B：video 强 -> audio 弱：约束 a_high
        if video_stronger.any():
            m = video_stronger
            protos_a_batch, valid_a = build_batch_prototypes(
                feats=a_high[m],
                labels=labels_idx[m],
                num_classes=out_v.shape[1],
                teacher_logits=out_v.detach()[m],
                T=2.0,
                eps=EPS,
                normalize=True
            )

            _update_proto_bank_ema(net.proto_bank_a, net.proto_inited_a, protos_a_batch, valid_a,
                                   beta=EMA_BETA, eps=EPS)

            if net.proto_inited_a.all():
                protos_for_loss = net.proto_bank_a
            else:
                protos_for_loss = protos_a_batch

            loss_whmb_video = whmb_loss_with_protos(
                student_high=a_high[m],
                protos=protos_for_loss,
                teacher_logits=out_v.detach()[m],  # teacher: video 选 hard neg
                labels_idx=labels_idx[m],
                K=K_neg,
                margin=margin_whmb,
                eps=EPS
            )

        loss_whmb = 0.5 * (loss_whmb_audio + loss_whmb_video)

        # ===== 5. 总损失 =====
        loss = (
            loss_a + loss_v + loss_f
            # + lambda_kd_logits * loss_kd_logits
            + lambda_whmb * (loss_whmb + loss_rol)
        )

        loss.backward()
        optimizer.step()

        _total_loss += loss.item()

        pbar.set_postfix(**{'train_loss': _total_loss / (step + 1), 'lr': optimizer.param_groups[0]['lr']})
        pbar.update(1)

    pbar.close()
    scheduler.step() # 学习率更新

    return _total_loss / epoch_step_train

def test(args, epoch, net, device, test_dataloader, optimizer, epoch_step_test):

    criterion = nn.CrossEntropyLoss()
    n_classes = 31
    _loss = 0

    # ---------- accuracy统计 ----------
    num = [0.0 for _ in range(n_classes)]

    acc_a = [0.0 for _ in range(n_classes)]   # audio
    acc_v = [0.0 for _ in range(n_classes)]   # visual
    acc_f = [0.0 for _ in range(n_classes)]   # fusion

    # ---------- mAP统计 ----------
    all_probs_a = []
    all_probs_v = []
    all_probs_f = []
    all_labels = []

    print('Start Testing')
    pbar = tqdm(total=epoch_step_test, desc=f'Epoch {epoch + 1}/{args.epochs}', postfix=dict, mininterval=0.3, ncols=120)
    net.eval()

    for step, (spec, image, label) in enumerate(test_dataloader):
        with torch.no_grad():
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)
            labels_idx = label

            out_a, out_v, emb_a, emb_v = net(spec.unsqueeze(1).float(), image.float())
            out_f = out_a + out_v

            # loss
            loss_a = criterion(out_a, labels_idx)
            loss_v = criterion(out_v, labels_idx)
            loss_f = criterion(out_f, labels_idx)
            loss = loss_a + loss_v + loss_f
            _loss += loss.item()

            # probs
            probs_a = F.softmax(out_a, dim=1)
            probs_v = F.softmax(out_v, dim=1)
            probs_f = F.softmax(out_f, dim=1)

            # preds
            preds_a = torch.argmax(probs_a, dim=1)
            preds_v = torch.argmax(probs_v, dim=1)
            preds_f = torch.argmax(probs_f, dim=1)

            # 更新每类accuracy统计
            correct_a = (preds_a == label).float()
            correct_v = (preds_v == label).float()
            correct_f = (preds_f == label).float()

            for i in range(len(label)):
                cls = label[i].item()
                num[cls] += 1
                acc_a[cls] += correct_a[i].item()
                acc_v[cls] += correct_v[i].item()
                acc_f[cls] += correct_f[i].item()

            all_probs_a.append(probs_a.detach().cpu().numpy())
            all_probs_v.append(probs_v.detach().cpu().numpy())
            all_probs_f.append(probs_f.detach().cpu().numpy())
            all_labels.append(label.detach().cpu().numpy())

            pbar.set_postfix(**{'test_loss': _loss / (step + 1), 'lr': optimizer.param_groups[0]['lr']})
            pbar.update(1)

    pbar.close()

    # 拼接
    all_probs_a = np.concatenate(all_probs_a, axis=0)
    all_probs_v = np.concatenate(all_probs_v, axis=0)
    all_probs_f = np.concatenate(all_probs_f, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # ---------- 计算mAP ----------
    def compute_map(all_probs, all_labels, n_classes):
        mAP = 0.0
        for i in range(n_classes):
            label_binary = (all_labels == i).astype(int)
            ap = average_precision_score(label_binary, all_probs[:, i])
            mAP += ap
        return mAP / n_classes

    mAP_a = compute_map(all_probs_a, all_labels, n_classes)
    mAP_v = compute_map(all_probs_v, all_labels, n_classes)
    mAP_f = compute_map(all_probs_f, all_labels, n_classes)

    # ---------- 计算accuracy ----------
    accuracy_a = sum(acc_a) / sum(num)
    accuracy_v = sum(acc_v) / sum(num)
    accuracy_f = sum(acc_f) / sum(num)

    return _loss / epoch_step_test, accuracy_f, mAP_f, accuracy_a, mAP_a, accuracy_v, mAP_v


if __name__ == '__main__':
    # ---------------------参数----------------------
    args = get_arguments()
    print(args)

    # ---------------------设备----------------------
    setup_seed(args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    print('GPU设备数量为:', torch.cuda.device_count())
    gpu_ids = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0')

    # ---------------------模型----------------------
    net = AVNet(args)
    net.apply(weight_init)
    net.to(device) # 将模型在指定的device上进行初始化，这里是3号GPU，索引为0号
    net = torch.nn.DataParallel(net, device_ids=gpu_ids) # 对模型进行封装，分发到多个GPU上运行
    net.cuda()
    fd_filter = FDFilter().cuda()

    # ---------------------数据----------------------
    train_dataset = VADataset(args, mode='train')
    test_dataset = VADataset(args, mode='test')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=32, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32, pin_memory=True)
    print('训练数据量: ', len(train_dataset))
    print('测试数据量: ', len(test_dataset))
    epoch_step_train = len(train_dataset) // train_dataloader.batch_size
    epoch_step_test = math.ceil(len(test_dataset) / test_dataloader.batch_size)  # 因为验证集没有drop_last，所以多一个step，向上取整

    # --------------------优化器---------------------
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    # ------------------训练and验证-------------------
    if True:
        best_acc = 0
        best_acc_epoch = 0
        per_save_epoch = 30

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_file = f'./result/result_{timestamp}.txt'
        with open(results_file, 'w') as f:
            f.write("Epoch | accuracy_f | mAP_f | accuracy_a | mAP_a | accuracy_v | mAP_v\n")
            f.write("-" * 80 + "\n")
        ckpt_dir = f'./checkpoint/{timestamp}'
        os.makedirs(ckpt_dir, exist_ok=True)

        for epoch in range(args.epochs):
            print(f"{Fore.BLUE}Current Epoch: {epoch + 1}{Style.RESET_ALL}")
            mean_train_loss = train(args, epoch, net, device, train_dataloader, optimizer, scheduler, epoch_step_train, fd_filter)
            mean_test_loss, accuracy_f, mAP_f, accuracy_a, mAP_a, accuracy_v, mAP_v = test(args, epoch, net, device, test_dataloader, optimizer, epoch_step_test)

            print(f"{Fore.RED}********************************************************************{Style.RESET_ALL}")
            print(f"{Fore.RED}Now train_loss: {mean_train_loss:.4f} || Now test_loss: {mean_test_loss:.4f}{Style.RESET_ALL}")
            print(f"{Fore.RED}Now test_acc:  {accuracy_f:.4f} || Now test_map:  {mAP_f:.4f}{Style.RESET_ALL}")

            best_acc, best_acc_epoch = save_results_and_checkpoints(
                epoch + 1,
                accuracy_f, mAP_f, accuracy_a, mAP_a, accuracy_v, mAP_v,
                results_file, ckpt_dir,
                net, optimizer, scheduler,
                best_acc, best_acc_epoch, per_save_epoch
            )

            print(f"{Fore.RED}Best Accuracy: {best_acc:.4f}, Best Epoch: {best_acc_epoch}{Style.RESET_ALL}")
            print(f"{Fore.RED}********************************************************************{Style.RESET_ALL}")