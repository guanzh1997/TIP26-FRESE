import torch
import torch.nn as nn
import numpy as np
import random
import  os
import torch.nn.functional as F

def setup_seed1(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def save_results_and_checkpoints(
        epoch, accuracy_f, mAP_f, accuracy_a, mAP_a, accuracy_v, mAP_v,
        results_file, ckpt_dir,
        net, optimizer, scheduler,
        best_acc, best_acc_epoch, per_save_epoch
    ):

    # ---------------- 写入 txt 结果 ---------------- #
    with open(results_file, 'a') as f:
        f.write(
            f"{epoch:5d} | "
            f"{accuracy_f:11.4f} | {mAP_f:11.4f} | {accuracy_a:11.4f} | {mAP_a:11.4f} | {accuracy_v:11.4f} | {mAP_v:11.4f}\n"
            # f"{test_acc:11.4f} | {test_map:11.4f}\n"
        )

    # ---------------- 保存 best 模型 ---------------- #
    if accuracy_f > best_acc:
        best_acc = float(accuracy_f)
        best_acc_epoch = epoch

        best_ckpt_path = os.path.join(ckpt_dir, 'best.pth')
        torch.save(
            {
                'epoch': best_acc_epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            },
            best_ckpt_path
        )
        # print(f"{Fore.GREEN}==> Saved new best model to {best_ckpt_path}{Style.RESET_ALL}")

    # ---------------- 每 per_save_epoch 保存一次模型 ---------------- #
    if epoch % per_save_epoch == 0:
        ckpt_path = os.path.join(
            ckpt_dir,
            f'epoch_{epoch:03d}_acc{accuracy_f:.4f}.pth'
        )

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'acc_f': float(accuracy_f),
                'best_acc': best_acc,
                'best_acc_epoch': best_acc_epoch,
            },
            ckpt_path
        )
        # print(f"{Fore.YELLOW}==> Saved checkpoint at epoch {epoch} to {ckpt_path}{Style.RESET_ALL}")

    return best_acc, best_acc_epoch


def get_modality_strength_masks(out_a: torch.Tensor,
                                out_v: torch.Tensor,
                                labels_idx: torch.Tensor):
    """
    根据你说的规则动态判断强弱模态（样本级）：
    1) 谁预测对、谁预测错（只有一边对的一方是强模态）；
    2) 如果都对或者都错，再看谁对目标类的置信度更高。

    输入:
        out_a: [B, C] audio logits
        out_v: [B, C] visual logits
        labels_idx: [B] 真实标签 index

    返回:
        audio_stronger: [B] bool，True 表示这一样本 audio 更强
        video_stronger: [B] bool，True 表示这一样本 video 更强
    """
    with torch.no_grad():
        # 预测类别
        pred_a = out_a.argmax(dim=1)   # [B]
        pred_v = out_v.argmax(dim=1)

        correct_a = (pred_a == labels_idx)   # [B] bool
        correct_v = (pred_v == labels_idx)

        B = labels_idx.size(0)
        idx = torch.arange(B, device=labels_idx.device)

        # 目标类置信度（用 softmax 即可）
        p_a = F.softmax(out_a, dim=1)  # [B, C]
        p_v = F.softmax(out_v, dim=1)
        conf_a = p_a[idx, labels_idx]  # [B]
        conf_v = p_v[idx, labels_idx]

        # 规则 1：谁预测对、谁预测错
        audio_stronger_1 = correct_a & (~correct_v)
        video_stronger_1 = correct_v & (~correct_a)

        # 规则 2：都对 或 都错 时，看置信度
        both_correct = correct_a & correct_v
        both_wrong   = (~correct_a) & (~correct_v)
        tie_region   = both_correct | both_wrong

        audio_stronger_2 = tie_region & (conf_a > conf_v)
        video_stronger_2 = tie_region & (conf_v >= conf_a)

        audio_stronger = audio_stronger_1 | audio_stronger_2
        video_stronger = video_stronger_1 | video_stronger_2

    return audio_stronger, video_stronger


class FDFilter(nn.Module):
    """
    简化版 FD-CMKD 里的频域滤波器：
    对 embedding 向量做 1D rFFT，将频域分为低频 / 高频两部分，
    再分别做 irFFT 得到低频特征和高频特征。
    """
    def __init__(self, low_ratio: float = 0.5):
        super(FDFilter, self).__init__()
        # 低频占比（0~1），0.5 表示前一半频率作为低频
        self.low_ratio = low_ratio

    def forward(self, x: torch.Tensor):
        # x: 原始embedding(位于特征域)
        # freq: x的频谱, 它是复数由幅度和相位构成(位于频域)
        # freq_low, freq_high: 低频频谱, 高频频谱(位于频域)
        # x_low, x_high: 由低频/高频重构的信号(特征域)

        # 对特征做 1D FFT（沿特征维度）
        freq = torch.fft.rfft(x, dim=1)  # (64,257)
        B, F = freq.shape
        split = int(F * self.low_ratio)

        # 构造低频 / 高频 mask（[F]，自动 broadcast 到 [B, F]）
        device = x.device
        low_mask = torch.zeros(F, device=device)
        high_mask = torch.zeros(F, device=device)
        low_mask[:split] = 1.0
        high_mask[split:] = 1.0

        # 频域滤波
        freq_low = freq * low_mask # (64,257)
        freq_high = freq * high_mask # (64,257)

        # 反变换回时域 / 特征域
        x_low = torch.fft.irfft(freq_low, n=x.shape[1], dim=1).real   # (64,512)
        x_high = torch.fft.irfft(freq_high, n=x.shape[1], dim=1).real # (64,512)

        return x_low, x_high

@torch.no_grad()
def _init_proto_state_if_needed(net, num_classes: int, feat_dim: int, device, dtype):
    """
    在 net 上挂载全局 prototype bank，保证跨 epoch 的 train() 调用也能保留状态。
    """
    if not hasattr(net, "proto_bank_v"):
        net.proto_bank_v = torch.zeros(num_classes, feat_dim, device=device, dtype=dtype)
        net.proto_inited_v = torch.zeros(num_classes, device=device, dtype=torch.bool)

    if not hasattr(net, "proto_bank_a"):
        net.proto_bank_a = torch.zeros(num_classes, feat_dim, device=device, dtype=dtype)
        net.proto_inited_a = torch.zeros(num_classes, device=device, dtype=torch.bool)

@torch.no_grad()
def build_batch_prototypes(
    feats: torch.Tensor,          # [B', D]  (e.g., v_high[m])
    labels: torch.Tensor,         # [B']
    num_classes: int,
    teacher_logits: torch.Tensor = None,  # [B', C] optional
    T: float = 1.0,
    eps: float = 1e-6,
    normalize: bool = True,
):
    """
    Return:
      protos: [C, D] batch prototypes
      proto_valid: [C] bool mask, class appears in this subset
    """
    if normalize:
        feats_n = F.normalize(feats, dim=1) # (35,512)
    else:
        feats_n = feats

    Bp, D = feats_n.shape
    C = num_classes
    device = feats_n.device

    # sample weights
    if teacher_logits is None:
        w = torch.ones(Bp, device=device, dtype=feats_n.dtype)
    else:
        p = F.softmax(teacher_logits / T, dim=1)          # [B', C]
        w = p.gather(1, labels.view(-1, 1)).squeeze(1)    # [B'] 取出它真实类别 labels[i] 对应的概率
        # 可选：避免极小权重
        # w = torch.clamp(w, min=0.05)

    # one-hot for aggregation
    onehot = F.one_hot(labels, num_classes=C).to(feats_n.dtype)  # [B', C]

    # weighted sum: [C, D] = (onehot^T @ (w * feats))
    weighted_feats = feats_n * w.view(-1, 1)          # [B', D]
    protos = onehot.t().matmul(weighted_feats)        # [C, D]

    # weight sum per class: [C]
    wsum = onehot.t().matmul(w.view(-1, 1)).squeeze(1)  # [C]
    proto_valid = wsum > eps

    # normalize by sum weights
    protos = protos / (wsum.view(-1, 1) + eps)

    # if normalize:
    #     protos = F.normalize(protos, dim=1)

    return protos, proto_valid

@torch.no_grad()
def _update_proto_bank_ema(proto_bank, proto_inited, protos_batch, valid_batch, beta=0.1, eps=1e-6):
    """
    EMA 更新全局原型 bank：只更新 valid 的类；首次出现的类直接赋值。
    protos_batch 默认已经是 normalize 后的（你的 build_batch_prototypes normalize=True）。
    """
    v = valid_batch

    # 第一次出现：直接赋值
    first = v & (~proto_inited)
    if first.any():
        proto_bank[first] = protos_batch[first]
        proto_inited[first] = True

    # 已初始化：EMA 更新
    upd = v & proto_inited
    if upd.any():
        proto_bank[upd] = (1 - beta) * proto_bank[upd] + beta * protos_batch[upd]
        proto_bank[upd] = F.normalize(proto_bank[upd], dim=1, eps=eps)


def whmb_loss_with_protos(
    student_high: torch.Tensor,     # (35,512)
    protos: torch.Tensor,           # (6,512)  batch prototypes
    teacher_logits: torch.Tensor,   # (35,6) (detach outside)
    labels_idx: torch.Tensor,       # (35,)
    K: int = 3,
    margin: float = 0.2,
    eps: float = 1e-6,
):
    # normalize for cosine-style logits (可选，但建议)
    z = F.normalize(student_high, dim=1)              # (35,512)
    P = F.normalize(protos, dim=1)                    # (6,512)

    # logits via prototype similarity: (35,6)
    s = z.matmul(P.t()) # 计算student与所有类原型的相似度

    # 选出 logits 最高的 Top-K 个非 GT 类 作为困难负类
    Bp, C = teacher_logits.shape
    t = teacher_logits.clone()
    t[torch.arange(Bp, device=t.device), labels_idx] = -1e9
    neg_idx = torch.topk(t, k=K, dim=1, largest=True).indices # (35,2)

    # # 选出 logits 最低的 Top-K 个非 GT 类 作为困难负类
    # Bp, C = teacher_logits.shape
    # t = teacher_logits.clone()
    # t[torch.arange(Bp, device=t.device), labels_idx] = float("inf")
    # neg_idx = torch.argsort(t, dim=1)[:, :K]  # [Bp, K] 取最小 K 个

    # (35,): 取出每个样本 GT 类别对应的相似度分数
    s_pos = s.gather(1, labels_idx.view(-1, 1)).squeeze(1)
    # (35,2): 取出每个样本的 K 个 hard negative 类别对应的相似度分数
    s_neg = s.gather(1, neg_idx)

    # hinge margin: max(0, s_neg - s_pos + margin)
    # 要求正类分数 s_pos 必须比负类分数 s_neg 大至少一个 margin
    # 每个样本的弱模态高频特征必须更贴近 GT 类原型，并且相对 teacher 选出的 Top-K 困难负类原型至少拉开 margin 的相似度间隔
    loss = F.relu(s_neg - s_pos.view(-1, 1) + margin).mean()
    return loss


def paper_style_rol_block(a_low, v_low, a_high, v_high):
    """
    Exact paper-style Representation Orthogonality Loss (SSE form)

    L_ROL = sum_{i=1}^4 sum_{j=1}^4 ( V_sim(i,j) - A(i,j) )^2
    """

    # ---------- 1. 组织表示（顺序必须与 A 对齐） ----------
    # shared: a_low, v_low
    # specific: a_high, v_high
    Z = [a_low, v_low, a_high, v_high]   # each: [B, D]
    Z = [F.normalize(z, dim=1) for z in Z]

    # ---------- 2. 计算相似度矩阵 V_sim ----------
    # W: [B, 4, D]
    W = torch.stack(Z, dim=1)

    # per-sample Gram matrix: [B, 4, 4]
    V_sim = torch.bmm(W, W.transpose(1, 2))

    # batch-mean similarity matrix: [4, 4]
    V_sim = V_sim.mean(dim=0)

    # ---------- 3. 构造目标矩阵 A（论文中的 A） ----------
    # block structure: shared / specific
    A = torch.tensor([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], device=V_sim.device, dtype=V_sim.dtype)

    # ---------- 4. 论文公式：平方误差和 ----------
    loss = ((V_sim - A) ** 2).sum()

    return loss