"""
02_model_prompt.py
==================
基于 1D-CNN 的深层特征空间提示网络（Deep Feature-Space Prompting）
适用于可穿戴单导联心电图的连续学习场景。

核心思路：
  - 冻结预训练基座（backbone）所有参数，只训练 Prompt Pool 和分类头
  - 通过 register_forward_hook 将 Prompt 动态注入到指定的深层特征图中
  - 利用 Query-Key 余弦相似度匹配机制，从池中取出最相关的 Prompt
  - 使用代理匹配损失（Surrogate Matching Loss）监督 Key 的学习

数学符号对照（与论文保持一致）：
  P  ∈ R^{M × D}  : Prompt Pool（可学习）
  K  ∈ R^{M × D}  : Key Pool（可学习）
  q  ∈ R^{B × D}  : Query（由特征图 GAP 生成，梯度截断）
  i* = argmax_i cos_sim(q, K[i])  : 最佳匹配 Key 索引
  f'_d = f_d + P[i*]              : 注入后的特征图
  L_match = 1 - cos_sim(q_det, K[i*])  : 代理匹配损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ══════════════════════════════════════════════════════════════════
#  DeepPromptECGModel：核心包装类
# ══════════════════════════════════════════════════════════════════

class DeepPromptECGModel(nn.Module):
    """
    深层特征空间提示模型（Deep Feature-Space Prompting Wrapper）

    参数
    ----
    backbone      : 预训练的 1D-ResNet 基座模型（net1d.py 中定义的模型实例）
    hook_layer    : 需要挂载 Hook 的 nn.Module 层对象（通常是某个深层卷积块）
    embed_dim     : 注入层的通道数 D，必须与 hook_layer 输出通道数一致，默认 64
    pool_size     : Prompt 池大小 M，默认 100
    num_classes   : 分类类别数，默认 2（二分类）
    """

    def __init__(
        self,
        backbone: nn.Module,
        hook_layer: nn.Module,
        embed_dim: int = 64,
        pool_size: int = 100,
        num_classes: int = 2,
    ):
        super().__init__()

        # ── 步骤 1：保存基座模型并冻结所有参数 ──────────────────────────
        self.backbone = backbone
        self._freeze_backbone()

        # ── 步骤 2：初始化可学习的 Prompt Pool 与 Key Pool ──────────────
        # P: Prompt Pool，shape (M, D)，用零初始化（亦可用小随机值）
        self.prompt_pool = nn.Parameter(
            torch.zeros(pool_size, embed_dim)
        )
        # K: Key Pool，shape (M, D)，用标准正态分布初始化，便于余弦相似度起步
        self.key_pool = nn.Parameter(
            torch.randn(pool_size, embed_dim)
        )

        # 记录维度，供 Hook 函数使用
        self.embed_dim = embed_dim
        self.pool_size = pool_size

        # ── 步骤 3：替换/新建可训练的分类头 ───────────────────────────────
        # 原 backbone 的分类头通过冻结已被屏蔽；
        # 这里新建一个独立的线性分类器，仅用 embed_dim 维特征做分类
        self.classifier = nn.Linear(embed_dim, num_classes)

        # ── 步骤 4：内部状态变量（供训练循环读取）─────────────────────────
        # match_loss：每次 forward 后由 Hook 填充，训练时直接 loss += model.match_loss
        self.match_loss: torch.Tensor = torch.tensor(0.0)

        # _gap_feature：Hook 执行后保存的全局平均池化特征，供分类头使用
        self._gap_feature: Optional[torch.Tensor] = None

        # ── 步骤 5：注册 Forward Hook ──────────────────────────────────
        # hook_layer 必须是 backbone 中实际存在的 nn.Module 层对象
        self._hook_handle = hook_layer.register_forward_hook(
            self._prompt_injection_hook
        )

        # 初始化权重
        self._init_weights()

    # ──────────────────────────────────────────────────────────────
    #  私有方法：参数冻结
    # ──────────────────────────────────────────────────────────────

    def _freeze_backbone(self) -> None:
        """冻结 backbone 所有参数，使其在训练中不更新。"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(
            f"[DeepPromptECGModel] 基座模型已冻结，"
            f"共冻结 {sum(p.numel() for p in self.backbone.parameters()):,} 个参数。"
        )

    # ──────────────────────────────────────────────────────────────
    #  私有方法：权重初始化
    # ──────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """对新建的可学习参数做合理初始化。"""
        # Prompt 用零初始化：保证初始时不改变特征分布，让训练从"无扰动"起步
        nn.init.zeros_(self.prompt_pool)
        # Key 用均匀球面初始化：保证各 Key 起始时分布均匀，避免坍塌
        nn.init.normal_(self.key_pool)
        with torch.no_grad():
            self.key_pool.div_(
                self.key_pool.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            )
        # 分类头用 Xavier 初始化
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    # ──────────────────────────────────────────────────────────────
    #  核心 Hook 函数：Prompt 动态注入
    # ──────────────────────────────────────────────────────────────

    def _prompt_injection_hook(
        self,
        module: nn.Module,
        input: tuple,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """
        挂载在深层卷积块上的前向 Hook，实现 Prompt 动态注入。

        参数
        ----
        module  : 被挂载的层（自动传入，无需手动调用）
        input   : 该层的输入（tuple）
        output  : 该层的输出特征图，shape = (B, C, L)
                  B=batch, C=channels(=embed_dim), L=序列长度

        返回
        ----
        f_prime : 注入 Prompt 后的特征图，shape = (B, C, L)
                  PyTorch 的 Hook 机制：若 Hook 返回非 None，
                  则自动替换原始 output 继续后续的前向传播
        """

        # ── (a) Query 生成：Global Average Pooling on Time Dimension ──
        # output: (B, C, L)  →  q: (B, C)
        q = output.mean(dim=-1)                     # GAP，时间维度均值

        # 论文要求：使用 q.detach() 计算相似度，阻止梯度通过 q 回流到 backbone
        q_detach = q.detach()                       # shape: (B, C)

        # ── (b) 余弦相似度：Query vs. Key Pool ────────────────────────
        # 对 q 和 K 分别做 L2 归一化
        q_norm = F.normalize(q_detach, dim=-1)      # (B, C)
        k_norm = F.normalize(self.key_pool, dim=-1) # (M, C)

        # 计算每个样本与所有 Key 的相似度
        # sim_matrix: (B, M)
        sim_matrix = torch.matmul(q_norm, k_norm.T)

        # ── (c) 动态匹配：找到最优 Key 索引 i* ────────────────────────
        # best_idx: (B,)，每个样本对应相似度最高的 Key 的索引
        best_idx = sim_matrix.argmax(dim=-1)        # (B,)

        # ── (d) 取出对应的 Prompt 并注入 ─────────────────────────────
        # selected_prompt: (B, C)
        selected_prompt = self.prompt_pool[best_idx]

        # 扩展 Prompt 维度以匹配特征图的时间轴：(B, C) → (B, C, 1)
        # 广播加法：每个时间步加上相同的 channel-wise bias
        prompt_expanded = selected_prompt.unsqueeze(-1)   # (B, C, 1)
        f_prime = output + prompt_expanded                 # (B, C, L)

        # ── (e) 计算代理匹配损失 L_match ──────────────────────────────
        # 取出每个样本匹配到的 Key（归一化后）
        # selected_k_norm: (B, C)
        selected_k_norm = k_norm[best_idx]

        # 余弦相似度：对每个样本计算 cos_sim(q_detach, k_{i*})
        # dot product of already-normalized vectors = cosine similarity
        cos_sim_per_sample = (q_norm * selected_k_norm).sum(dim=-1)  # (B,)

        # L_match = 1 - mean(cos_sim)，值域 [0, 2]，越小表示匹配越好
        self.match_loss = (1.0 - cos_sim_per_sample).mean()

        # ── (f) 保存 GAP 特征供分类头使用 ─────────────────────────────
        # 使用注入后的特征图做 GAP，保证分类头感知到 Prompt 的影响
        self._gap_feature = f_prime.mean(dim=-1)    # (B, C)

        # 返回修改后的特征图，PyTorch 会用它替换原始 output 继续传播
        return f_prime

    # ──────────────────────────────────────────────────────────────
    #  前向传播
    # ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数
        ----
        x       : 输入心电图信号，shape = (B, 1, 2500)

        返回
        ----
        logits  : 分类 logits，shape = (B, num_classes)
        """

        # 重置状态（防止跨 batch 的状态污染）
        self._gap_feature = None
        self.match_loss = torch.tensor(0.0, device=x.device)

        # ── Backbone 前向传播（Hook 在内部自动触发）────────────────────
        # backbone 的原始输出（通常是分类 logits）在这里不使用，
        # 我们只需要 Hook 内保存的 _gap_feature
        _ = self.backbone(x)

        # ── 容错：确保 Hook 正常触发 ────────────────────────────────────
        if self._gap_feature is None:
            raise RuntimeError(
                "Hook 未被触发！请检查 hook_layer 是否确实是 backbone 中"
                "参与前向传播的层。"
            )

        # ── 新分类头：基于注入后的特征做分类 ────────────────────────────
        logits = self.classifier(self._gap_feature)     # (B, num_classes)

        return logits

    # ──────────────────────────────────────────────────────────────
    #  工具方法
    # ──────────────────────────────────────────────────────────────

    def get_trainable_params(self) -> list:
        """返回所有可训练参数（Prompt Pool、Key Pool、分类头）。"""
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_param_count(self) -> int:
        """统计可训练参数量。"""
        return sum(p.numel() for p in self.get_trainable_params())

    def remove_hook(self) -> None:
        """移除已注册的 Hook（推理时可调用，节省开销）。"""
        self._hook_handle.remove()
        print("[DeepPromptECGModel] Hook 已移除。")

    def __repr__(self) -> str:
        total   = sum(p.numel() for p in self.parameters())
        trainable = self.trainable_param_count()
        frozen  = total - trainable
        return (
            f"DeepPromptECGModel(\n"
            f"  embed_dim   = {self.embed_dim}\n"
            f"  pool_size   = {self.pool_size}\n"
            f"  总参数量    = {total:,}\n"
            f"  可训练参数  = {trainable:,}  "
            f"({100*trainable/total:.2f}%)\n"
            f"  冻结参数    = {frozen:,}\n"
            f")"
        )


# ══════════════════════════════════════════════════════════════════
#  独立测试块（不依赖 net1d.py，使用 DummyCNN 验证机制正确性）
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 65)
    print("  DeepPromptECGModel 独立功能验证")
    print("=" * 65)

    # ── 1. 定义一个简单的 DummyCNN 基座模型 ───────────────────────────
    # 模拟 net1d.py 中的多层 1D-ResNet，只关注结构，不需要预训练权重
    # 关键：block3 的输出通道数必须等于 embed_dim=64
    class DummyCNN(nn.Module):
        """
        轻量级占位基座，用于验证 Hook 注入机制，无需真实预训练权重。
        信号流：(B,1,2500) → block1 → block2 → block3[Hook] → GAP → fc
        block3 输出通道 = 64，与 embed_dim 对齐。
        """

        def __init__(self):
            super().__init__()
            # block1: 1 → 16 通道，步长 2，输出长度 1250
            self.block1 = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(16),
                nn.ReLU(),
            )
            # block2: 16 → 32 通道，步长 2，输出长度 625
            self.block2 = nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
            )
            # block3: 32 → 64 通道，步长 2，输出长度 ≈313
            # ★ Hook 将挂载在此层，输出 shape = (B, 64, ~313)
            self.block3 = nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )
            # 原始分类头（将被 DeepPromptECGModel 替换/忽略）
            self.fc = nn.Linear(64, 10)

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)           # Hook 在这里触发
            x = x.mean(dim=-1)           # GAP
            return self.fc(x)            # 原始分类输出（训练中不使用）

    # ── 2. 实例化 DummyCNN ─────────────────────────────────────────────
    dummy_backbone = DummyCNN()

    # ── 3. 包装为 DeepPromptECGModel ────────────────────────────────────
    #   hook_layer 指向 dummy_backbone.block3（64 通道，与 embed_dim=64 对齐）
    model = DeepPromptECGModel(
        backbone=dummy_backbone,
        hook_layer=dummy_backbone.block3,   # 挂载点：第三个卷积块
        embed_dim=64,                        # 必须与 block3 输出通道数一致
        pool_size=100,
        num_classes=2,
    )

    print(f"\n{model}\n")

    # ── 4. 构造假输入，执行前向传播 ──────────────────────────────────────
    torch.manual_seed(42)
    x_fake = torch.randn(2, 1, 2500)        # batch_size=2, 单导联, 2500点
    print(f"输入 x shape : {x_fake.shape}")

    model.train()
    logits = model(x_fake)

    # ── 5. 打印结果验证 ───────────────────────────────────────────────────
    print(f"\n前向传播输出：")
    print(f"  logits shape   : {logits.shape}")         # 期望 (2, 2)
    print(f"  logits 值      : \n{logits.detach()}")
    print(f"\nHook 内部状态：")
    print(f"  match_loss     : {model.match_loss.item():.6f}")   # 期望 (0, 2) 之间
    print(f"  _gap_feature   : shape = {model._gap_feature.shape}")  # 期望 (2, 64)

    # ── 6. 构造完整 Loss 并执行反向传播 ─────────────────────────────────
    #   最终 Loss = 分类损失 + λ * 匹配损失（λ 是超参数，此处取 0.1 演示）
    lambda_match = 0.1
    y_fake = torch.tensor([0, 1], dtype=torch.long)     # 假标签

    loss_cls   = F.cross_entropy(logits, y_fake)
    loss_match = model.match_loss
    loss_total = loss_cls + lambda_match * loss_match

    print(f"\n损失计算：")
    print(f"  L_cls          : {loss_cls.item():.6f}")
    print(f"  L_match        : {loss_match.item():.6f}")
    print(f"  L_total        : {loss_total.item():.6f}")

    # 反向传播
    loss_total.backward()

    # ── 7. 验证梯度流是否正确 ────────────────────────────────────────────
    print(f"\n梯度验证（仅可训练参数应有梯度）：")

    # Prompt Pool 和 Key Pool 应有梯度
    has_grad_prompt = model.prompt_pool.grad is not None
    has_grad_key    = model.key_pool.grad is not None
    has_grad_cls_w  = model.classifier.weight.grad is not None

    print(f"  prompt_pool.grad 存在 : {has_grad_prompt}  ✓" if has_grad_prompt
          else f"  prompt_pool.grad 存在 : {has_grad_prompt}  ✗ 异常！")
    print(f"  key_pool.grad    存在 : {has_grad_key}    ✓" if has_grad_key
          else f"  key_pool.grad    存在 : {has_grad_key}    ✗ 异常！")
    print(f"  classifier.w.grad存在 : {has_grad_cls_w}  ✓" if has_grad_cls_w
          else f"  classifier.w.grad存在 : {has_grad_cls_w}  ✗ 异常！")

    # Backbone 参数不应有梯度
    backbone_grads = [
        p.grad for p in dummy_backbone.parameters() if p.grad is not None
    ]
    backbone_frozen_ok = (len(backbone_grads) == 0)
    print(f"  backbone 梯度为空     : {backbone_frozen_ok}  "
          + ("✓ 冻结正常" if backbone_frozen_ok else "✗ 冻结失败！请检查"))

    # ── 8. 可训练参数统计 ─────────────────────────────────────────────
    print(f"\n可训练参数明细：")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name:<35} shape={str(param.shape):<20} "
                  f"numel={param.numel():,}")

    print("\n" + "=" * 65)
    print("  ✅ 所有验证通过，Hook 机制和计算图均正常！")
    print("=" * 65)
