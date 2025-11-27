import numpy as np
import matplotlib.pyplot as plt
"画的sigmoid曲线随参数变化的图"
# 定义 Sigmoid 函数：σ((M - τ)/s)
def sigmoid(M, tau=0.05, s=0.02):
    z = (M - tau) / (s + 1e-12)
    return 1.0 / (1.0 + np.exp(-z))

# M 范围（-0.5 到 1.5，用于显示完整形态）
M = np.linspace(-0.5, 1.5, 1000)

# 参数组合 + 标准线
param_list = [
    (0.02, 0.01),
    (0.02, 0.02),
    (0.05, 0.02),
    (0.05, 0.05),
    (0.10, 0.02),
    (0.0, 1.0),  # 标准 Sigmoid 线
]

save_path = '/root/autodl-fs/sigmoid_curve.png'  # 保存路径，可自行修改

plt.figure(figsize=(10, 5))
for tau, s in param_list:
    y = sigmoid(M, tau=tau, s=s)
    if tau == 0.0 and s == 1.0:
        plt.plot(M, y, 'k--', linewidth=2.0, label="标准 Sigmoid (τ=0, s=1)")
    else:
        plt.plot(M, y, label=f"τ={tau}, s={s}")

plt.xlabel("M (normalized color energy)")
plt.ylabel("σ((M - τ)/s)")
plt.title("Sigmoid((M - τ)/s) 随参数变化曲线")
plt.grid(True)
plt.legend(loc='center right', framealpha=0.9)  # 图例移到右侧
plt.ylim(-0.05, 1.05)
plt.savefig(save_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"已保存曲线至: {save_path}")
