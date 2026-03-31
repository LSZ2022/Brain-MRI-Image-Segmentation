import json
import numpy as np
import matplotlib.pyplot as plt

# ======================
# 加载你3个实验的json
# ======================
def get_mean(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    loss = np.array(data["loss_per_epoch_per_patient"]).mean(1)
    dsc  = np.array(data["dsc_per_epoch_per_patient"]).mean(1)
    return loss, dsc

loss_mp, dsc_mp = get_mean("ckpt/260329-111426 lishangzhe_ckpt_maxpool/losses_dsc_values.json")
loss_st, dsc_st = get_mean("ckpt/260329-123713 lishangzhe_ckpt_strided/losses_dsc_values.json")
loss_pu, dsc_pu = get_mean("ckpt/260329-154559 lishangzhe_ckpt_pixelunshuffle/losses_dsc_values.json")

epochs = np.arange(1, 21)

# ======================
# 绘图
# ======================
plt.rcParams['figure.figsize'] = (13, 5)

plt.subplot(1, 2, 1)
plt.plot(epochs, loss_mp, label='MaxPool2d', linewidth=2)
plt.plot(epochs, loss_st, label='StridedConv2d', linewidth=2)
plt.plot(epochs, loss_pu, label='PixelUnshuffle', linewidth=2)
plt.title('Validation Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, dsc_mp, label='MaxPool2d', linewidth=2)
plt.plot(epochs, dsc_st, label='StridedConv2d', linewidth=2)
plt.plot(epochs, dsc_pu, label='PixelUnshuffle', linewidth=2)
plt.title('Validation DSC Comparison')
plt.xlabel('Epoch')
plt.ylabel('DSC')
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('downsample_comparison.png', dpi=300)
plt.show()