import json
import numpy as np
import matplotlib.pyplot as plt

def get_mean(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    loss = np.array(data["loss_per_epoch_per_patient"]).mean(1)
    dsc  = np.array(data["dsc_per_epoch_per_patient"]).mean(1)
    return loss, dsc

path1 = "ckpt/260330-080229 lishangzhe_ckpt_Up_ConvTrans/losses_dsc_values.json"
path2 = "ckpt/260330-113104 lishangzhe_ckpt_Up_Bilinear/losses_dsc_values.json"
path3 = "ckpt/260330-101408 lishangzhe_ckpt_Up_PixelShuffle/losses_dsc_values.json"

loss_ct, dsc_ct = get_mean(path1)
loss_bi, dsc_bi = get_mean(path2)
loss_ps, dsc_ps = get_mean(path3)

epochs = np.arange(1, 21)

plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(epochs, loss_ct, label='ConvTranspose2d')
plt.plot(epochs, loss_bi, label='Bilinear')
plt.plot(epochs, loss_ps, label='PixelShuffle')
plt.title('Upsample Loss')
plt.grid(True)
plt.legend()

plt.subplot(122)
plt.plot(epochs, dsc_ct, label='ConvTranspose2d')
plt.plot(epochs, dsc_bi, label='Bilinear')
plt.plot(epochs, dsc_ps, label='PixelShuffle')
plt.title('Upsample DSC')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("upsample_comparison.png", dpi=300)
plt.show()
