"""Contact sheet: LR (upscaled to HR size with nearest) above HR, same frames."""
import numpy as np
from PIL import Image

OUT = "/home/ma-user/modelarts/user-job-dir/sol_h3/results/vae_decode_check"
S = "H3O20261001_0001"

hi = np.load(f"{OUT}/{S}_high.npy")
lo = np.load(f"{OUT}/{S}_low.npy")
print("hi", hi.shape, "lo", lo.shape)

picks = [0, 30, 60, 90, 123]
TH = 216
rows = []
for src, label in ((lo, "LR 448x256"), (hi, "HR 1344x768")):
    tiles = []
    for f in picks:
        im = Image.fromarray(src[f]).resize((int(TH * 1344 / 768), TH), Image.NEAREST)
        tiles.append(np.asarray(im))
    rows.append(np.concatenate(tiles, axis=1))
sheet = np.concatenate(rows, axis=0)
Image.fromarray(sheet).save(f"{OUT}/{S}_contact.png")
print("wrote", f"{OUT}/{S}_contact.png", sheet.shape)
