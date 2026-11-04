import random
import cv2
import numpy as np
import torch

cv2.setNumThreads(1)

def read_clip(record, rng=None, crop=None, height=448, width=768):
    assert height % 16 == width % 16 == 0
    rng = rng or random
    if crop is None:
        crop = dict(start=rng.randint(0, record['num_frames']-22),
                    top=rng.randint(0, record['height']-height),
                    left=rng.randint(0, record['width']-width), height=height, width=width)
    cap = cv2.VideoCapture(record['video_path'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, crop['start'])
    frames = []
    for _ in range(22):
        ok, frame = cap.read()
        if not ok:
            cap.release(); raise RuntimeError(f"cannot decode consecutive 22F: {record['video_path']} {crop}")
        assert frame.shape == (record['height'], record['width'], 3)
        y,x,h,w = (crop[k] for k in ('top','left','height','width'))
        frames.append(cv2.cvtColor(frame[y:y+h,x:x+w], cv2.COLOR_BGR2RGB))
    cap.release()
    rgb = torch.from_numpy(np.stack(frames)).permute(3,0,1,2).unsqueeze(0).float()/255
    assert rgb.shape == (1,3,22,crop['height'],crop['width'])
    return rgb, crop
