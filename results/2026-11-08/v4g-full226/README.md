# v4g 真实 226 帧 / 67-latent 全长评价与视频交付

日期：2026-11-08。阶段：在单卡昇腾 910B3 共享主机上，对当前 v4g 权重做真实生产长度的独立评测与视频交付。被测权重 `v4g_final.pt`，SHA256 `049b4a8f0ae594b397f0543775a6ccfaccb24ce3c260990fda9c8ab0d9bffe33`，与 11-08 baseline 相同。本轮是独立新建的评测包，冻结审阅包 17 个文件未改动。

本目录只公开脱敏后的回执、抽帧图和说明。私有主机路径、主机编号、设备编号、用户名与本地绝对路径均已移除；所有实测数字、张量形状和数据文件 SHA256 保持不变。原始输入视频、推理权重、完整 VAE / teacher 权重、解码后的 uint8 RGB 缓存和含输入路径的原始日志均未发布。见 [脱敏 manifest](manifest.public.json) 与 [逐文件校验和](SHA256SUMS)。

## 一、这次补齐了什么

此前公开的 v2 是 5 秒视频的 120 帧 / 37 latent 演示；历史 A–F 审计里的 67 latent 是 37 帧真实序列加反转 30 帧的**合成长输入**，不能写成真实 226 帧验收。本轮使用真实存在的 345 帧、24 fps、1344×768 视频 `VIDGEN_0429.mp4`（SHA256 `aa52435623bc49ed27a191bf1ac9c05fd0a999c3366271f4291297fd2693d11c`），严格读取**连续前 226 帧**：无 tpad、无 loop、无 setpts 减速、无反转。226 帧逐帧 SHA256 全部互不相同（`distinct_lr_frames = 226`），回文匹配 `0` 次；两个 reader 各收到精确 226 帧（LR 610959360 B，HR 1416314880 B），fps filter 无重复帧。源视频帧数 `345`、帧率 `24/1`、时长 `14.375000` 秒。

用发布版 FP32 tiled VAE 编码得到 draft `[24, 67, 44, 80]` 与 reference `[24, 67, 68, 120]`，即真实 226 像素帧 → 67 latent；`released_decoded_frames(67) = 226`，所以整段 VAE 解码输出正好 226 帧，**不裁剪**（`time_crop = none (all 226 frames kept)`）。这不是 37+反转30 的合成延长序列。

## 二、全长 latent 评价（阈值 0.01）

原始 teacher 权重先保存指纹、再加载 v4g、完成全部前向、最后恢复并核验指纹（`teacher_restored_verified = True`，SHA `68210b78441a86f445525e622b601e08c5c6545815fc5dfa2b6639ed9ca96ae7`）。六个门全部在真实 67-latent 序列上重跑：

| 门 | 本次实测 | 判定 |
| --- | --- | --- |
| future_cut_every_k | 66 个切点，all_exact_zero=True，worst=0.0 | 通过 |
| prefix_consistency | 11 个前缀，worst k=21，0.0010257959365844727 | 通过 |
| input_tail_swap | shared 20 帧，native=0.0，stream=0.0，consumed=[67, 67] | 通过 |
| streaming_real67 | max_abs=0.0006000250577926636，cached convs=62，cache bytes=1756385280，max 时间长度=4 | 通过 |
| stream_determinism | 完整 67 帧 rerun=0.0，37-of-67 前缀=0.0 | 通过 |
| state_machine | 三布尔 True, True, True | 通过 |

`all_passed = True`。主指标为全 67 latent 上 v4g 相对**原始双向 teacher** 的 macro latent MSE：`0.06744341552257538`；stream 版本 `0.06744343787431717`，noised 版本 `0.8419111371040344`。stream 对 full 的 max_abs `0.0006000250577926636`、mean_abs `2.7877129468834028e-05`；noised 的 past 前缀 max_abs `0.0`。full / teacher 对 HR 重建参考的 macro MSE 分别为 `0.08758459240198135` / `0.09774463623762131`。这些都是 latent 运算结论，teacher 不是 GT。数值详见 [eval_226.json](receipts/eval_226.json) 与 [manifest.public.json](manifest.public.json)。

## 三、整段 VAE 解码与 226 帧视频

五路信号（stream / full / teacher / noised / reference）各自解码为 226 帧 1088×1920 RGB，形状一致，解码器为 released FP32 tiled AutoencoderKLMiniMaxH3。组装 2×3 六面板视频，`2888×1160`、24 fps、`9.416666666666666` 秒；远端 ffmpeg 4.2 无 libx264，用 MPEG-4 q=2 编码。

远端验收：ffprobe `nb_frames = 226`，全帧 `-xerror` 解码 exit `0`，解码帧数 `226`；`passed = True`，原始视频 SHA256 `eeca414351c1986e80beea59a582f3be618250fb4b3616fa0be5facf96056ba5`，`44486319` 字节。

本地交付：用 ffmpeg 7.1 / libx264 crf18 生成有损 H.264 播放副本，两路文件都做完整解码逐帧计数，各 `226` 与 `226` 帧，exit 均为 0；H.264 副本 `17713744` 字节、SHA256 `c2df9466232cfc201a95340272df6a439f0971fc21bb1dd00cf63748c2a8a811`。抽帧图 [第 0 / 113 / 225 帧](full226_sampled_frames_000_113_225.png) 及 [frame000](frame000.png)、[frame113](frame113.png)、[frame225](frame225.png)。视频文件本体作为 [Release 资产](https://github.com/RedRoman/Tencent-streaming-sr/releases/tag/v4g-full226-20261108) 发布，校验和见 [RELEASE_SHA256SUMS](RELEASE_SHA256SUMS)。

像素诊断（**仅诊断，不是 GT 指标**）：单位 0-255 RGB MAE，参考是 HR latent 的 VAE 重建；stream-vs-full mean MAE `0.0036848712625260284`，stream / full / teacher 对参考分别 `5.350527158904099` / `5.350473621374365` / `4.514606004845476`，noised `38.13907113720361`。没有真实 LR/HR 配对 GT，**不给 PSNR / SSIM**。

## 四、耗时与资源边界

| 阶段 | 秒 |
| --- | ---: |
| strict_reader | 26.991 |
| vae_encode | 266.775 |
| latent_eval_67_real | 345.546 |
| whole_vae_decode_226 | 880.006 |
| pixel_diagnostics | 49.034 |
| video_226f | 85.853 |

各阶段合计 `1654.205` 秒，进程 wall `1660.479` 秒；`offline batch whole-VAE encode/decode; NOT real-time throughput`。启动前通过三把设备 flocks 与 `npu-smi` 空闲验证，worker 全程继承锁，runner 退出后锁自动释放。同机其他卡是他人进程，未触碰；没有 kill、没有绕过锁、没有覆盖旧产物。冻结审阅包复核仍为 16/16 OK。

## 五、边界与尚未完成

- 整段 VAE 编码 / 解码是**离线批处理**，不是在线流式 VAE 链路，不构成实时性能或端到端生成速度验收。
- LR 是 HR 的 area 下采样合成退化，HR 由 1344×768 lanczos 上采样到 1088×1920；没有真实 GT，像素 MAE 只是相对 VAE 重建参考的诊断值。
- 因果性结论只覆盖 upscaler 的 latent 运算。误差面板时间轴是 67→226 的近似显示重采样，未验证 VAE 逐点时间映射，因此不画像素切割线。
- **未做人工实际检视，不声称看过视频**；视频以 ffprobe + 全帧解码逐帧计数验收。
- v2 的 120 帧视频已发布在 `1f513c6` / tag `v4g-causal-viz-20261107-v2`，本轮**不重复生成或发布该内容**；本轮新增的是真实 226 帧结果。
- iteration-1 的 A/B 双臂（legacy3 对 channels，各 300 steps）仍未启动；td_m 分母未修，没有新 best 模型。
- 关于「单就 upscaler 本身，是否已是正式管线中已确定的可交付流式因果 upsampler」的判断，见 [ANALYSIS_UPSCALER_STATUS.md](ANALYSIS_UPSCALER_STATUS.md)。

## 六、目录

`manifest.public.json` 是脱敏总览；`receipts/` 是逐份脱敏原始回执（DONE、eval_226、video_audit、encode / decode、pixel_diagnostics、stages、source_provenance、identity、local_media_acceptance、acceptance_summary、source_proof 的 JSON）；`SHA256SUMS` 覆盖本目录全部文件，`RELEASE_SHA256SUMS` 覆盖 Release 视频资产。所有数字均来自原始回执，未手工录入。
