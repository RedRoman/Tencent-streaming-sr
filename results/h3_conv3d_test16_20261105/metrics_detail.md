# H3 Conv3d hybrid：固定step1800的test16扩展评估

本轮仅扩大像素评估覆盖，不训练、不换checkpoint、不重新选模。新增13条test，加上此前3条test组成全部16条；既有1条legacy单独报告。重跑anchor只验证历史一致性，不重复计入均值。

数据已被前序实验使用，不是盲测。固定H3 decoder仍非因果；这些结果不能证明端到端RGB流式或任意输入尺寸数值等价。

主参考为冻结decoder解码的HR reference；原source RGB为次要参考。每片120帧、24fps、5秒；全片、前102帧和后18帧分别报告。均值按clip等权，PSNR取逐clip PSNR均值。

## all16_test（16条）

### full120 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.4083 | 0.131143 | 0.0016295 |
| old_causal_ewarp | 24.3314 | 0.200394 | 0.0023937 |
| cmn | 20.7444 | 0.373120 | 0.0034168 |
| cmn_hybrid | 21.2410 | 0.307227 | 0.0030357 |
| conv3d_hybrid | 25.5670 | 0.131721 | 0.0019088 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 16 / 0 / 0；lpips_alex_v0_1: 16 / 0 / 0；temporal_difference_mse: 16 / 0 / 0

### full120 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 25.9717 | 0.140243 | 0.0018464 |
| old_causal_ewarp | 24.0302 | 0.207775 | 0.0025935 |
| cmn | 20.5871 | 0.378067 | 0.0036020 |
| cmn_hybrid | 21.1062 | 0.316272 | 0.0032193 |
| conv3d_hybrid | 25.2658 | 0.143361 | 0.0021142 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 16 / 0 / 0；lpips_alex_v0_1: 16 / 0 / 0；temporal_difference_mse: 16 / 0 / 0

### prefix102 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.4135 | 0.130671 | 0.0016382 |
| old_causal_ewarp | 24.3375 | 0.199247 | 0.0023934 |
| cmn | 20.7860 | 0.371311 | 0.0033941 |
| cmn_hybrid | 21.2798 | 0.305795 | 0.0030196 |
| conv3d_hybrid | 25.5734 | 0.131200 | 0.0019132 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 16 / 0 / 0；lpips_alex_v0_1: 16 / 0 / 0；temporal_difference_mse: 16 / 0 / 0

### prefix102 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 25.9720 | 0.139808 | 0.0018566 |
| old_causal_ewarp | 24.0321 | 0.206700 | 0.0025941 |
| cmn | 20.6251 | 0.376428 | 0.0035813 |
| cmn_hybrid | 21.1414 | 0.315086 | 0.0032047 |
| conv3d_hybrid | 25.2680 | 0.142835 | 0.0021187 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 16 / 0 / 0；lpips_alex_v0_1: 16 / 0 / 0；temporal_difference_mse: 16 / 0 / 0

### tail18 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.3984 | 0.133818 | 0.0015689 |
| old_causal_ewarp | 24.3179 | 0.206894 | 0.0023700 |
| cmn | 20.5440 | 0.383370 | 0.0034922 |
| cmn_hybrid | 21.0537 | 0.315343 | 0.0030932 |
| conv3d_hybrid | 25.5500 | 0.134674 | 0.0018694 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 16 / 0 / 0；lpips_alex_v0_1: 16 / 0 / 0；temporal_difference_mse: 16 / 0 / 0

### tail18 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 25.9874 | 0.142707 | 0.0017786 |
| old_causal_ewarp | 24.0404 | 0.213865 | 0.0025674 |
| cmn | 20.4052 | 0.387350 | 0.0036686 |
| cmn_hybrid | 20.9373 | 0.322992 | 0.0032693 |
| conv3d_hybrid | 25.2727 | 0.146343 | 0.0020767 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 16 / 0 / 0；lpips_alex_v0_1: 16 / 0 / 0；temporal_difference_mse: 16 / 0 / 0

## remaining13_test（13条）

### full120 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.8860 | 0.122280 | 0.0014430 |
| old_causal_ewarp | 24.7832 | 0.189389 | 0.0021949 |
| cmn | 21.1826 | 0.358996 | 0.0031350 |
| cmn_hybrid | 21.7035 | 0.294593 | 0.0027827 |
| conv3d_hybrid | 26.0387 | 0.123811 | 0.0017246 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 13 / 0 / 0；lpips_alex_v0_1: 13 / 0 / 0；temporal_difference_mse: 13 / 0 / 0

### full120 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.4337 | 0.131829 | 0.0016370 |
| old_causal_ewarp | 24.4721 | 0.197479 | 0.0023746 |
| cmn | 21.0176 | 0.364461 | 0.0033030 |
| cmn_hybrid | 21.5611 | 0.303736 | 0.0029497 |
| conv3d_hybrid | 25.7294 | 0.135234 | 0.0019091 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 13 / 0 / 0；lpips_alex_v0_1: 13 / 0 / 0；temporal_difference_mse: 13 / 0 / 0

### prefix102 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.8929 | 0.122002 | 0.0014663 |
| old_causal_ewarp | 24.7835 | 0.188564 | 0.0022219 |
| cmn | 21.2242 | 0.357306 | 0.0031630 |
| cmn_hybrid | 21.7421 | 0.293430 | 0.0028145 |
| conv3d_hybrid | 26.0445 | 0.123399 | 0.0017455 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 13 / 0 / 0；lpips_alex_v0_1: 13 / 0 / 0；temporal_difference_mse: 13 / 0 / 0

### prefix102 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.4350 | 0.131652 | 0.0016632 |
| old_causal_ewarp | 24.4683 | 0.196738 | 0.0024038 |
| cmn | 21.0555 | 0.363004 | 0.0033341 |
| cmn_hybrid | 21.5962 | 0.302912 | 0.0029841 |
| conv3d_hybrid | 25.7305 | 0.134901 | 0.0019312 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 13 / 0 / 0；lpips_alex_v0_1: 13 / 0 / 0；temporal_difference_mse: 13 / 0 / 0

### tail18 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.8693 | 0.123851 | 0.0013023 |
| old_causal_ewarp | 24.8025 | 0.194059 | 0.0020186 |
| cmn | 20.9886 | 0.368573 | 0.0029311 |
| cmn_hybrid | 21.5223 | 0.301181 | 0.0025746 |
| conv3d_hybrid | 26.0276 | 0.126143 | 0.0015950 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 13 / 0 / 0；lpips_alex_v0_1: 13 / 0 / 0；temporal_difference_mse: 13 / 0 / 0

### tail18 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.4454 | 0.132832 | 0.0014806 |
| old_causal_ewarp | 24.5151 | 0.201676 | 0.0021880 |
| cmn | 20.8416 | 0.372721 | 0.0030840 |
| cmn_hybrid | 21.3976 | 0.308405 | 0.0027283 |
| conv3d_hybrid | 25.7444 | 0.137118 | 0.0017739 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 13 / 0 / 0；lpips_alex_v0_1: 13 / 0 / 0；temporal_difference_mse: 13 / 0 / 0

## previous3_test（3条）

### full120 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 24.3379 | 0.169553 | 0.0024375 |
| old_causal_ewarp | 22.3734 | 0.248083 | 0.0032549 |
| cmn | 18.8454 | 0.434322 | 0.0046379 |
| cmn_hybrid | 19.2370 | 0.361975 | 0.0041321 |
| conv3d_hybrid | 23.5229 | 0.166000 | 0.0027069 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 3 / 0 / 0；lpips_alex_v0_1: 3 / 0 / 0；temporal_difference_mse: 3 / 0 / 0

### full120 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 23.9697 | 0.176704 | 0.0027538 |
| old_causal_ewarp | 22.1151 | 0.252393 | 0.0035421 |
| cmn | 18.7218 | 0.437023 | 0.0048978 |
| cmn_hybrid | 19.1350 | 0.370594 | 0.0043878 |
| conv3d_hybrid | 23.2568 | 0.178581 | 0.0030032 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 3 / 0 / 0；lpips_alex_v0_1: 3 / 0 / 0；temporal_difference_mse: 3 / 0 / 0

### prefix102 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 24.3359 | 0.168237 | 0.0023830 |
| old_causal_ewarp | 22.4048 | 0.245536 | 0.0031363 |
| cmn | 18.8873 | 0.431998 | 0.0043954 |
| cmn_hybrid | 19.2763 | 0.359375 | 0.0039080 |
| conv3d_hybrid | 23.5323 | 0.165005 | 0.0026402 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 3 / 0 / 0；lpips_alex_v0_1: 3 / 0 / 0；temporal_difference_mse: 3 / 0 / 0

### prefix102 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 23.9653 | 0.175152 | 0.0026945 |
| old_causal_ewarp | 22.1418 | 0.249871 | 0.0034186 |
| cmn | 18.7599 | 0.434602 | 0.0046525 |
| cmn_hybrid | 19.1704 | 0.367840 | 0.0041610 |
| conv3d_hybrid | 23.2637 | 0.177215 | 0.0029313 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 3 / 0 / 0；lpips_alex_v0_1: 3 / 0 / 0；temporal_difference_mse: 3 / 0 / 0

### tail18 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 24.3579 | 0.177009 | 0.0027239 |
| old_causal_ewarp | 22.2178 | 0.262513 | 0.0038924 |
| cmn | 18.6177 | 0.447492 | 0.0059236 |
| cmn_hybrid | 19.0231 | 0.376711 | 0.0053406 |
| conv3d_hybrid | 23.4806 | 0.171638 | 0.0030585 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 3 / 0 / 0；lpips_alex_v0_1: 3 / 0 / 0；temporal_difference_mse: 3 / 0 / 0

### tail18 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 24.0025 | 0.185496 | 0.0030696 |
| old_causal_ewarp | 21.9837 | 0.266680 | 0.0042111 |
| cmn | 18.5145 | 0.450742 | 0.0062018 |
| cmn_hybrid | 18.9423 | 0.386203 | 0.0056138 |
| conv3d_hybrid | 23.2288 | 0.186320 | 0.0033887 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 3 / 0 / 0；lpips_alex_v0_1: 3 / 0 / 0；temporal_difference_mse: 3 / 0 / 0

## previous1_legacy（1条）

### full120 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 30.1122 | 0.055607 | 0.0006380 |
| old_causal_ewarp | 27.5638 | 0.094863 | 0.0010836 |
| cmn | 22.9767 | 0.209113 | 0.0020568 |
| cmn_hybrid | 23.5062 | 0.174050 | 0.0018883 |
| conv3d_hybrid | 29.2425 | 0.064575 | 0.0008072 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### full120 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 29.4946 | 0.065084 | 0.0007332 |
| old_causal_ewarp | 27.1374 | 0.103784 | 0.0011670 |
| cmn | 22.8586 | 0.214944 | 0.0021228 |
| cmn_hybrid | 23.4388 | 0.180863 | 0.0019535 |
| conv3d_hybrid | 28.8778 | 0.074277 | 0.0009002 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### prefix102 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 29.8820 | 0.056574 | 0.0006655 |
| old_causal_ewarp | 27.3883 | 0.096247 | 0.0011222 |
| cmn | 22.8084 | 0.212754 | 0.0021468 |
| cmn_hybrid | 23.3296 | 0.175855 | 0.0019669 |
| conv3d_hybrid | 29.0506 | 0.064949 | 0.0008371 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### prefix102 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 29.2997 | 0.066331 | 0.0007638 |
| old_causal_ewarp | 26.9757 | 0.105443 | 0.0012089 |
| cmn | 22.6913 | 0.218902 | 0.0022151 |
| cmn_hybrid | 23.2615 | 0.183274 | 0.0020344 |
| conv3d_hybrid | 28.7061 | 0.075018 | 0.0009335 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### tail18 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 31.7133 | 0.050126 | 0.0004840 |
| old_causal_ewarp | 28.7200 | 0.087015 | 0.0008596 |
| cmn | 24.0773 | 0.188478 | 0.0015512 |
| cmn_hybrid | 24.6710 | 0.163823 | 0.0014475 |
| conv3d_hybrid | 30.5267 | 0.062454 | 0.0006383 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### tail18 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 30.8031 | 0.058020 | 0.0005621 |
| old_causal_ewarp | 28.1889 | 0.094388 | 0.0009257 |
| cmn | 23.9524 | 0.192519 | 0.0016036 |
| cmn_hybrid | 24.6082 | 0.167200 | 0.0014993 |
| conv3d_hybrid | 30.0042 | 0.070078 | 0.0007131 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

## test_category_animal（3条）

### full120 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.8043 | 0.134622 | 0.0023113 |
| old_causal_ewarp | 25.0316 | 0.186089 | 0.0028368 |
| cmn | 21.9555 | 0.312975 | 0.0034563 |
| cmn_hybrid | 22.4001 | 0.265396 | 0.0031284 |
| conv3d_hybrid | 26.0807 | 0.131555 | 0.0025021 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 3 / 0 / 0；lpips_alex_v0_1: 3 / 0 / 0；temporal_difference_mse: 3 / 0 / 0

### full120 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.2831 | 0.142395 | 0.0026201 |
| old_causal_ewarp | 24.6541 | 0.190917 | 0.0031237 |
| cmn | 21.7551 | 0.315914 | 0.0037160 |
| cmn_hybrid | 22.2490 | 0.274016 | 0.0033844 |
| conv3d_hybrid | 25.7391 | 0.144806 | 0.0027951 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 3 / 0 / 0；lpips_alex_v0_1: 3 / 0 / 0；temporal_difference_mse: 3 / 0 / 0

### prefix102 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.8616 | 0.133597 | 0.0022935 |
| old_causal_ewarp | 25.0566 | 0.184460 | 0.0028208 |
| cmn | 22.0056 | 0.311471 | 0.0034572 |
| cmn_hybrid | 22.4461 | 0.263796 | 0.0031383 |
| conv3d_hybrid | 26.1311 | 0.130833 | 0.0024822 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 3 / 0 / 0；lpips_alex_v0_1: 3 / 0 / 0；temporal_difference_mse: 3 / 0 / 0

### prefix102 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.3243 | 0.141492 | 0.0025989 |
| old_causal_ewarp | 24.6701 | 0.189420 | 0.0031038 |
| cmn | 21.7992 | 0.314467 | 0.0037140 |
| cmn_hybrid | 22.2887 | 0.272337 | 0.0033916 |
| conv3d_hybrid | 25.7758 | 0.143885 | 0.0027715 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 3 / 0 / 0；lpips_alex_v0_1: 3 / 0 / 0；temporal_difference_mse: 3 / 0 / 0

### tail18 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.4944 | 0.140426 | 0.0023784 |
| old_causal_ewarp | 24.9090 | 0.195321 | 0.0028594 |
| cmn | 21.6916 | 0.321496 | 0.0033664 |
| cmn_hybrid | 22.1612 | 0.274465 | 0.0030036 |
| conv3d_hybrid | 25.8105 | 0.135647 | 0.0025642 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 3 / 0 / 0；lpips_alex_v0_1: 3 / 0 / 0；temporal_difference_mse: 3 / 0 / 0

### tail18 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.0598 | 0.147517 | 0.0027070 |
| old_causal_ewarp | 24.5857 | 0.199404 | 0.0031703 |
| cmn | 21.5228 | 0.324118 | 0.0036439 |
| cmn_hybrid | 22.0437 | 0.283529 | 0.0032771 |
| conv3d_hybrid | 25.5453 | 0.150026 | 0.0028807 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 3 / 0 / 0；lpips_alex_v0_1: 3 / 0 / 0；temporal_difference_mse: 3 / 0 / 0

## test_category_daily（5条）

### full120 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 25.8736 | 0.112087 | 0.0012299 |
| old_causal_ewarp | 23.7005 | 0.177010 | 0.0020055 |
| cmn | 19.8255 | 0.369519 | 0.0027762 |
| cmn_hybrid | 20.3027 | 0.300914 | 0.0024597 |
| conv3d_hybrid | 24.8757 | 0.111650 | 0.0015307 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 5 / 0 / 0；lpips_alex_v0_1: 5 / 0 / 0；temporal_difference_mse: 5 / 0 / 0

### full120 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 25.5567 | 0.117907 | 0.0013544 |
| old_causal_ewarp | 23.4875 | 0.182314 | 0.0021189 |
| cmn | 19.7274 | 0.372554 | 0.0028785 |
| cmn_hybrid | 20.2293 | 0.307251 | 0.0025604 |
| conv3d_hybrid | 24.6748 | 0.120108 | 0.0016458 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 5 / 0 / 0；lpips_alex_v0_1: 5 / 0 / 0；temporal_difference_mse: 5 / 0 / 0

### prefix102 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 25.8323 | 0.112359 | 0.0012941 |
| old_causal_ewarp | 23.6493 | 0.177555 | 0.0021157 |
| cmn | 19.8702 | 0.368512 | 0.0029044 |
| cmn_hybrid | 20.3389 | 0.300221 | 0.0025879 |
| conv3d_hybrid | 24.8290 | 0.112174 | 0.0016076 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 5 / 0 / 0；lpips_alex_v0_1: 5 / 0 / 0；temporal_difference_mse: 5 / 0 / 0

### prefix102 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 25.5196 | 0.117995 | 0.0014197 |
| old_causal_ewarp | 23.4373 | 0.182706 | 0.0022301 |
| cmn | 19.7705 | 0.371583 | 0.0030078 |
| cmn_hybrid | 20.2648 | 0.306768 | 0.0026890 |
| conv3d_hybrid | 24.6300 | 0.120683 | 0.0017227 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 5 / 0 / 0；lpips_alex_v0_1: 5 / 0 / 0；temporal_difference_mse: 5 / 0 / 0

### tail18 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.1334 | 0.110551 | 0.0008611 |
| old_causal_ewarp | 24.0053 | 0.173924 | 0.0013579 |
| cmn | 19.6394 | 0.375225 | 0.0020028 |
| cmn_hybrid | 20.1558 | 0.304839 | 0.0017047 |
| conv3d_hybrid | 25.1610 | 0.108679 | 0.0010856 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 5 / 0 / 0；lpips_alex_v0_1: 5 / 0 / 0；temporal_difference_mse: 5 / 0 / 0

### tail18 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 25.7908 | 0.117411 | 0.0009812 |
| old_causal_ewarp | 23.7868 | 0.180089 | 0.0014690 |
| cmn | 19.5477 | 0.378056 | 0.0021023 |
| cmn_hybrid | 20.0846 | 0.309991 | 0.0018066 |
| conv3d_hybrid | 24.9490 | 0.116851 | 0.0012038 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 5 / 0 / 0；lpips_alex_v0_1: 5 / 0 / 0；temporal_difference_mse: 5 / 0 / 0

## test_category_fantasy（1条）

### full120 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.7709 | 0.114004 | 0.0023749 |
| old_causal_ewarp | 24.7323 | 0.177222 | 0.0033211 |
| cmn | 21.1202 | 0.339573 | 0.0047747 |
| cmn_hybrid | 21.8710 | 0.282720 | 0.0043008 |
| conv3d_hybrid | 26.0849 | 0.127004 | 0.0026603 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### full120 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.0882 | 0.126945 | 0.0026988 |
| old_causal_ewarp | 24.2110 | 0.187663 | 0.0036005 |
| cmn | 20.8233 | 0.347915 | 0.0050177 |
| cmn_hybrid | 21.5711 | 0.298239 | 0.0045467 |
| conv3d_hybrid | 25.5198 | 0.140843 | 0.0029715 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### prefix102 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.7048 | 0.115179 | 0.0024719 |
| old_causal_ewarp | 24.6786 | 0.178274 | 0.0034565 |
| cmn | 21.1267 | 0.338958 | 0.0049652 |
| cmn_hybrid | 21.8844 | 0.282207 | 0.0044692 |
| conv3d_hybrid | 26.0270 | 0.127698 | 0.0027560 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### prefix102 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.0151 | 0.128444 | 0.0028121 |
| old_causal_ewarp | 24.1528 | 0.189270 | 0.0037496 |
| cmn | 20.8226 | 0.347902 | 0.0052201 |
| cmn_hybrid | 21.5760 | 0.298635 | 0.0047273 |
| conv3d_hybrid | 25.4544 | 0.142055 | 0.0030825 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### tail18 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 27.1654 | 0.107348 | 0.0018054 |
| old_causal_ewarp | 25.0492 | 0.171256 | 0.0025184 |
| cmn | 21.0839 | 0.343056 | 0.0036032 |
| cmn_hybrid | 21.7958 | 0.285630 | 0.0032815 |
| conv3d_hybrid | 26.4285 | 0.123070 | 0.0020953 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### tail18 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.5271 | 0.118453 | 0.0020310 |
| old_causal_ewarp | 24.5567 | 0.178561 | 0.0027151 |
| cmn | 20.8268 | 0.347986 | 0.0037731 |
| cmn_hybrid | 21.5434 | 0.295991 | 0.0034510 |
| conv3d_hybrid | 25.9101 | 0.133976 | 0.0023148 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

## test_category_food（2条）

### full120 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 27.2094 | 0.080828 | 0.0012968 |
| old_causal_ewarp | 24.6396 | 0.136675 | 0.0022977 |
| cmn | 20.4402 | 0.325194 | 0.0036158 |
| cmn_hybrid | 20.9395 | 0.264656 | 0.0033126 |
| conv3d_hybrid | 26.2652 | 0.093261 | 0.0016840 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 2 / 0 / 0；lpips_alex_v0_1: 2 / 0 / 0；temporal_difference_mse: 2 / 0 / 0

### full120 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.8006 | 0.088037 | 0.0014338 |
| old_causal_ewarp | 24.4011 | 0.143048 | 0.0024140 |
| cmn | 20.3532 | 0.329751 | 0.0037122 |
| cmn_hybrid | 20.8694 | 0.270849 | 0.0034103 |
| conv3d_hybrid | 26.0119 | 0.101120 | 0.0018144 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 2 / 0 / 0；lpips_alex_v0_1: 2 / 0 / 0；temporal_difference_mse: 2 / 0 / 0

### prefix102 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 27.1772 | 0.080936 | 0.0012978 |
| old_causal_ewarp | 24.6356 | 0.135235 | 0.0022178 |
| cmn | 20.4006 | 0.326492 | 0.0035091 |
| cmn_hybrid | 20.9063 | 0.265601 | 0.0032074 |
| conv3d_hybrid | 26.2425 | 0.092409 | 0.0016532 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 2 / 0 / 0；lpips_alex_v0_1: 2 / 0 / 0；temporal_difference_mse: 2 / 0 / 0

### prefix102 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.7714 | 0.087884 | 0.0014415 |
| old_causal_ewarp | 24.3978 | 0.141460 | 0.0023378 |
| cmn | 20.3141 | 0.330985 | 0.0036133 |
| cmn_hybrid | 20.8365 | 0.271768 | 0.0033122 |
| conv3d_hybrid | 25.9944 | 0.100038 | 0.0017840 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 2 / 0 / 0；lpips_alex_v0_1: 2 / 0 / 0；temporal_difference_mse: 2 / 0 / 0

### tail18 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 27.4073 | 0.080215 | 0.0013031 |
| old_causal_ewarp | 24.7032 | 0.144836 | 0.0027681 |
| cmn | 20.6821 | 0.317840 | 0.0042276 |
| cmn_hybrid | 21.1431 | 0.259304 | 0.0039345 |
| conv3d_hybrid | 26.4261 | 0.098093 | 0.0018812 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 2 / 0 / 0；lpips_alex_v0_1: 2 / 0 / 0；temporal_difference_mse: 2 / 0 / 0

### tail18 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.9831 | 0.088904 | 0.0014014 |
| old_causal_ewarp | 24.4641 | 0.152048 | 0.0028649 |
| cmn | 20.5925 | 0.322756 | 0.0042794 |
| cmn_hybrid | 21.0717 | 0.265639 | 0.0039908 |
| conv3d_hybrid | 26.1476 | 0.107251 | 0.0020104 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 2 / 0 / 0；lpips_alex_v0_1: 2 / 0 / 0；temporal_difference_mse: 2 / 0 / 0

## test_category_nature（1条）

### full120 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.6299 | 0.177560 | 0.0017869 |
| old_causal_ewarp | 24.7959 | 0.243253 | 0.0023914 |
| cmn | 22.1683 | 0.393391 | 0.0029614 |
| cmn_hybrid | 22.9185 | 0.315318 | 0.0024518 |
| conv3d_hybrid | 25.9483 | 0.164380 | 0.0020355 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### full120 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 25.9975 | 0.175297 | 0.0022638 |
| old_causal_ewarp | 24.3720 | 0.235346 | 0.0028524 |
| cmn | 21.9218 | 0.386793 | 0.0034078 |
| cmn_hybrid | 22.6767 | 0.320557 | 0.0028921 |
| conv3d_hybrid | 25.4664 | 0.168458 | 0.0024944 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### prefix102 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.7278 | 0.174932 | 0.0017523 |
| old_causal_ewarp | 24.8621 | 0.240505 | 0.0023697 |
| cmn | 22.2768 | 0.387292 | 0.0028520 |
| cmn_hybrid | 23.0328 | 0.311949 | 0.0023744 |
| conv3d_hybrid | 26.0418 | 0.162773 | 0.0020019 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### prefix102 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.0846 | 0.172396 | 0.0022256 |
| old_causal_ewarp | 24.4338 | 0.232026 | 0.0028259 |
| cmn | 22.0262 | 0.380343 | 0.0032946 |
| cmn_hybrid | 22.7871 | 0.317088 | 0.0028102 |
| conv3d_hybrid | 25.5508 | 0.165940 | 0.0024563 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### tail18 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.1131 | 0.192457 | 0.0019745 |
| old_causal_ewarp | 24.4386 | 0.258825 | 0.0025025 |
| cmn | 21.5999 | 0.427951 | 0.0035603 |
| cmn_hybrid | 22.3221 | 0.334409 | 0.0028874 |
| conv3d_hybrid | 25.4532 | 0.173486 | 0.0022231 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### tail18 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 25.5344 | 0.191736 | 0.0024778 |
| old_causal_ewarp | 24.0374 | 0.254162 | 0.0029965 |
| cmn | 21.3732 | 0.423342 | 0.0040347 |
| cmn_hybrid | 22.0988 | 0.340215 | 0.0033599 |
| conv3d_hybrid | 25.0174 | 0.182722 | 0.0027133 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

## test_category_space（1条）

### full120 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.9782 | 0.139732 | 0.0008823 |
| old_causal_ewarp | 25.1983 | 0.224574 | 0.0011982 |
| cmn | 21.3687 | 0.419213 | 0.0015531 |
| cmn_hybrid | 21.7423 | 0.365602 | 0.0013101 |
| conv3d_hybrid | 26.3228 | 0.153428 | 0.0009902 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### full120 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.6241 | 0.155325 | 0.0010414 |
| old_causal_ewarp | 24.9278 | 0.239236 | 0.0013493 |
| cmn | 21.2057 | 0.428978 | 0.0016921 |
| cmn_hybrid | 21.6001 | 0.377380 | 0.0014477 |
| conv3d_hybrid | 26.0766 | 0.170890 | 0.0011434 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### prefix102 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 27.0873 | 0.138464 | 0.0008461 |
| old_causal_ewarp | 25.2816 | 0.222368 | 0.0011370 |
| cmn | 21.4780 | 0.416634 | 0.0014631 |
| cmn_hybrid | 21.8403 | 0.364092 | 0.0012306 |
| conv3d_hybrid | 26.4156 | 0.152276 | 0.0009419 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### prefix102 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.7155 | 0.154325 | 0.0010053 |
| old_causal_ewarp | 24.9991 | 0.237506 | 0.0012878 |
| cmn | 21.3085 | 0.426731 | 0.0016035 |
| cmn_hybrid | 21.6925 | 0.376281 | 0.0013697 |
| conv3d_hybrid | 26.1581 | 0.170007 | 0.0010947 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### tail18 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.4066 | 0.146921 | 0.0010689 |
| old_causal_ewarp | 24.7539 | 0.237073 | 0.0015191 |
| cmn | 20.7969 | 0.433824 | 0.0020286 |
| cmn_hybrid | 21.2249 | 0.374156 | 0.0017335 |
| conv3d_hybrid | 25.8316 | 0.159956 | 0.0012445 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### tail18 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.1393 | 0.160990 | 0.0012301 |
| old_causal_ewarp | 24.5444 | 0.249041 | 0.0016754 |
| cmn | 20.6649 | 0.441711 | 0.0021624 |
| cmn_hybrid | 21.1107 | 0.383609 | 0.0018652 |
| conv3d_hybrid | 25.6414 | 0.175898 | 0.0014027 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

## test_category_sport（1条）

### full120 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 25.8736 | 0.147130 | 0.0013174 |
| old_causal_ewarp | 23.5524 | 0.234715 | 0.0024640 |
| cmn | 19.8731 | 0.414023 | 0.0047811 |
| cmn_hybrid | 20.2385 | 0.345067 | 0.0043396 |
| conv3d_hybrid | 25.0528 | 0.152908 | 0.0016410 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### full120 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 25.5346 | 0.156923 | 0.0014555 |
| old_causal_ewarp | 23.3266 | 0.243773 | 0.0025784 |
| cmn | 19.7743 | 0.420773 | 0.0048807 |
| cmn_hybrid | 20.1573 | 0.353078 | 0.0044389 |
| conv3d_hybrid | 24.8160 | 0.163009 | 0.0017652 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### prefix102 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 25.8846 | 0.142259 | 0.0012215 |
| old_causal_ewarp | 23.6501 | 0.227142 | 0.0021916 |
| cmn | 19.9189 | 0.405975 | 0.0041051 |
| cmn_hybrid | 20.2770 | 0.337351 | 0.0037014 |
| conv3d_hybrid | 25.0964 | 0.148077 | 0.0015173 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### prefix102 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 25.5439 | 0.151642 | 0.0013517 |
| old_causal_ewarp | 23.4184 | 0.236085 | 0.0022991 |
| cmn | 19.8161 | 0.412651 | 0.0041988 |
| cmn_hybrid | 20.1913 | 0.345346 | 0.0037957 |
| conv3d_hybrid | 24.8597 | 0.157577 | 0.0016337 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### tail18 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 25.8117 | 0.174727 | 0.0018418 |
| old_causal_ewarp | 23.0367 | 0.277632 | 0.0039899 |
| cmn | 19.6224 | 0.459628 | 0.0084691 |
| cmn_hybrid | 20.0264 | 0.388790 | 0.0078495 |
| conv3d_hybrid | 24.8140 | 0.180284 | 0.0023240 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

### tail18 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 25.4823 | 0.186850 | 0.0020291 |
| old_causal_ewarp | 22.8401 | 0.287342 | 0.0041480 |
| cmn | 19.5452 | 0.466797 | 0.0086047 |
| cmn_hybrid | 19.9694 | 0.396890 | 0.0079816 |
| conv3d_hybrid | 24.5758 | 0.193788 | 0.0024956 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 1 / 0 / 0；lpips_alex_v0_1: 1 / 0 / 0；temporal_difference_mse: 1 / 0 / 0

## test_category_travel（2条）

### full120 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.0401 | 0.196955 | 0.0020166 |
| old_causal_ewarp | 24.0734 | 0.304935 | 0.0028953 |
| cmn | 20.7528 | 0.483404 | 0.0045582 |
| cmn_hybrid | 21.2465 | 0.388428 | 0.0039305 |
| conv3d_hybrid | 25.2555 | 0.185191 | 0.0023430 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 2 / 0 / 0；lpips_alex_v0_1: 2 / 0 / 0；temporal_difference_mse: 2 / 0 / 0

### full120 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 25.5343 | 0.218299 | 0.0022913 |
| old_causal_ewarp | 23.7217 | 0.323982 | 0.0031612 |
| cmn | 20.5301 | 0.497298 | 0.0048349 |
| cmn_hybrid | 21.0308 | 0.405548 | 0.0042039 |
| conv3d_hybrid | 24.8796 | 0.206691 | 0.0026050 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 2 / 0 / 0；lpips_alex_v0_1: 2 / 0 / 0；temporal_difference_mse: 2 / 0 / 0

### prefix102 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.0554 | 0.197725 | 0.0019865 |
| old_causal_ewarp | 24.1201 | 0.304017 | 0.0028312 |
| cmn | 20.8037 | 0.481078 | 0.0045041 |
| cmn_hybrid | 21.2984 | 0.386711 | 0.0038841 |
| conv3d_hybrid | 25.2857 | 0.185095 | 0.0023019 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 2 / 0 / 0；lpips_alex_v0_1: 2 / 0 / 0；temporal_difference_mse: 2 / 0 / 0

### prefix102 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 25.5391 | 0.219952 | 0.0022662 |
| old_causal_ewarp | 23.7581 | 0.323804 | 0.0031026 |
| cmn | 20.5748 | 0.495972 | 0.0047878 |
| cmn_hybrid | 21.0760 | 0.404820 | 0.0041645 |
| conv3d_hybrid | 24.8993 | 0.207318 | 0.0025683 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 2 / 0 / 0；lpips_alex_v0_1: 2 / 0 / 0；temporal_difference_mse: 2 / 0 / 0

### tail18 / vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 25.9563 | 0.192587 | 0.0021822 |
| old_causal_ewarp | 23.8241 | 0.310135 | 0.0032426 |
| cmn | 20.4827 | 0.496585 | 0.0048225 |
| cmn_hybrid | 20.9708 | 0.398153 | 0.0041678 |
| conv3d_hybrid | 25.0922 | 0.185730 | 0.0025705 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 2 / 0 / 0；lpips_alex_v0_1: 2 / 0 / 0；temporal_difference_mse: 2 / 0 / 0

### tail18 / vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 25.5076 | 0.208933 | 0.0024296 |
| old_causal_ewarp | 23.5245 | 0.324987 | 0.0034785 |
| cmn | 20.2908 | 0.504810 | 0.0050607 |
| cmn_hybrid | 20.7883 | 0.409678 | 0.0044025 |
| conv3d_hybrid | 24.7715 | 0.203136 | 0.0028094 |

相对旧causal eWarp逐clip提升 / 持平 / 退步：psnr_db: 2 / 0 / 0；lpips_alex_v0_1: 2 / 0 / 0；temporal_difference_mse: 2 / 0 / 0

## 逐clip结果

所有双参考、prefix/tail与各baseline逐片差值见[重新核算的完整报告](aggregate_recomputed.json)。以下为全120帧新Conv3d减旧causal eWarp，PSNR差值越大越好，其余差值越小越好。

| clip | split | ΔPSNR 主参考 | ΔLPIPS 主参考 | Δ时间MSE 主参考 | ΔPSNR sourceRGB | ΔLPIPS sourceRGB | Δ时间MSE sourceRGB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VIDGEN5S_0014509 | test | +1.1523363 | -0.0788734 | -0.0003559 | +1.0944959 | -0.0668889 | -0.0003579 |
| VIDGEN5S_0059416 | test | +1.2541535 | -0.0752573 | -0.0004358 | +1.2203395 | -0.0728070 | -0.0004248 |
| VIDGEN5S_0086441 | test | +1.2651904 | -0.0667040 | -0.0002320 | +1.2474566 | -0.0633968 | -0.0002302 |
| VIDGEN5S_0100402 | test | +1.0674008 | -0.0486425 | -0.0001340 | +1.0856632 | -0.0498690 | -0.0001315 |
| VIDGEN5S_0128747 | test | +0.6826695 | -0.0977363 | -0.0005894 | +0.6882165 | -0.0772740 | -0.0005734 |
| VIDGEN5S_0184359 | test | +0.9622393 | -0.0217275 | -0.0001233 | +0.9915814 | -0.0202006 | -0.0001235 |
| VIDGEN5S_0288595 | test | +1.7299093 | -0.0572725 | -0.0010168 | +1.7043738 | -0.0564419 | -0.0009913 |
| VIDGEN5S_0361101 | test | +1.5026458 | -0.0441382 | -0.0002916 | +1.5752814 | -0.0408580 | -0.0002889 |
| VIDGEN5S_0362565 | test | +1.5214642 | -0.0295547 | -0.0002104 | +1.5170970 | -0.0274148 | -0.0002078 |
| VIDGEN5S_0441001 | test | +1.3526151 | -0.0502175 | -0.0006609 | +1.3087296 | -0.0468206 | -0.0006290 |
| VIDGEN5S_0496359 | test | +1.6171538 | -0.0386385 | -0.0014373 | +1.6109891 | -0.0366397 | -0.0014377 |
| VIDGEN5S_0602499 | test | +1.1245488 | -0.0711460 | -0.0002080 | +1.1487400 | -0.0683460 | -0.0002059 |
| VIDGEN5S_0632078 | test | +1.1099888 | -0.1642313 | -0.0006688 | +1.0954504 | -0.1617749 | -0.0006876 |
| VIDGEN5S_0820879 | test | +0.9170699 | -0.0576160 | -0.0001426 | +0.9634722 | -0.0574186 | -0.0001392 |
| VIDGEN5S_0829322 | test | +1.5004571 | -0.0818071 | -0.0008229 | +1.4894032 | -0.0807647 | -0.0008132 |
| VIDGEN5S_0864318 | test | +1.0096453 | -0.1152005 | -0.0004282 | +1.0287214 | -0.1037058 | -0.0004268 |
| VIDGEN5S_0380588 | legacy | +1.6787192 | -0.0302878 | -0.0002763 | +1.7403764 | -0.0295074 | -0.0002668 |

以下为同一批逐clip新模型减teacher的差值；legacy仍单列标记，不并入test均值。

| clip | split | ΔPSNR 主参考 | ΔLPIPS 主参考 | Δ时间MSE 主参考 | ΔPSNR sourceRGB | ΔLPIPS sourceRGB | Δ时间MSE sourceRGB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VIDGEN5S_0014509 | test | -0.6816003 | -0.0131806 | +0.0002486 | -0.5310744 | -0.0068396 | +0.0002306 |
| VIDGEN5S_0059416 | test | -0.6864250 | +0.0089907 | +0.0002164 | -0.5321513 | +0.0086455 | +0.0002063 |
| VIDGEN5S_0086441 | test | -1.1495772 | +0.0055558 | +0.0002006 | -1.0251291 | +0.0074662 | +0.0001904 |
| VIDGEN5S_0100402 | test | -0.8538510 | +0.0007562 | +0.0000453 | -0.7616263 | +0.0004047 | +0.0000418 |
| VIDGEN5S_0128747 | test | -0.4747960 | -0.0219928 | +0.0002839 | -0.3951456 | -0.0079198 | +0.0002482 |
| VIDGEN5S_0184359 | test | -0.8193435 | +0.0125673 | +0.0001103 | -0.6156852 | +0.0134510 | +0.0001025 |
| VIDGEN5S_0288595 | test | -1.0606692 | +0.0159048 | +0.0006123 | -0.8853834 | +0.0157154 | +0.0006077 |
| VIDGEN5S_0361101 | test | -0.8764639 | +0.0002254 | +0.0001782 | -0.6209781 | +0.0017010 | +0.0001743 |
| VIDGEN5S_0362565 | test | -0.8275739 | +0.0089623 | +0.0001622 | -0.6920926 | +0.0104504 | +0.0001535 |
| VIDGEN5S_0441001 | test | -0.6860202 | +0.0129998 | +0.0002854 | -0.5684305 | +0.0138977 | +0.0002727 |
| VIDGEN5S_0496359 | test | -1.1362957 | +0.0147510 | +0.0009559 | -1.0174391 | +0.0150675 | +0.0009440 |
| VIDGEN5S_0602499 | test | -0.6553354 | +0.0136957 | +0.0001079 | -0.5474969 | +0.0155654 | +0.0001020 |
| VIDGEN5S_0632078 | test | -0.8827204 | -0.0325187 | +0.0004364 | -0.7773095 | -0.0318626 | +0.0004212 |
| VIDGEN5S_0820879 | test | -1.0914569 | +0.0123286 | +0.0001194 | -0.9607224 | +0.0126600 | +0.0001152 |
| VIDGEN5S_0829322 | test | -0.8207310 | +0.0057787 | +0.0003237 | -0.7186099 | +0.0060855 | +0.0003096 |
| VIDGEN5S_0864318 | test | -0.7582446 | -0.0355810 | +0.0001826 | -0.6445937 | -0.0245954 | +0.0001656 |
| VIDGEN5S_0380588 | legacy | -0.8696367 | +0.0089681 | +0.0001692 | -0.6168062 | +0.0091928 | +0.0001670 |

## 新增13例展示

六列顺序：teacher、旧causal eWarp、原CMN、CMN hybrid、新Conv3d hybrid、decoded HR reference。展示转码不参与指标；每段H264完整解码120帧、24fps、5秒。六列比较从4032×412缩放至3840×392，单列672×384不变；原始RGB指标不受展示缩放影响。

| clip | 六列比较 | 新模型 |
| --- | --- | --- |
| VIDGEN5S_0014509 | [comparison](VIDGEN5S_0014509/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0014509/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0059416 | [comparison](VIDGEN5S_0059416/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0059416/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0100402 | [comparison](VIDGEN5S_0100402/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0100402/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0184359 | [comparison](VIDGEN5S_0184359/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0184359/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0288595 | [comparison](VIDGEN5S_0288595/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0288595/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0361101 | [comparison](VIDGEN5S_0361101/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0361101/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0362565 | [comparison](VIDGEN5S_0362565/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0362565/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0441001 | [comparison](VIDGEN5S_0441001/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0441001/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0496359 | [comparison](VIDGEN5S_0496359/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0496359/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0602499 | [comparison](VIDGEN5S_0602499/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0602499/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0632078 | [comparison](VIDGEN5S_0632078/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0632078/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0820879 | [comparison](VIDGEN5S_0820879/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0820879/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0864318 | [comparison](VIDGEN5S_0864318/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0864318/conv3d_hybrid.h264.mp4) |

权重、teacher/latent/source缓存不发布；原始与脱敏JSON的SHA分别记录，原合同内部SHA保持原含义。旧发布目录及manifest保持原字节。
