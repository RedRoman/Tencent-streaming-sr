# 已发布与远端验收

核验时间：2026-11-08T14:01:56.375961+00:00。本轮代码、配置、完整226帧指标与负结果已推送，Release已公开。之前的draft状态文件是保留的发布前快照，以本页和[实际发布回执](publication_verified_v1.json)的时间为准。

- [实验源提交 55965e2](https://github.com/RedRoman/Tencent-streaming-sr/commit/55965e210a816a1615b7cee5cc1078c051e39e4e)；Release tag指向该提交。
- [21条完整视频目录](VIDEO_LINKS.md)：每条226帧/24fps，含三列主视频、同位置局部和原生1080单路。
- [公开Release与完整推理权重](https://github.com/RedRoman/Tencent-streaming-sr/releases/tag/h3-limiteddev-226f-20261108)：共27资产，其文件大小、SHA-256、资产ID及最终下载URL均已核验；匿名API可访问，分支与tag已核对。

权重为1,381,220,283字节，SHA-256 `7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5`。完整源文件、上传流与GitHub服务器摘要一致。原上传脚本在全量上传完成后误拒绝草稿untagged地址；只读v2确认既有资产，未重传，原失败证据保留。本机独立SFTP副本仍在续传，其完成状态不由Release成功推断。

完整native-HR L1守门仍为**FAILED**（+2.7611%，超过2%）；LPIPS改善不等于整体画质通过。连续主观观看未完成，已查看静态联系页和技术解码不能替代此检查。数据仅2train/3dev，完整16dev未通过，test未解锁。新LPIPS0.1消融训练另行监督，不能用训练loss宣称改善。
