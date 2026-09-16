# R24 嵌入重建的计算精度核对

2026年9月15日，正式训练、预测冻结和读出完成后，原审查在TimesFM的一个嵌入坐标退出。失败记录、原审查脚本、执行日志及队列终态保留在repair-audit-v001/failure_snapshot，模型和预测未重新生成。

失败位于native_da6438eec9c69566_l192、pattern、第二次编码调用的坐标[1,4,911]。原嵌入值2.313371181488037，加入修复后的float32值0.013886213302612305；独立float64计算得到0.013887492585296535，差1.279282684e-6。较大修复量与原嵌入相消，使最终较小数值处的相对误差放大。这一数组61440个坐标中仅此处超过既定rtol1e-5/atol1e-6。

对全部保存轨迹的诊断比较了float64 NumPy、float32 NumPy、float32 CPU矩阵及float32 GPU矩阵四种独立计算。float64有1处超阈值，后面三种均无超阈值；GPU矩阵计算与原运行逐值相同。在失败坐标处，float32 NumPy、CPU矩阵与GPU矩阵的差分别为7.15e-7、9.54e-7和0。完整诊断保留于repair-audit-numerics-v001/diagnostic.json，没有读取或筛选准确度结果。

修正审查audit_patch_repair_v2.py按实际运行的float32精度独立计算LayerNorm、投影、GELU及残差相加；公式、样本、参数与全部数值阈值保持不变。原float64审查继续保留，不以宽松容差替代失败。新的完整核验输出到repair-audit-v002；仅在通过后解释实验效果。
