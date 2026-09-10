# R3 文献定位记录

核查日期：2026-09-10。以下记录用于后续论文定位，不代表已完成全面文献综述。

《Task-oriented Time Series Imputation Evaluation via Generalized Representers》（NeurIPS 2024）研究训练标签的填补如何影响下游模型学习后的表现，并使用广义表征近似减少重复训练。与本项目需要明确区分的对象是冻结预测器推理时的历史上下文填补。后续必须在正文讨论其任务导向评价与组合思想，避免泛称首次使用预测目标评价填补。

来源：https://proceedings.neurips.cc/paper_files/paper/2024/file/f88264fcc54775ee1706116e90fe351a-Paper-Conference.pdf 。已核查全文中的任务定义。

《TS-ICL: A Flexible Time-Indexed Foundation Model for Time Series via In-Context Learning》（2026，arXiv:2606.05878v2）统一预测与填补。第 5.2 节、表 4 已评估部分观测上下文：fev-bench 的 100 个单变量任务、长度 4092 的历史窗口，以及 30%—90% 缺失率，对比 Chronos-2。第 3 节允许不规则观测和任意查询时间，第 B.2.2 节区分预测与填补检查点。它说明论文动机需要涵盖已有原生缺失处理能力；“缺失历史下的预测”不能单独作为新问题主张。其长上下文单变量实验也不能直接代替本项目 96 步多变量输入、固定候选流程选择的验证。

来源：[TS-ICL 全文](https://arxiv.org/html/2606.05878v2)。已核查上述方法与实验段落，尚未复现。全文对部分预测模型预训练数据重叠的讨论提示后续确认协议也需要逐模型检查来源。

《Decision Theoretic Foundations for Conformal Prediction: Optimal Uncertainty Quantification for Risk-Averse Agents》（ICML 2025）把预测集合与风险敏感决策联系起来，提出 Risk-Averse Calibration。它是风险与效用叙述需要区分的已有工作。当前实现只在开发数据上选择切换阈值，不等同于该文的形式保证。

来源：https://proceedings.mlr.press/v267/kiyani25a.html 。当前核查元数据与摘要，形式假设和证明尚待全文核查。

《Mask-Conditional Conformal Prediction: Valid Uncertainty For All Missing Data Mechanisms》（AISTATS 2026）研究缺失协变量下按掩码条件化的覆盖问题。若后续加入形式校准保证，需要核查其缺失机制假设及与本研究行动选择风险之间的区别。

来源：https://proceedings.mlr.press/v300/fan26a.html 。当前核查论文索引和摘要，尚未引用其理论结论。
