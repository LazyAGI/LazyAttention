# CTA内部遍历的2×2×2矩阵

固定已接受的文档组任务重排，关闭长任务切分；保持CTA队列、GQA4、Q64/KV128和其余算子优化不变。只改变一个CTA内部的四个query head与Q tile遍历。

- `head_major=False`：Q tile在外层，一个Q tile依次处理四个head（现有默认）。
- `head_major=True`：head在外层，一个head遍历完整Q列表后再处理下一个head。producer、MMA、dQ写回四条角色路径同时交换循环；K/V仍每任务加载一次，dK/dV仍完整累加后存储。
- `reverse=False`：原[Partial列表,Full列表]顺序。
- `reverse=True`：完整反转原遍历，先逆序Full列表，再逆序Partial列表。`full_first`编译常量交换列表访问顺序并保持Full标记正确；不把Full块当Partial计算。
- `interleave=8`：在每类列表的8-tile局部组内，将起点按KV tile编号模组长旋转。尾组按实际长度处理。同一CTA四个head共享排列，保留Q外层时的元数据复用。`interleave=0`关闭。

倒序和交错的数组排列在执行前一次性准备，保留稀疏计数、有效Q集合及padding。准备费用计入`预处理时间/42+完整GPU backward`，重用42次；没有每次执行重新排序。head/类访问优先是编译常量；交错不在内循环增加取模计算。

矩阵入口：`python -m benchmarks.traversal240 --capture-root ... --output-root ...`。按head外层开关、倒序开关、交错0/8的顺序构造8组，组0为本轮实验基准Q外层+正序+不交错。每例使用相同输入、相同任务重排表、两套共享输出缓冲区。8个Williams顺序平衡时间位置和相邻处理顺序，16个测量块，每组64个Graph内事件样本。240例在同一进程跑一轮，不做A/A或三次独立重复。运行开始预申请约70GiB并释放到缓存，同时保留约60GiB活跃预留区，其他内存循环复用；不调用empty_cache。

`prepare_q_traversal`和带额外编译参数的`_interface`保留为实验入口。完整矩阵之后，公共`prepare_attention`/`prepare_execution`默认采用Q外层、完整倒序、不交错；任务重排开、切分关。公开`backward`从prepared state读取编译常量并复用任务计划；实验runner显式固定其自身基准，不随公开默认值漂移。最终选择由完整矩阵的有效耗时决定。GPU时间、准备时间和所有原始配置结果均保留，无收益时先检查循环覆盖、管线状态、mask标记、K/V重复加载及预处理实现，再决定是否继续优化，不据此否定遍历策略。
