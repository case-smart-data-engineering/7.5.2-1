# 算法示例


本示例采用的数据集是算法自动生成的数据集，通过data_generator.py文件生成含有十个结点的图。
参数num_examples_train控制生成训练用例的个数，每个用例就是一张图，利用生成的用例训练出模型，将模型存放于gnn_J2_lyr1_Ntr10_num10文件中。
参数num_examples_test的设置来控制生成测试用例的个数，使用训练出的模型训练用例。
输出的结果是每个用例的社区分配结果和平均损失。

## 使用指南


1. 按 `CTRL + P` 打开命令行面板，输入 "terminal: Create New Terminal" 打开一个命令行终端.
2. 在命令行里输入 `cd 1_算法示例` 并按 `ENTER` 进入"算法示例"目录。
3. 在命令行里输入 `python solution.py` 按 `ENTER` 运行示例程序。
