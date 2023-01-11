# 算法示例


采用自动生成测试用例的方式，通过data_generator.py文件生成训练以及测试所需数据，用SBM、SBM_multiclass函数生成关系图。
通过对参数num_examples_train的设置来控制生成训练用例的个数，利用生成的用例训练出模型，将模型存放于gnn_J2_lyr1_Ntr10_num10文件中。
通过对参数num_examples_test的设置来控制生成测试用例的个数，使用训练出的模型训练用例，得出损失和输出。


## 使用指南


1. 按 `CTRL + P` 打开命令行面板，输入 "terminal: Create New Terminal" 打开一个命令行终端.
2. 在命令行里输入 `cd 1_算法示例` 并按 `ENTER` 进入"算法示例"目录。
3. 在命令行里输入 `python solution.py` 按 `ENTER` 运行示例程序。
