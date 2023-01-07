#!/usr/bin/env python3

from my_solution import *


# 测试用例
def test_solution():
    
    result=count_parameters(gnn)

    # 正确答案
    correct_solution = '582'
    
    # 程序求解结果
    assert correct_solution == result

