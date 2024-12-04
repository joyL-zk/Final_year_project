# 一、安装版本
1. 仿真软件：HFSS 2022 R1
2. 系统：Windows
3. python 版本： python 3.8.20
4. python中需安装 pywin32 库 ——`pip install pywin32` ，能实现`import win32com`即成功。(**安装库时需关闭代理**)
5. 应安装aedt库——`pip install pyaedt`

# 二、HFSS手动建模仿真
仿真微带贴片天线
参考<https://blog.csdn.net/weixin_57548164/article/details/130286807?spm=1001.2014.3001.5506>
![微带贴片天线](./images/微带贴片天线.png)
## 1. HFSS软件设计流程
分为四步：建模、仿真、后处理及扫频与优化设计
### 1.1 建模
1. 设置变量——HFSS->Design Properties->add variables
2. 绘制基板、天线和传输线等
3. 设置边界条件——右键选中的部分->Assign Boundary->Perfect E
4. 绘制空气腔及其边界条件（设置为radiation）
5. 设置端口——选中端口->Assign Excitation->port->Wave port
### 1.2 分析仿真
1. 仿真设置——右键Analysis->add solution setup-> Advanced 
2. 扫频设置——右键setup->add frequency sweep
3. 检查——HFSS->validation check 
4. 仿真——HFSS->analyze all
5. 查看——右键results->create modal solution data report
6. 查看实时的电场强度等——选中天线表面->右键Field Overlays->Plot Fields->E->Mag_E(做成动图——Animate)
![初次仿真后得到的S参数](./images/微带天线仿真S参数.png)

### 1.3 优化
1. 优化设置添加变量——右键Optimetrics->add->parametric
2. 仿真分析——parametric setup->analyze
优化L0:
![优化L0得到的S参数](./images/优化L0后的天线S参数.png)
优化W1：
![优化W1得到的S参数](./images/优化W1后的天线S参数.png)
优化后的天线S参数：
![优化后的天线参数](./images/优化后的天线S参数.png)
## 2. HFSS微带贴片天线手动仿真出现的问题
1. 无法调出hfss模块——Tools->options->General options，新建project将默认以hfss创建
参考<https://www.cnblogs.com/ystwyfe/p/7616008.html>
2. [error] Parametric Analysis failed. (11:28:07 上午  10月 28, 2024)
    [error] Parametric Analysis failed. (11:28:07 上午  10月 28, 2024)
    [error] Machine Local Machine: Engine terminated unexpectedly, or machine reported error or was inaccessible. (11:28:07 上午  10月 28, 2024)——文件路径出现中文
# 三、 Python调用HFSS仿真微带贴片天线问题及解决方案
1. 记录HFSS的python脚本——和普通仿真一样，建立模型，设置各种条件，输出S11等results,用Tools->Record script to File 记录操作。
2. 记录操作并重用，该类方法容易导致代码量大且混乱，而且不适合二次编辑。
3. `pip install pyaedt`,选择使用pyaedt库 
4. 文件路径为**绝对路径**
![Pyaedt sheet](./images/pyaedt_API_cheat_sheet.png)

# 四、基于张润东师兄的代码的复现
1. 需`pip install pyDOE`,导入拉丁超立方采样
2. cpu运存爆了
![out of memory](./images/out%20of%20memory.png)
![out of memory hfss](./images/out%20of%20memory%20hfss.png)——更换更简易的模型(patch_by_python)
3. 拉丁超立方采样——分层、采样、乱序。![data sample](./images/data%20sample.png)
4. 成功跑通
5. 数据集的收集（目前的理解）——通过拉丁超立方采样设置不同变量的值的组合，对每一组进行扫描分析得到S参数，这样就可以得到样本和标签了，从而构成数据集。![variables S parameters](./images/different%20variables%20S%20parameters.png)
## 1.数据集制作总结
1. 确定每个要优化的设计变量的取值范围，从而形成高维欧氏解空间
2. 使用拉丁超立方采样在解决方案空间内生成多个天线设计解决方案组合。
3. 将生成的天线设计变量组合输入到HFSS中进行仿真，计算天线的各种性能指标，并保存指标数据。（所以样本要包含两类数据——不同的天线设计变量和天线性能指标值）
4. 样本顺序随机化，80%训练集，20%测试集

# 五、框架搭建
![framework](./images/Framework.png)
input: 10 parameters (500,10)
output: 3 index (500,3) per 1Ghz
每GHZ训练500次
## 思路
1. GT_Label:(500,3,21)--> (21,500,3)--> (500,3);
input: (500,10) --> (21,500,10)--> (500,10)
predict: (21,500,3)
2. input: (500,10)
output: (500,3,21)-->(500,63),这里也就希望替代模型能够预测63个输出
3. **问题**  使用pytorch框架，这里如何使用CNN？ ——使用几个Linear层，毕竟是一个框架，使用线性层训练速度更快
