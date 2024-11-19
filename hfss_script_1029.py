from pyaedt import Hfss
import numpy as np
import pandas as pd
import time
import os
hfss = Hfss(version=221, project="C:/Users/24762/Desktop/Graduation Project_lzk/patch_by_python.aedt", solution_type="Modal") # 仿真的文件路径应为绝对路径

"""检查模型的变量参数是否会存入hfss中"""
# 检查模型的变量参数——导入hfss模型，其中设置的变量会自然存入hfss当中
#print(hfss["L0"])——28mm
"""模型仿真分析"""
setup0 = hfss.create_setup(name='my_setup_1',setup_type="HFSSDriven",Frequency="2.45GHz")
sweep0 = hfss.create_linear_count_sweep(setup="my_setup_1", units="GHz", 
                                        start_frequency=2,
                                        stop_frequency=3,
                                        name="LinearCountSweep",
                                        sweep_type="Interpolating",
                                        num_of_freq_points=101)
hfss.analyze_setup(name="my_setup_1",cores=4)
time.sleep(2)
hfss.post.create_report("dB(S(1,1))",
                        setup_sweep_name="my_setup_1: LinearCountSweep",
                        domain="Sweep",
                        plot_type="Rectangular Plot")

datasetpath = "C:\\Users\\24762\\Desktop\\Graduation Project_lzk"
designname = "HFSSDesign1"
csv_path = os.path.join(datasetpath, designname,"S.csv")
# hfss.post.export_report_to_file(output_dir=datasetpath, 
#                                 plot_name="S Parameter Plot 1",
#                                 extension=".csv")
hfss.post.export_report_to_csv(project_dir=datasetpath,
                               plot_name="S Parameter Plot 1")


# s11_data = hfss.post.get_solution_data("dB(S(1,1))", 
#                                        setup_sweep_name="my_setup_1 : LinearCountSweep",
#                                        domain="Sweep"
#                                        )





# s11_data = hfss.export_parametric_results(sweep="LinearCountSweep",output_file='C:/Users/24762/Desktop/Graduation Project_lzk/S_data.csv')
"""遍历区间的参数然后优化"""
# # 设置设计参数范围
# L0_range = np.linspace(27, 28, 5)  # 取5个不同的长度值
# # 遍历长度参数
# for L0 in L0_range:
#     # 设置参数值
#     hfss["L0"] = f"{L0}mm"
#     setup1 = hfss.create_setup(setupname="my_hfss_run")  
#     linear_count_sweep = hfss.create_linear_count_sweep(setupname="my_hfss_run",
#                                                     sweepname="LinearCountSweep",
#                                                     unit="GHz", freqstart=1.5,
#                                                     freqstop=3.5, num_of_freq_points=201)

#     # 运行仿真
#     hfss.analyze_setup("my_hfss_run", num_cores=4)
#     time.sleep(2)  # 添加等待时间，确保仿真完成

#     # # 获取仿真结果——提取S参数
#     # s11_data = hfss.export_parametric_results(sweep="LinearCountSweep")  #

#     # # 将结果保存到数据列表
#     # data.append([L0, s11_data])
"""利用optimetrics函数进行参数扫描"""
# setups_2 = hfss.parametrics.add("L0", start_point=27, 
#                                 end_point=29, 
#                                 step=0.2, 
#                                 variation_type="Linearstep", 
#                                 name="L0_parametrics") #执行该代码导致HFSS崩溃
# hfss.analyze(setup="L0_parametrics",num_cores=4)
# time.sleep(2)


hfss.save_project(project_file="C:/Users/24762/Desktop/Graduation Project_lzk/patch_by_python.aedt", overwrite=True )
# 关闭HFSS会话
hfss.close_project()
# hfss.close_desktop()
