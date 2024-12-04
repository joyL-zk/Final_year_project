# My Final year Project in XDU
A Framework for optimizing Antenna using Deep learning and GA
### A simple script about calling HFSS 
hfss_script_1029.py----use PyAEDT library
first_script_1028.py----use HFSS 'Tools/Record script to file'

We used the recording script function in HFSS to build the HFSS library based on python code, which recorded most of the operations using HFSS, including model construction, parameter scanning, setting up simulation analysis, etc. Due to laboratory requirements, the library is not open source, but you can create it by yourself according to the above method or just use "PyAEDT library".

### AI Framework
Antenna Optimization
```
Antenna Optimization/
│
├── log/
│   ├── best_individuals_log.txt
│   └── performance_log.txt
│
├── trained_model/
│   ├── dnn_model_iteration_1.pth
│   └── dnn_model_iteration_2.pth
│
├── updated_dataset/
│   ├── updated_values_1.npy
│   └── updated_S_1.npy
│
│
└── initial_dataset/
    ├── initial_input_values.npy
    └── initial_S.npy

```
How to generate the initial dataset, see the python调用HFSS.md file for details
