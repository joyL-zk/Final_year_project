import win32com.client
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import subprocess
import pandas as pd
import datetime

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(10,128)
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,63)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train_DNN(model, num_epochs, input_values_path, S_path,save_path):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    """加载数据集"""
    input_values = np.load(input_values_path) #[500,10]
    S_values = np.load(S_path)  #[500,3,21]

    inputs = torch.tensor(input_values, dtype=torch.float32)
    targets = torch.tensor(S_values,dtype=torch.float32)
    targets = targets.reshape(targets.shape[0],-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = inputs.to(device)
    targets = targets.to(device)
    """train"""
    for epoch in range(num_epochs):
        model.train()
        predictions = model(inputs)
        loss = criterion(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:  # Print every 10 epochs
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Training Completed")
    torch.save(model.state_dict(),save_path)


class GA_algorithm(object):
    def __init__(self,model,
                population_size,
                chromosome_num, 
                generations,
                crossover_rate, 
                mutation_rate,
                param_ranges,
                iterations,
                initial_param,
                max_retries=10,
                ):
        
        self.model = model
        self.population_size = population_size
        self.chromosome_num = chromosome_num
        self.generations = generations
        self.crossover_rate = crossover_rate   #0.8
        self.mutation_rate = mutation_rate    #0.1
        self.max_retries = max_retries
        self.iterations = iterations
        self.initial_param = initial_param
        self.param_ranges = param_ranges
    
    def fitness_function(self, individual,iterations):
        model = self.model
        model.load_state_dict(torch.load(f"./trained_model/dnn_model_iteration_{iterations}.pth"))
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        individual_tensor = torch.from_numpy(individual).float().to(device)
        S_pred = model(individual_tensor).reshape(3,21)

        output_S_pred = S_pred[:,10]
    
        penalty = torch.tensor(0.0,device=device)
        if output_S_pred[0] >= -20:    # dm_matching < -20
            penalty += abs(output_S_pred[0] + 20)
        if output_S_pred[1] >= -20:    # cm_matching < -20
            penalty += abs(output_S_pred[1] + 20)
        if output_S_pred[2] >= -70:    # dm_isolation < -70
            penalty += abs(output_S_pred[2] + 70)
        return penalty.detach().cpu().numpy()
        

    def tournament_selection(self, population,fitness,tournament_size=100):
        selected = []
        for _ in range(2):
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmin(tournament_fitness)]
            selected.append(population[winner_index])
        return selected


    def initialize_population(self):
        population = np.zeros((self.population_size, len(self.initial_param)))
        population[0] = self.initial_param# 将initial_param指定为第一个个体

        for i in range(1,self.population_size):
            for j in range(len(self.initial_param)):
                low, high = self.param_ranges[j]
                population[i][j] = np.random.uniform(low,high)

            population[i] = np.round(population[i],1)
        return population
    
    def check_sample(self,individual):
        # 所有参数应都在范围值内
        within_bounds = True
        for i, (low, high) in enumerate(self.param_ranges):
            if individual[i] < low:
                within_bounds = False
            elif individual[i] > high:
                within_bounds = False

        couple_w = individual[7]  #第8个参数
        couple_gap = individual[6]  #第7个参数
        rx_itl = individual[8]  #第9个参数
        condition = (30 - couple_w * 2 - couple_gap) / 2 >= 2
        additional_condition = rx_itl / 2 - ((30 - couple_w * 2 - couple_gap) / 2) >= 1
        combined_condition = condition & additional_condition  # True/false
        return combined_condition and within_bounds

    def crossover(self, parent1, parent2):  #交叉操作——交换个体的部分基因
        crossover_point = np.random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def mutate(self, child):
        for i in range(len(child)):
            if np.random.rand() < self.mutation_rate:
                low, high = self.param_ranges[i]
                mutation_range = (high - low) * 0.1
                child[i] += np.random.uniform(-mutation_range, mutation_range)
                child[i] = np.clip(child[i], low, high)
        child = np.round(child, 1)
        return child

    # def crossover_mutate(self,parent1, parent2):
    #     retries = 0
    #     while retries < self.max_retries:
    #         # 执行交叉
    #         child1, child2 = self.crossover(parent1, parent2)
    #         # 执行变异
    #         child1 = self.mutate(child1)
    #         child2 = self.mutate(child2)
    #         # 强制修正在初始参数范围
    #         for i, (low, high) in enumerate(self.param_ranges):
    #             child1[i] = np.clip(child1[i], low, high)
    #             child2[i] = np.clip(child2[i], low, high)
    #         # 检查并修正子代是否符合所有条件
    #         if self.check_sample(child1) and self.check_sample(child2):
    #             return child1, child2
    #         retries += 1
    #     # 尝试多次后仍不满足条件，返回父代直接作为子代
    #     return parent1, parent2

    def optimize(self):
        population = self.initialize_population()
        best_individual =self.initial_param  # 最佳的结构参数
        best_fitness = float('inf') # 最佳的适应度函数

        for generation in range(self.generations):
            fitness = [self.fitness_function(x,self.iterations) for x in population]  # fitness类型为tensor,需转换为数值
       
            elite_individual = population[np.argmin(fitness)]  # 此时最小的适应度函数对应的个体
            elite_fitness = np.min(fitness)
            # 更新
            if elite_fitness < best_fitness:
                best_individual = elite_individual
                best_fitness = elite_fitness
            # 如果fitness==0，则退出
            if best_fitness == 0:
                print(f"Optimal solution found at generation {generation}. Terminating early.")
                break

            new_population = [elite_individual]
            while len(new_population) < self.population_size - 1:
                parent1, parent2 = self.tournament_selection(population, fitness)
                if np.random.rand() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    new_population.extend([child1, child2])
                else:
                    mutated_parent1 = self.mutate(parent1)
                    mutated_parent2 = self.mutate(parent2)
                    new_population.extend(mutated_parent1,mutated_parent2)
                    # # 确保变异后的个体符合条件
                    # if self.check_sample(mutated_parent1):
                    #     new_population.append(mutated_parent1)
                    # if self.check_sample(mutated_parent2):
                    #     new_population.append(mutated_parent2)
                    # offspring.extend([mutate(parent1, mutation_rate), mutate(parent2, mutation_rate)])
            population = np.array(new_population[:self.population_size])
            print(f"Generation {generation}/{self.generations} - Best Fitness: {best_fitness}")
        return best_individual,best_fitness, population

def opendesignname(Projectname,designname):
    oAnsoftApp = win32com.client.Dispatch('AnsoftHfss.HfssScriptInterface')
    oDesktop = oAnsoftApp.GetAppDesktop()
    oDesktop.RestoreWindow()
    oProject = oDesktop.SetActiveProject(Projectname)
    oDesign = oProject.SetActiveDesign(designname)
    oEditor = oDesign.SetActiveEditor("3D Modeler")
    oModule = oDesign.GetModule("Solutions")

def simulate(Projectname,designname):
    logging.info(f"Simulating Project: {Projectname}, Design: {designname}")
    command = f"python analyze.py {Projectname} {designname}"
    subprocess.run(command, shell=True)

def changevalue(Projectname,designname,value):
    # rx_cpw_w,rx_cpw_s,cpw_w,cpw_s,bar_w,m11_couple_cut,couple_gap,couple_w,rx_itl,couple_cut
    oAnsoftApp = win32com.client.Dispatch('AnsoftHfss.HfssScriptInterface')
    oDesktop = oAnsoftApp.GetAppDesktop()
    oDesktop.RestoreWindow()
    oProject = oDesktop.SetActiveProject(Projectname)
    oDesign = oProject.SetActiveDesign(designname)
    oDesign.ChangeProperty(
        ["NAME:AllTabs", ["NAME:LocalVariableTab", ["NAME:PropServers", "LocalVariables"],
                          ["NAME:ChangedProps",
                           ["NAME:rx_cpw_w", "Value:=", str(value[0]) + "um"],
                           ["NAME:rx_cpw_s", "Value:=", str(value[1]) + "um"],
                           ["NAME:cpw_w", "Value:=", str(value[2]) + "um"],
                           ["NAME:cpw_s", "Value:=", str(value[3]) + "um"],
                           ["NAME:bar_w", "Value:=", str(value[4]) + "um"],
                           ["NAME:m11_couple_cut", "Value:=", str(value[5]) + "um"],
                           ["NAME:couple_gap", "Value:=", str(value[6]) + "um"],
                           ["NAME:couple_w", "Value:=", str(value[7]) + "um"],
                           ["NAME:rx_itl", "Value:=", str(value[8]) + "um"],
                           ["NAME:couple_cut", "Value:=", str(value[9]) + "um"],
                           ]]])
    
def exportS11(Projectname,designname):
    oAnsoftApp = win32com.client.Dispatch('AnsoftHfss.HfssScriptInterface')
    oDesktop = oAnsoftApp.GetAppDesktop()
    oDesktop.RestoreWindow()
    oProject = oDesktop.SetActiveProject(Projectname)
    oDesign = oProject.SetActiveDesign(designname)
    oEditor = oDesign.SetActiveEditor("3D Modeler")
    oModule = oDesign.GetModule("Solutions")
    oModule = oDesign.GetModule("ReportSetup")
    csvpath = datasetpath +  "\\" + "S.csv"
    oModule.ExportToFile("Terminal S Parameter Plot 1", csvpath)

def readcsv(designname):
    csvpath = datasetpath + "\\" + "S.csv"
    df = pd.read_csv(csvpath)
    df = df.values
    dm_matching = df[:,1]
    cm_matching = df[:,2]
    dm_isolation = df[:,3]
    S_total = np.stack((dm_matching,cm_matching,dm_isolation))
    np.save(datasetpath + "\\" + "new_S.npy",S_total)

def stackS(iter,finalpath,newpath):
    newresult = np.load(newpath)
    newresult = np.expand_dims(newresult,axis=0)
    if iter == 1:
        np.save(finalpath,newresult)
    else:
        finalresult = np.load(finalpath)
        finalresult = np.concatenate((finalresult,newresult),axis=0)
        np.save(finalpath,finalresult)

def dataprocess(Projectname,designname,iter=1):
    exportS11(Projectname,designname)
    readcsv(designname)
    datasetpath_current = datasetpath
    stackS(iter,finalpath=datasetpath_current + "\\" + "total_S.npy",newpath=datasetpath_current + "\\" + "new_S.npy")

def HFSS_validation(Projectname, Designname, input_values):
    changevalue(Projectname,Designname,input_values)
    simulate(Projectname,Designname)
    dataprocess(Projectname, Designname)

def validate(model, input_values, validation_new_S, threshold):
    input_tensor = torch.from_numpy(input_values).float().to(next(model.parameters()).device)
    with torch.no_grad():
        predicted_S = model(input_tensor).cpu().numpy().reshape(3,21)  #predicted_S.shape(1,3,21)

    error_metrics = np.linalg.norm(predicted_S - validation_new_S) / input_tensor.shape[0]
    print(f"Validation error:{error_metrics}")
    retrain_flag = error_metrics > threshold
    return retrain_flag
    
def dataset_update(initial_values_path,initial_S_path,new_values_path,new_S_path,iteration):
    # 加载数据
    initial_values = np.load(initial_values_path)  #[500,10]
    initial_S = np.load(initial_S_path) #[500,3,21]
    new_values = np.load(new_values_path) #[10,]
    new_S = np.load(new_S_path)#[3,21]

    new_values = np.expand_dims(new_values, axis=0)  # Shape: [1, 10]
    new_S = np.expand_dims(new_S, axis=0)  # Shape: [1, 3, 21]

    updated_values = np.vstack((initial_values, new_values))
    updated_S = np.vstack((initial_S, new_S))
    # 保存更新后的文件路径
    updated_values_path = f'./updated_dataset/updated_values_{iteration}.npy'
    updated_S_path = f'./updated_dataset/updated_S_{iteration}.npy'

    np.save(updated_values_path, updated_values)
    np.save(updated_S_path, updated_S)
    print(f"Dataset updated successfully. New size;{updated_values.shape[0]}")
    return updated_values_path, updated_S_path


if __name__ =="__main__":
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    terminate = False  # 判断是否结束
    iteration = 1   
    max_iteration = 10
    num_epochs = 3000
    initial_param = [3, 10, 5, 10, 2, 10, 3, 10, 20, 10]
    param_ranges = [(2, 6), (2, 10), (3, 10), (2, 20), (2, 5), (5, 14), (2, 6), (5, 12), (10, 30), (5, 15)]  #结构参数范围
    datasetpath = "C:\\Users\\24762\\Desktop\\Final-year Project_lzk\\Antenna optimization\\framework\\initial_dataset" # Ant_with_CMR5.aedt的绝对路径
    projectname = 'Ant_with_CMR5'  #  Antenna project
    designname = '1'  #  Antenna design[1, 2, 3]
    initial_values_path = "./initial_dataset/initial_input_values.npy"  
    initial_S_path = "./initial_dataset/initial_S.npy"
    values_path = initial_values_path  # initial_input_values.npy---DNN input
    S_path = initial_S_path  # initial_S.npy----DNN output
    current_param = initial_param
    while not terminate and iteration <= max_iteration:
        print(current_param)
        """Stage1: 利用数据集对AI model 进行训练"""
        model = DNN()
        save_path = f"./trained_model/dnn_model_iteration_{iteration}.pth"  # dnn_model_iteration_1.pth
        print(f"Training the iteration_{iteration} DNN model")
        train_DNN(model,num_epochs, values_path, S_path, save_path)
        print(f"DNN model_{iteration} trained and saved to {save_path}")

        """Stage2: 使用GA进行优化"""
        print("Starting GA optimization")
        GA_opt = GA_algorithm(model = model,
                          population_size=500,  
                          chromosome_num=10,
                          generations=100,
                          crossover_rate=0.8,
                          mutation_rate=0.1,
                          iterations=iteration,
                          initial_param= current_param,
                          param_ranges=param_ranges,
                          )
        best_individual, best_fitness, final_population = GA_opt.optimize()
        log_path = "./log/best_individuals_log.txt"
        # 追加写入到日志文件
        with open(log_path, mode='a') as file:
            file.write(f"Time: {current_time}\n")
            file.write(f"Iteration {iteration}:\n")
            file.write(f"Best Individual: {best_individual}\n")

        """Stage3: 对优化得到的数据进行HFSS验证并判断是否需要重新训练模型"""
        output_file_path = f"./new_input_values.npy"
        np.save(output_file_path, best_individual)
        print(f"Optimization completed. Best individual saved to {output_file_path}")

        input_values = np.load(f'new_input_values.npy')
        HFSS_validation(projectname,designname,input_values) 
        validation_new_S_path = f"./initial_dataset/new_S.npy"
        validation_new_S = np.load(validation_new_S)
       
        # 更新数据集
        updated_values_path, updated_S_path = dataset_update(values_path,S_path,output_file_path,validation_new_S_path,iteration)
        
        # 查看一轮迭代下来的优化指标
        dm_matching_140GHz = validation_new_S[0][10]
        cm_matching_140GHz = validation_new_S[1][10]
        dm_isolation_140GHz = validation_new_S[2][10]
        print(f"new dm_matching @ 140GHz:{dm_matching_140GHz}")
        print(f"new cm_matching @ 140GHz:{cm_matching_140GHz}")
        print(f"new dm_isolation @ 140GHz:{dm_isolation_140GHz}")
        log_path = "./log/performance_log.txt"
        with open(log_path, mode='a') as file:
            file.write(f"Time: {current_time}\n")
            file.write(f"Iteration {iteration}:\n")
            file.write(f"dm_matching @ 140GHz: {dm_matching_140GHz}\n")
            file.write(f"cm_matching @ 140GHz: {cm_matching_140GHz}\n")
            file.write(f"dm_isolation @ 140GHz: {dm_isolation_140GHz}\n")

        if dm_matching_140GHz < -20 and cm_matching_140GHz < -20 and dm_isolation_140GHz < -70:
            terminate = True
        else:
            print("update dataset and retained model")

        values_path = updated_values_path
        S_path = updated_S_path
        current_param = best_individual
        iteration += 1
    
    print("Finish!")
    