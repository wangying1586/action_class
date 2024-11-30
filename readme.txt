1.将原始数据集进行train,val,test集分割：
（修改 original_data_dir 为待分割的数据集，修改 output_data_dir 为分割完成数据集的存储位置）
运行命令行： python split_dataset.py  实现train,val,test集分割 

2.启动模型训练：
（修改loaded_train_dataset加载的root路径为步骤1分割好的train数据集路径，修改loaded_val_dataset 加载的root路径为步骤1分割好的val数据集路径）
运行命令行：python train_student.py  开启模型训练，并且在每个训练epoch后，开启模型验证，输出每个类别的精确度，召回率和F1分数，并输出整体的精确度，召回率和F1分数

3.启动模型推理：
（修改folder_path为待推理的图片数据）
运行命令行：python inference_student.py  开启模型推理