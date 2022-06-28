1、train.py文件增加--teacher_exp_file和--T_model_file   用于加载教师模型使用，注意修改为自己路径

2、教师模型的加载在trainer.py 131行左右实现

3、distill损失的实现见yolo_head.py 255行distill_loss，实现很简单，具体参数根据自己训练情况进行调整。

4、项目中增加了focal_loss和eiou，如果需要和原版进行对比请自行修改相关文件。
