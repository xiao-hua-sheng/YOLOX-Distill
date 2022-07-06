一、教师模型训练
  教师模型的训练使用官方版本即可，一般选择m或者l的模型。
二、学生模型训练
1、train.py文件增加--teacher_exp_file和--T_model_file   
  teacher_exp_file用于加载教师模型相关参数，可以直接修改exps/example/*-teacher.py,主要注意num_classes、depth、width、input_size；
  T_model_file 教师模型路径，第一步中训练的教师模型的绝对路径（或者复制到本工程下的相对路径）。

2、教师模型的加载在trainer.py 131行左右实现

3、distill损失的实现见yolo_head.py 255行distill_loss，实现很简单，具体参数根据自己训练情况进行调整。

4、项目中增加了focal_loss和eiou，如果需要和原版进行对比请自行修改相关文件。
