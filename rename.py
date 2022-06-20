import os
import xml.dom.minidom
import cv2

file_path = '3875/'
save_path = 'tmp/'

basename = 'AIROBOT0700100003935_'
xml_list = os.listdir(file_path)


num = 1
classdict={'fabric':0,'cable':0,'scale':0,'faeces':0,'pedestal':0}
haveobject = 0
for na in xml_list:
	if na.endswith('xml'):
		classlist=[]
		dom=xml.dom.minidom.parse(file_path+na)
		root = dom.documentElement
		class_name = root.getElementsByTagName('name')
		filename = root.getElementsByTagName('filename')
		path = root.getElementsByTagName('path')
		if len(class_name) >0:
			haveobject += 1
		for i in range(len(class_name)):
			if class_name[i].firstChild.data not in classlist:
				classlist.append(class_name[i].firstChild.data)
			if class_name[i].firstChild.data == 'fabric':
				classdict['fabric'] += 1
			elif class_name[i].firstChild.data == 'cable':
				classdict['cable'] += 1
			elif class_name[i].firstChild.data == 'scale':
				classdict['scale'] += 1
			elif class_name[i].firstChild.data == 'faeces':
				classdict['faeces'] += 1
			elif class_name[i].firstChild.data == 'pedestal':
				classdict['pedestal'] += 1
		classlist.sort()
		
		# tmp = '000000%d_'%num
		# tmp = tmp[-7:]
		# for i in classlist:
			# tmp += (i+'_')
		# img_name = basename + tmp+'.jpg'
		# xml_name = basename + tmp+'.xml'
		# path[0].firstChild.data = 'company/datasets/'+img_name
		# filename[0].firstChild.data = img_name
		# with open(save_path+xml_name,'w') as fh:
			# dom.writexml(fh)
		# img = cv2.imread(file_path+na[:-3]+'jpg')
		# cv2.imwrite(save_path+img_name,img)
		num += 1

#统计相关比例
totalxml = len(xml_list)/2
#有物体占所有图片的比例
print('have object proportion:%.2f%%'%((haveobject/totalxml)*100))
#每个类别占有物体的比例
total_classnumber = classdict['fabric']+classdict['cable']+classdict['scale']+classdict['faeces']+classdict['pedestal']
print('fabric:%.2f%%'%((classdict['fabric']/total_classnumber)*100))
print('cable:%.2f%%'%((classdict['cable']/total_classnumber)*100))
print('scale:%.2f%%'%((classdict['scale']/total_classnumber)*100))
print('faeces:%.2f%%'%((classdict['faeces']/total_classnumber)*100))
print('pedestal:%.2f%%'%((classdict['pedestal']/total_classnumber)*100))
