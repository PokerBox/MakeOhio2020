import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pylab as plt
#import matplotlib.pyplot as plt
from scipy.stats import kde
import cv2

tree = ET.parse('sample.xml')
root = tree.getroot()
print(root[0].tag)

# chair_x=np.arange(426)
# chair_y=np.arange(426)
# chair=np.vstack((chair_x,chair_y))

# table_x=np.zeros(426)
# table_y=np.zeros(426)
# table=[table_x,table_y]

# person_x=np.zeros(426)
# person_y=np.zeros(426)
# person=[person_x,person_y]

# num_chair = 0
# num_person = 0
# num_table = 0
# num_data = 0

# for data in root:
# 	num_data += 1

# c_data_map = np.zeros((num_data, 1))
# t_data_map = np.zeros((num_data, 1))
# p_data_map = np.zeros((num_data, 1))

# count_count = 0

chair_x = []
chair_y = []
person_x = []
person_y = []
table_x = []
table_y = []

for data in root:
	# for item in data:
	# 	if(item.tag=='item'):
	# 		obj=item[0].text
	# 		if(obj=='chair'):
	# 			num_chair += 1
	# 		elif(obj=='person'):
	# 			num_person += 1
	# 		elif(obj=='table'):
	# 			num_table += 1
    
	# chair_map = np.zeros((num_chair, 2))
	# person_map = np.zeros((num_person, 2))
	# table_map = np.zeros((num_table, 2))

	# c_count = 0
	# p_count = 0
	# t_count = 0

	for item in data:
		if(item.tag=='item'):
			obj=item[0].text
			if(obj=='chair'):
				#chair_map[int(item[1].text)-1][int(item[2].text)-1] += 1
				chair_x.append(int(item[1].text)-1)
				chair_y.append(int(item[2].text)-1)
				# c_count += 1
			elif(obj=='person'):
				person_x.append(int(item[1].text)-1)
				person_y.append(int(item[2].text)-1)
				# p_count += 1
			elif(obj=='table'):
				table_x.append(int(item[1].text)-1)
				table_y.append(int(item[2].text)-1)
				# t_count += 1
	
	# c_data_map[count_count][0] = chair_map
	# p_data_map[count_count][0] = person_map
	# t_data_map[count_count][0] = table_map
	# count_count += 1

# print(chair_x,chair_y)

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(21, 5))
nbins = 10
chair = (chair_x, chair_y)

heat_map = np.zeros([260, 426], dtype = np.uint8)
i_max = len(chair_x) - 1
i = 0
while i < i_max:
	i = i + 1
	print(i)
	if chair_x[i]>=0 and chair_x[i]<=260 and chair_y[i]>=0 and chair_y[i]<=426:
		heat_map[chair_x[i],chair_y[i]] = heat_map[chair_x[i],chair_y[i]] + 100
		print("found chair")



heat_map = cv2.normalize(heat_map, None, 0, 255, cv2.NORM_MINMAX)
heat_map = cv2.GaussianBlur(heat_map,(13,13),cv2.BORDER_DEFAULT)
heat_map = cv2.normalize(heat_map, None, 0, 255, cv2.NORM_MINMAX)
heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
cv2.imshow("map",heat_map)
cv2.waitKey(0)

# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
k = kde.gaussian_kde(chair)
chair_x, chair_y = np.mgrid[chair_x.min():chair_x.max():nbins*1j, chair_y.min():chair_y.max():nbins*1j]
chair_z = k(np.vstack([chair_x.flatten(), chair_y.flatten()]))

# add shading
axes[0].set_title('2D Density with shading')
axes[0].pcolormesh(chair_x, chair_y, chair_z.reshape(chair_x.shape), shading='gouraud', cmap=plt.cm.BuPu)


k = kde.gaussian_kde(person)
person_x, person_y = np.mgrid[person_x.min():person_x.max():nbins*1j, person_y.min():person_y.max():nbins*1j]
person_z = k(np.vstack([person_x.flatten(), person_y.flatten()]))

# add shading
axes[1].set_title('2D Density with shading')
axes[1].pcolormesh(person_x, person_y, person_z.reshape(person_x.shape), shading='gouraud', cmap=plt.cm.BuPu)


k = kde.gaussian_kde(table)
table_x, table_y = np.mgrid[table_x.min():table_x.max():nbins*1j, table_y.min():table_y.max():nbins*1j]
table_z = k(np.vstack([table_x.flatten(), table_y.flatten()]))

# add shading
axes[2].set_title('2D Density with shading')
axes[2].pcolormesh(table_x, table_y, table_z.reshape(table_x.shape), shading='gouraud', cmap=plt.cm.BuPu)

