#!/usr/bin/python3
import matplotlib.pyplot as plt
fo = open("score_out.txt", "r+")
print ("name: ", fo.name)
list = []
for line in fo.readlines():                        
    list.append(int(line))
fo.close()

plt.plot(list)
plt.ylabel('episode scores')
plt.xlabel('training episodes')
plt.title('scores')
plt.show()
