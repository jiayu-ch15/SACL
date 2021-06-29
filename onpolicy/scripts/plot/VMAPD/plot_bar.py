import matplotlib.pyplot as plt  
plt.style.use('ggplot')
name_list = ['0','1','10','100'] 

action00 = [14, 20, 14, 2]  
action01 = [6, 2, 0, 0] 
action10 = [6, 17, 26, 38]
action11 = [14, 1, 0, 0] 

x = list(range(len(name_list))) 

total_width, n = 0.8, len(name_list)  
width = total_width / n  
  
plt.bar(x, action00, width=width, label='Actions {0,0}', edgecolor = 'white') 
for a,b in zip(x, action00):
    plt.text(a,b,'%d'%b, ha='center',va='bottom',fontsize=15)

for i in range(len(x)):  
    x[i] = x[i] + width 

plt.bar(x, action01, width=width, label='Actions {0,1}', edgecolor = 'white', tick_label = name_list)  
for a,b in zip(x, action01):
    plt.text(a,b,'%d'%b, ha='center',va='bottom',fontsize=15)

for i in range(len(x)):  
    x[i] = x[i] + width

plt.bar(x, action10, width=width, label='Actions {1,0}', edgecolor = 'white') 
for a,b in zip(x, action10):
    plt.text(a,b,'%d'%b, ha='center',va='bottom',fontsize=15)

for i in range(len(x)):  
    x[i] = x[i] + width 

plt.bar(x, action11, width=width, label='Actions {1,1}', edgecolor = 'white')
for a,b in zip(x, action11):
    plt.text(a,b,'%d'%b, ha='center',va='bottom',fontsize=15)


plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Alpha(z=40)', fontsize=20)
plt.ylabel('Frequency', fontsize=15)
# plt.title(title_name, fontsize=20)
plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=15)

plt.savefig("bar.png", bbox_inches="tight")