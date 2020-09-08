import os
import pandas as pd
import matplotlib.pyplot as plt
import sys

os.chdir('../outputs')

def importance_plot(filename, mode='error'):
    fp = open(filename,"r")
    accs = []
    original_acc = 0
    num_node = 0
    for line in fp:
        if line[0:8] == "Original":
            original_acc = float(line[line.index("(") + 1:].split(",")[0])
        elif line[0:4] == "Node" or line[0:4] == "Edge":
            accs.append(float(line.strip().split(" ")[-1]))
            if line[0:4] == "Node":
                num_node += 1
    assert mode == 'error' or 'accuracy'
    if mode == 'accuracy':
        normalized = [(original_acc - x) / original_acc for x in accs]
    elif mode == 'error':
        normalized = [(x - original_acc)/ (1 - original_acc) for x in accs]

    edge_ind = []
    for ind in range(1,num_node+1):
        k = ind + 1
        while k < num_node+1:
            edge_ind.append((ind,k))
            k += 1
        
    df = pd.DataFrame(normalized, index=range(len(normalized)))
    new_index = []
    for i in range(1,num_node+1):
        new_index.append("Node" + str(i))
    for j in range(1,len(accs) - num_node + 1):
        new_index.append("Edge" + str(j))
    df.index = new_index
    df_sort = df.sort_values(by=0, ascending=True)

    nodes = [198,199,200,201,202,58,70,72,73,96,138,147,150,151,152,154,170,171,172,173,174,175,176,177,178,179,180,181,182,183,197,203,204,205]
    select_edges = df.iloc[num_node:,:].sort_values(by=0, ascending=False).iloc[0:10,:].index.values
    for e in select_edges:
        print("{} links the pair of nodes:".format(e))
        pair = edge_ind[int(e[4:])-1]
        print(nodes[pair[0]-1],nodes[pair[1]-1])
    
    plt.figure(figsize=(25,50))
    plt.barh(df_sort.iloc[-100:,:].index.values, df_sort.iloc[-100:,:][0]*100)
    for y, (x, c) in enumerate(zip(df_sort.iloc[-100:,:][0]*100+0.3, df_sort.iloc[-100:,:][0]*100)):
        plt.text(x, y, str(round(c,4))+"%", ha='center', va='center')
    plt.xlabel("relative increasing {} (%)".format(mode))
    plt.title("Importance of edges and nodes")
    plt.savefig("img/vi/" + '.'.join(filename.split("/")[-1].split(".")[0:3]) +".{}.eps".format(mode),format="eps")
    plt.show()

importance_plot(sys.argv[1],mode=sys.argv[2])
