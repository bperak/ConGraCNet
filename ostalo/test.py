import numpy as np
from collections import Counter

arr = np.load('conj/fofs_vs_array_with_odv_true.npy', allow_pickle=True)
arr[3]

#%% acess list of fof nodes and attr
def change_odv_status_vertices(arr, v_list, status_initial, status_change):
    '''
    goes through the arr and changes the status of the vertex
    '''
    for item in range(0, len(arr)): # node
        # change source node values
        if arr[item][0] in v_list:
            if arr[item][2] == status_initial: 
               arr[item][2] = status_change    
   
        # change the odv_value of a nodes in vertices if it has status_initial 
        for x in arr[item][3:3+arr[item][1]]:
            if x[0] in v_list:
                if x[4] == status_initial:
                    x[4] = status_change
                #update the fof_p
                try:
                    c= Counter([x[4] for x in arr[item][3:3+arr[item][1]]])
                    arr[item][-1] = round((c[True]+c[False])/arr[item][1], 2)        
                except:
                    arr[item][-1] = 0
        
    print('Changed status for ', len(v_list), 'items in array')           
    return arr

#%%
order_list=[]
maximum = 1
while maximum != None:
    lst = [arr[n][-1] for n in range(0, len(arr)) if arr[n][2] == None]
    maximum= max(lst)
    # maximum2 = max(lst, key = lambda x: min(lst)-1 if (x == 1.0) else x)
    print(maximum)
    # PRONAĐI SVE ČVOROVE SA MAX
    nodes_for_fill = []
    for item in range(0, len(arr)): # node
        # change source node values
        if arr[item][-1] == maximum:
            nodes_for_fill.append(arr[item][0]) 
    print(nodes_for_fill)
    order_list.append(nodes_for_fill)
    # change nodes
    arr = change_odv_status_vertices(arr, nodes_for_fill, None, False)
    arr[nodes_for_fill[0]]

#%% spremiti listu
import pickle
with open("punjenje.txt", "wb") as fp:   #Pickling
   pickle.dump(order_list, fp)

#%% za load liste 
with open("punjenje.txt", "rb") as fp:   # Unpickling
   punjenje = pickle.load(fp)