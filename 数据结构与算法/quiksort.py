# 快速排序
def quiksort(array):
    if len(array)<2:
        return array  #基线条件
    else:
        pivot = array[0]  #递归条件
        less = [i for i in array[1:] if i<pivot]  #由所有小于基准值的元素的元素组成的字数
        greater = [i for i in array[1:] if i>pivot] #由所有大于基准值的元素组成的子数组
        return quiksort(less)+pivot+quiksort(greater)
print(quiksort([10,5,2,3]))


#二分查找,返回的是索引值
def binary_search(myllist,item):
    low = 0
    high = len(mylist)-1
    s = 0
    if (low>high or low<0):
        return -1
    while low<=high:
        s+=1
        mid = int((low+high)/2)
        if mylist[mid] == item:
            return mid
        if mylist[mid]<=item:
            low = mid-1
        else:
            high = mid -1
    return -1

my_list = [1,3,5,7,9,11,13,15]
print(binary_search(my_list,-3))

# 选择排序
# 先写一个求最小值的函数
def find_min(list_name):
    list_min = list_name[0]
    for i,score in enumerate(list_name):
        if list_min>=score:
            list_min = score
            list_index = i
    return list_index

#再进行排序
def selectionSort(list_name):
    new_list = []
    for i in range(len(list_name)):
        list_min_index = find_min(list_name)
        new_list.append(list_name.pop(list_min_index))
    return new_list

zk = [6,2,3,4,5,6,1,2]
print(selectionSort(zk_list))



