def maxInversions(arr):
    inversions = 0
    size = len(arr)
    for i in range(1, size-1): # treat each element as the middle
        smaller, larger = 0, 0
        for j in range(i+1, size): 
            if arr[j] < arr[i]: 
                smaller += 1
        for j in range(i): 
            if arr[j] > arr[i]: 
                larger += 1
        inversions += smaller * larger
    return inversions   