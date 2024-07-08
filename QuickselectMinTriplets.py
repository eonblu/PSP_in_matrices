def partition(array, CompsObj, left, right):     
    x = array[right] 
    # Could be randomized
    i = left 
    for j in range(left, right):
        CompsObj.increment()         
        if array[j][0] <= x[0]: 
            array[i], array[j] = array[j], array[i]
            i += 1            
    array[i], array[right] = array[right], array[i]
    return i 

def select_kth_triplet_main(array, k, CompsObj, left, right): 
    if (k > 0 and k <= right - left + 1): 
        i = partition(array, CompsObj, left, right) 
        if (i - left == k - 1): 
            return array[i] 
        if (i - left > k - 1): 
            return select_kth_triplet_main(array, k, CompsObj, left, i - 1) 
        return select_kth_triplet_main(array, k - i + left - 1, CompsObj, i + 1, right)
    print("Out of bounds")

def select_kth_triplet(array, k, CompsObj):
    return select_kth_triplet_main(array, k, CompsObj, 0, len(array)-1)

if __name__ == '__main__':
    select_kth_triplet([[13, 0, 0], [14, 1, 2], [6, 2, 0], [13, 3, 2], [8, 4, 1], [15, 5, 2]], 0, None)