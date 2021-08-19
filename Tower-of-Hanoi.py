def towerOfHanoi(n, fromRod, toRod, auxRod):
    if n == 1:
        print("Moved disk 1 from rod", fromRod, "to rod", toRod)
    towerOfHanoi(n-1, fromRod, auxRod, toRod)
    print("Moved disk", n, "from rod", fromRod, "to rod", toRod)
    towerOfHanoi(n-1, auxRod, toRod, fromRod)
    
towerOfHanoi(5, 'a', 'b', 'c')
