# Put M idenctical apples on N identical plates, 
# allow some plates to be left empty, 
# how many different methods are there?

# 0<=m<=10,1<=n<=10. 0<=n<=10<=m<=10

# Cases:
# (1) when number of plates is more than the number of apples, 
#     there must be n-m leftover plates 
# (2) when number of plates is less than the number of apples,
#     a. when there is an empty plate at least one plate is empty 
#     b. when there is no empty plate, all plates have apples,
#        and removing one from each plate has no effect

def function(m: int, n: int):
    if m == 0 or n == 0: # all apples on a plate, or no apples
        return 1
    if m < n:
        return function(m, m) # number of plates remaining is equal to the number of apples equal to m
    if m >= n:
        return function(m, n-1) + function(m-n, n)

