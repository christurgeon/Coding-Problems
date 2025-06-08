
## Bits
#### Clears the lowest set bit of x
```
x & x - 1 

>>> bin(11 & (11 - 1))
'0b1010'
>>> bin(11)
'0b1011'
```
#### Extracts the lowest set bit of x
```
x & ~(x - 1)

>>> bin(12 & ~(12 - 1))
'0b100'
>>> bin(12)
'0b1100'
```

```
x - 1

Is X but the lowest set bit is cleared and all other bits to the right are 1

01010000 => 01001111

The ~ operator will invert...

x = 5  # 00000101 in binary
result = ~x  # 11111010 in binary (inverted bits)
print(result)  # Output: -6
```
#### Setting a Bit:
```
* To set (turn on) a specific bit, use the bitwise OR (|) operation with a mask where the target bit is 1 and all other bits are 0.
* Example: To set the 3rd bit (counting from 0) of x
x = x | (1 << 3)
```
#### Clearing a Bit:
```
* To clear (turn off) a specific bit, use the bitwise AND (&) operation with a mask where the target bit is 0 and all other bits are 1.
* Example: To clear the 3rd bit of x
x = x & ~(1 << 3)
```
#### Toggling a Bit:
```
* To toggle (flip) a specific bit, use the bitwise XOR (^) operation with a mask where the target bit is 1 and all other bits are 0.
* Example: To toggle the 3rd bit of x
x = x ^ (1 << 3)
```
#### Checking a Bit
```
* To check if a specific bit is set (i.e., if it is 1), use the bitwise AND (&) operation with a mask where only the target bit is 1.
* Example: To check if the 3rd bit of x is set
is_set = x & (1 << 3) != 0
```


#### Things to Know 
* Minimum Spanning Trees
* Leaky Bucket 
* Concurrency 
* Design Patterns 
* System Design
* Dijkstra Graph Search
* Dynamic Programming (e.g. napsack)
* Backtracking with Search 
* Heaps and Queues 
* LinkedLists
* HashMaps
* Prefix Tree / Trie 
* Graph (matrix, list)
