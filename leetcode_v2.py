##################################################
# A collection of additional LeetCode problems 
# since leetcode.py is getting a bit too large...
##################################################


# https://leetcode.com/problems/count-pairs-whose-sum-is-less-than-target/
def countPairs(nums: List[int], target: int) -> int:
    size = len(nums)
    result = 0
    for idx_i, i in enumerate(nums):
        for idx_j in range(idx_i + 1, size):
            if i + nums[idx_j] < target:
                result += 1
    return result


# https://leetcode.com/problems/count-negative-numbers-in-a-sorted-matrix/ 
def countNegatives(self, grid: List[List[int]]) -> int:
    depth = len(grid)
    width = len(grid[0])
    r_ptr = 0
    c_ptr = width - 1
    result = 0

    while r_ptr < depth and c_ptr >= 0:
        if grid[r_ptr][c_ptr] < 0:
            result += depth - r_ptr
            c_ptr -= 1
        else:
            r_ptr += 1
          
    return result


# https://leetcode.com/problems/partition-list/ 
def partition(head: Optional[ListNode], x: int) -> Optional[ListNode]:
    if not head:
        return head

    sec_head, sec_tail = None, None
    fst_head, fst_tail = None, None
    
    curr = head
    while curr: 
        if curr.val < x:
            if fst_head:
                fst_tail.next = curr
                fst_tail = fst_tail.next
            else:
                fst_head = curr
                fst_tail = curr
        else:
            if sec_head:
                sec_tail.next = curr
                sec_tail = sec_tail.next
            else:
                sec_head = curr
                sec_tail = curr
        curr = curr.next

    if sec_tail:
        sec_tail.next = None
    if fst_tail:
        fst_tail.next = sec_head
    
    if fst_head:
        return fst_head
    return sec_head
        
