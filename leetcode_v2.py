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


# https://leetcode.com/problems/generate-binary-strings-without-adjacent-zeros/
def validStrings(n: int) -> List[str]:
    result = []

    def generate(current, last_one_idx, depth):
        nonlocal result
        # create the next two results
        v1 = current + "0"
        v2 = current + "1"
        # if we are done, aggregate the result
        if depth == n:
            if depth - last_one_idx <= 1:
                result.append(v1)
            result.append(v2)
            return
        # recurse
        if depth - last_one_idx <= 1:
            generate(v1, last_one_idx, depth+1)
        generate(v2, depth, depth+1)
    
    generate("", 0, 1)
    return result


# https://leetcode.com/problems/validate-stack-sequences/
def validateStackSequences(pushed: List[int], popped: List[int]) -> bool:
    stack = []
    popped_index = 0
    for i in pushed:
        stack.append(i)
        while len(stack) > 0 and stack[-1] == popped[popped_index]:
            stack.pop()
            popped_index += 1
    return len(stack) == 0


# https://leetcode.com/problems/find-occurrences-of-an-element-in-an-array/
def occurrencesOfElement(nums: List[int], queries: List[int], x: int) -> List[int]:
    lookup = {}

    result, occurrences = [], 1
    for idx, n in enumerate(nums):
        if n == x:
            lookup[occurrences] = idx
            occurrences += 1

    for q in queries:
        if q in lookup:
            result.append(lookup[q])
        else:
            result.append(-1)

    return result
