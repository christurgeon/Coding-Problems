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


# https://leetcode.com/problems/minimum-number-of-moves-to-seat-everyone/ 
def minMovesToSeat(seats: List[int], students: List[int]) -> int:
    seats = sorted(seats)
    students = sorted(students)
    total_moves = 0
    for i, j in zip(seats, students):
        total_moves += abs(i - j)
    return total_moves


# https://leetcode.com/problems/palindrome-linked-list/
def isPalindrome(head: Optional[ListNode]) -> bool:
    if head is None or head.next is None:
        return True

    # use the 2-iterator approach
    slow, fast = head, head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next

    # reverse the second half
    prev = None
    while slow:
        temp = slow.next
        slow.next = prev
        prev = slow
        slow = temp

    # compare the contents of 1st and 2nd hald
    fst, scd = head, prev
    while scd:
        if fst.val != scd.val:
            return False
        fst, scd = fst.next, scd.next
    return True


# https://leetcode.com/problems/linked-list-random-node/
class GetRandomNodeValue:
    
    def __init__(self, head: Optional[ListNode]):
        self.values = []
        while head:
            self.values.append(head.val)
            head = head.next

    def getRandom(self) -> int:
        return random.choice(self.values)


# https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/
def buildTree(inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
    """
    The rightmost value in the postorder list is the root.

    If we look up the index in the inorder list, we know everything to the left
    is on the left subtree path and everything on the right is on the right
    subtree path.

    Input: inorder = [9,3,15,20,7], postorder = [9,15,7,20,3]
    Output: [3,9,20,null,null,15,7]
    """
    if not inorder or not postorder:
        return None
    root = TreeNode(postorder.pop())
    index = inorder.index(root.val)
    root.right = self.buildTree(inorder[index+1:], postorder)
    root.left = self.buildTree(inorder[:index], postorder)
    return root
        

# https://leetcode.com/problems/maximum-strong-pair-xor-i/
def maximumStrongPairXor(nums: List[int]) -> int:
    # Sort to reduce the search space.
    nums.sort()
    ans = 0
    for i in range(0, len(nums)):
        # Search for j
        # So in a brute force, we could search from i to the end 
        # But since we are sorted we only need to search such that the difference is not negative.
        # Therefore, we can get the rightmost value by: 
        # => nums[j] - nums[i] <= nums[i]
        # => nums[j] <= nums[i] + nums[i]
        # so j must be at the index of twice nums[i]
        rightmost_value = bisect_right(nums, nums[i]+nums[i])
        for j in range(i, rightmost_value):
            ans = max(ans, nums[i]^nums[j])
    return ans


# https://leetcode.com/problems/count-prefix-and-suffix-pairs-i/
def countPrefixSuffixPairs_v1(words: List[str]) -> int:
    
    def isPrefixAndSuffix(str1, str2):
        s1 = len(str1)
        s2 = len(str2)
        if s1 > s2:
            return False
        return str1 == str2[:s1] and str1 == str2[-s1:len(str2)]

    ans = 0
    for i, w1 in enumerate(words):
        for j in range(i+1, len(words)):
            if isPrefixAndSuffix(w1, words[j]):
                ans += 1
    return ans

    
# https://leetcode.com/problems/count-prefix-and-suffix-pairs-i/
def countPrefixSuffixPairs_v2(self, words: List[str]) -> int:
    ans = 0
    n = len(words)
    for i in range(n):
        for j in range(i+1, n):
            if words[j].startswith(words[i]) and words[j].endswith(words[i]):
                ans += 1
    return ans
    

# https://leetcode.com/problems/count-prefix-and-suffix-pairs-i/
class TrieNode:
    def __init__(self):
        self.children = {}
        self.end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        # Traverse down the trie, adding a value as we go
        curr = self.root
        for c in word:
            curr.children[c] = TrieNode()
            curr = curr.children[c]
        curr.end = True

    def search(self, word):
        # Search down the trie
        curr = self.root
        for c in word:
            if c not in curr.children:
                return False
            curr = curr.children[c]
        return curr.end

    def clear(self):
        self.root = TrieNode()

class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        ans = 0
        prefix_trie, suffix_trie = Trie(), Trie()        
        for i in reversed(range(1, len(words))):
            # Create a trie to hold the suffix and the prefix
            # Then we can do lookups each time on it, as we move to words to the left 
            prefix_trie.insert(words[i])
            suffix_trie.insert(words[i][::-1])
            for j in range(0, i):
                if len(words[j]) > len(words[i]):
                    continue
                if prefix_trie.search(words[j]) and suffix_trie.search(words[j][::-1]):
                    ans += 1
            prefix_trie.clear()
            suffix_trie.clear()
        return ans


# https://leetcode.com/problems/reverse-words-in-a-string/ 
def reverseWords(s: str) -> str:
    stack = []
    for word in s.split(" "):
        if word != "":
            stack.append(word)
    builder = []
    while stack:
        builder.append(stack.pop())
    return " ".join(builder)
            
