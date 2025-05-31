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


# https://leetcode.com/problems/product-of-array-except-self/
def productExceptSelf(nums: List[int]) -> List[int]:
    prefix = suffix = []

    # compute the prefix
    curr_prod = 1
    for i in range(len(nums)):
        prefix.append(curr_prod)
        curr_prod *= nums[i]

    # compute the suffix
    curr_prod = 1
    for i in reversed(range(len(nums))):
        suffix[i] *= curr_prod
        curr_prod *= nums[i]

    return suffix


# https://leetcode.com/problems/counting-bits/
def countBits(n: int) -> List[int]:
    counter = [0]
    offset = 1
    for i in range(1, n+1): 
        if i == offset * 2:
            offset = offset * 2
        counter.append(counter[i - offset] + 1)
    return counter


# https://leetcode.com/problems/reverse-vowels-of-a-string/
def reverseVowels(s: str) -> str:
    vowel = set(["a", "e", "i", "o", "u", "A", "E", "I", "O", "U"])
    vowels_stack = [c for c in s if c in vowel]
    result = []
    for c in s:
        if c in vowel:
            result.append(vowels_stack.pop())
        else:
            result.append(c)
    return "".join(result)


# https://leetcode.com/problems/implement-trie-prefix-tree/
class TrieNode:
    def __init__(self):
        self.children = {}
        self.end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        curr_level = self.root
        for c in word:
            if c not in curr_level.children:
                curr_level.children[c] = TrieNode()
            curr_level = curr_level.children[c]
        curr_level.end = True

    def search(self, word: str) -> bool:
        curr_level = self.root
        for c in word:
            if c in curr_level.children:
                curr_level = curr_level.children[c]
            else:
                return False
        return curr_level.end

    def startsWith(self, prefix: str) -> bool:
        curr_level: TrieNode = self.root
        for c in prefix:
            if c in curr_level.children:
                curr_level = curr_level.children[c]
            else:
                return False
        return True


# https://leetcode.com/problems/guess-number-higher-or-lower/
def guessNumber(n: int) -> int:
    # The guess API is already defined for you. (def guess(num: int) -> int)
    # @return -1 if num is higher than the picked number
    #          1 if num is lower than the picked number
    #          otherwise return 0
    lo, hi = 1, n
    while lo <= hi:
        mid = (lo + hi) // 2
        result = guess(mid)
        if result == 0:
            return mid
        elif result == -1:
            hi = mid - 1
        elif result == 1:
            lo = mid + 1
        else:
            raise ValueError("guess() return unexpected value") 
    return -1


# https://leetcode.com/problems/daily-temperatures/
def dailyTemperatures(temperatures: List[int]) -> List[int]:
    n = len(temperatures)
    ans, stack = [0] * n, []
    for i in range(n):
        temperature = temperatures[i]
        while stack and stack[-1][1] < temperature:
            element = stack.pop()
            ans[element[0]] = i - element[0]
        stack.append((i, temperature))
    return ans


# https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/ 
def buildTreev1(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """
    The first element in the preorder traversal is the root of the tree. In the inorder traversal, elements to the
    left of the root belong to the left subtree, and elements to the right belong to the right subtree.

    Build the tree up to {stopping_value} from the inorder list.
    """
    p_idx = 0
    i_idx = 0
    i_len = len(inorder)
    def helper(stopping_value):
        nonlocal p_idx
        nonlocal i_idx
        if i_idx < i_len and inorder[i_idx] != stopping_value:
            curr_root_val = preorder[p_idx]
            curr_root = TreeNode(curr_root_val)
            p_idx += 1
            curr_root.left = helper(curr_root_val)
            i_idx += 1
            curr_root.right = helper(stopping_value)
            return curr_root
        return None
    return helper(None)


# # https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/ 
def buildTreev2(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    if not preorder or not inorder:
        return None
    root = TreeNode(preorder[0])
    index = inorder.index(preorder[0])
    root.left = self.buildTreev2(preorder[1:index+1], inorder[:index])
    root.right = self.buildTreev2(preorder[index+1:], inorder[index+1:])
    return root


# https://leetcode.com/problems/smallest-number-in-infinite-set/
class SmallestInfiniteSet:
    """
    The idea is that any time we add back a number, we store it inside a min heap.
    Then when we pop a number we check if the smallest in the min heap is less than
    the smallest that would otherwise be in the set.
    """
    def __init__(self):
        self.added_back_heap, self.size = [], 0
        self.lowest = 1

    def popSmallest(self) -> int:
        if self.size == 0 or self.lowest < self.added_back_heap[0]:
            self.lowest += 1
            return self.lowest - 1
        else:
            val = heapq.heappop(self.added_back_heap)
            self.size -= 1
            # Since we allow adding duplicates, continue to pop any out
            while self.size > 0 and self.added_back_heap[0] == val:
                heapq.heappop(self.added_back_heap)
                self.size -= 1
            return val

    def addBack(self, num: int) -> None:
        if num < self.lowest:
            heapq.heappush(self.added_back_heap, num)
            self.size += 1


# https://leetcode.com/problems/find-median-from-data-stream/
class MedianFinder:
    def __init__(self):
        # Stores the numbers greater than median
        self.gt = []
        # Stores the numbers less than median
        self.lt = []

    def addNum(self, num: int) -> None:
        # Add the new number to the heap that stores elements greater than the median and 
        # then pop out an element that is the smallest. We will add that element to the heap
        # holding elements that are less than the median.
        heapq.heappush(self.gt, num)
        val = heapq.heappop(self.gt)
        heapq.heappush(self.lt, -val)
        if len(self.lt) > len(self.gt):
            heapq.heappush(self.gt, -heapq.heappop(self.lt))

    def findMedian(self) -> float:
        # Equal numbers greater than and equal to the median
        if len(self.lt) == len(self.gt):
            return 0.5 * (self.gt[0] + (-self.lt[0]))
        # Extra number in `gt` heap
        else:
            return self.gt[0]


# https://leetcode.com/problems/reorder-list/ 
def reorderList(head: Optional[ListNode]) -> None:
    size = 0
    curr = head
    while curr:
        size += 1
        curr = curr.next
    if size < 3:
        return head
    
    # find the middle
    middle = head
    for _ in range(size // 2):
        middle = middle.next

    # reverse the second half of the list
    temp = middle.next
    middle.next = None
    left, right = temp, temp.next
    temp.next = None
    while right:
        temp = right.next
        right.next = left
        left = right
        right = temp

    # reconstruct the list
    dummy_head = ListNode(0, None)
    curr = dummy_head
    while head or left:
        if head:
            curr.next = head
            head = head.next
            curr = curr.next
        if left:
            curr.next = left
            left = left.next
            curr = curr.next

    return dummy_head.next


# https://leetcode.com/problems/keys-and-rooms/ 
def canVisitAllRooms(rooms: List[List[int]]) -> bool:
    def dfs(room_idx):
        for key in rooms[room_idx]:
            if key not in seen:
                seen.add(key)
                dfs(key)
    seen = set([0]) # room 0 is seen
    dfs(0)
    return len(seen) == len(rooms)


# https://leetcode.com/problems/removing-stars-from-a-string/ 
def removeStars(s: str) -> str:
    stack = []
    for c in s:
        if c == "*":
            stack.pop()
        else:
            stack.append(c)
    return "".join(stack)


# https://leetcode.com/problems/sum-of-nodes-with-even-valued-grandparent/ 
def sumEvenGrandparent(root: TreeNode) -> int:
    result = 0

    # store a queue (node, parent)
    dq = deque( [(root, None)] )
    while dq:
        (curr, parent) = dq.popleft()
        if not curr:
            continue
        if parent:
            if curr.left and parent % 2 == 0:
                result += curr.left.val
            if curr.right and parent % 2 == 0:
                result += curr.right.val
        dq.append((curr.left, curr.val))
        dq.append((curr.right, curr.val))

    return result


# https://leetcode.com/problems/deepest-leaves-sum/ 
def deepestLeavesSum(root: Optional[TreeNode]) -> int:
    max_depth, ans = 0, 0

    def helper(root, curr_depth):
        nonlocal max_depth, ans
        if not root:
            return 
        if curr_depth > max_depth:
            ans = root.val
            max_depth = curr_depth
        elif curr_depth == max_depth:
            ans += root.val
        helper(root.left, curr_depth + 1)
        helper(root.right, curr_depth + 1)

    helper(root, 0)
    return ans


# https://leetcode.com/problems/binary-search-tree-to-greater-sum-tree/ 
def bstToGst(root: TreeNode) -> TreeNode:
    # The GST is essentially everything to the right + the GST of the parent
    # Or the reverse of the in-order traversal
    running_total = 0
    def convertToGst(root):
        nonlocal answer
        if not root:
            return
        convertToGst(root.right)
        running_total += root.val
        root.val = running_total
        convertToGst(root.left)
    convertToGst(root)
    return root 
             

# https://leetcode.com/problems/asteroid-collision/
def asteroidCollision(asteroids: List[int]) -> List[int]:
    stack = []
    for asteroid in asteroids:
        add_value = True
        while stack and asteroid < 0 and stack[-1] > 0:
            diff = abs(stack[-1]) - abs(asteroid)
            if diff == 0:
                add_value = False
                stack.pop()
                break
            elif diff > 0:
                add_value = False
                break
            else:
                add_value = True
                stack.pop()
        if add_value:
            stack.append(asteroid)
    return stack


# https://leetcode.com/problems/find-peak-element/
def findPeakElement(nums: List[int]) -> int:
    L, R = 0, len(nums) - 1
    while L < R:
        mid = (L + R) // 2
        # remember: nums[i] != nums[i + 1] for all valid i
        # so, if at a given value, if we found one that is less
        # than the value to the right, then it means there must
        # be a value to the right that is a peak element
        if nums[mid] < nums[mid+1]:
            L = mid + 1
        else:
            R = mid
    return L


# https://leetcode.com/problems/rotate-array/
def rotatev1(nums: List[int], k: int) -> None:
    """
    Do not return anything, modify nums in-place instead.

    This uses O(n) space
    """
    n = len(nums)
    k = k % n
    for idx, value in enumerate(nums[n-k:n] + nums[0:n-k]):
        nums[idx] = value


# https://leetcode.com/problems/rotate-array/
def rotateV2(nums: List[int], k: int) -> None:
    """
    Do not return anything, modify nums in-place instead.

    This uses O(1) space
    """
    n = len(nums)
    k = k % n
    def reverse(l, r):
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1
    # Reverse the whole list
    reverse(0, n-1)
    # Reverse the first k elements
    reverse(0, k-1)
    # Reverse the second group et voila
    reverse(k, n-1)


# https://leetcode.com/problems/count-good-nodes-in-binary-tree/
def goodNodes(root: TreeNode) -> int:
    result = 0

    def search(node, max_seen):
        nonlocal result
        if not node:
            return
        if max_seen <= node.val:
            result += 1
        curr_max = max(node.val, max_seen)
        search(node.left, curr_max)
        search(node.right, curr_max)
        
    search(root, float('-inf'))
    return result
        

# https://leetcode.com/problems/binary-tree-right-side-view/ 
def rightSideView(root: Optional[TreeNode]) -> List[int]:
    result = []
    if not root:
        return result
    curr_stack, next_stack = [root], []
    while curr_stack:
        # Add the rightmost value to the result
        if not next_stack:
            result.append(curr_stack[-1].val)
        # Populate the next row to search
        for node in curr_stack:
            if node.left:
                next_stack.append(node.left)
            if node.right:
                next_stack.append(node.right)
        curr_stack, next_stack = next_stack, []
    return result


# https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/
def maxVowels(s: str, k: int) -> int:
    vowels = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}
    i, j, total, max_so_far = 0, 0, 0, 0
    # Advance j to setup the window
    for idx in range(k):
        if s[idx] in vowels:
            total += 1
        j += 1
    max_so_far = total
    # Advance to the end, keeping the window
    for j in range(k, len(s)):
        if s[i] in vowels:
            total -= 1
        if s[j] in vowels:
            total += 1
        max_so_far = max(max_so_far, total)
        i += 1
    return max_so_far
        

# https://leetcode.com/problems/search-in-a-binary-search-tree/
def searchBST(root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
    if root:
        if root.val == val:
            return root
        if root.val >= val:
            return self.searchBST(root.left, val)
        else:
            return self.searchBST(root.right, val)
    return None


# https://leetcode.com/problems/word-search/
def exist(board: List[List[str]], word: str) -> bool:
    m, n = len(board), len(board[0])

    def generate_search_space(row, col):
        space = []
        for x, y in [(row+1, col), (row, col+1), (row-1, col), (row, col-1)]:
            if x >= 0 and y >= 0 and x < m and y < n:
                space.append((x, y))
        return space
    
    def dfs(row, col, word_index):
        if word_index == len(word):
            return True
        if board[row][col] == word[word_index]:
            # Store the current value then set it to a garbage value so it isn't visited again
            temp = board[row][col]
            board[row][col] = ""
            result = any([dfs(x, y, word_index+1) for x, y in generate_search_space(row, col)])
            # After DFS let's reset it back, this is the backtracking idea...
            board[row][col] = temp
            return result
        return False

    for r in range(m):
        for c in range(n):
            if board[r][c] == word[0]:
                if len(word) == 1:
                    return True
                if dfs(r, c, 0):
                    return True
    return False


# https://leetcode.com/problems/total-cost-to-hire-k-workers/
def totalCost(costs: List[int], k: int, candidates: int) -> int:
    n = len(costs)
    i, j = 0, n - 1
    heap = []

    if n == 1:
        return costs[0]

    for _ in range(candidates):
        if i >= j:
            break
        heap.append((costs[i], i))
        i += 1
        heap.append((costs[j], j))
        j -= 1
    heapq.heapify(heap)

    result = 0
    for _ in range(k):
        value, idx = heapq.heappop(heap)
        result += value
        if i <= j:
            if idx < i:
                heapq.heappush(heap, (costs[i], i))
                i += 1
            elif idx > j:
                heapq.heappush(heap, (costs[j], j))
                j -= 1

    return result 


# https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/
def maxLevelSum(root: Optional[TreeNode]) -> int:
    result_level, curr_level, result_max_sum = 1, 1, float('-inf')
    nodes_to_visit = [root]
    while nodes_to_visit:
        next_nodes_to_visit = []
        curr_sum = 0
        for node in nodes_to_visit:
            curr_sum += node.val
            if node.left:
                next_nodes_to_visit.append(node.left)
            if node.right:
                next_nodes_to_visit.append(node.right)
        if curr_sum > result_max_sum:
            result_level, result_max_sum = curr_level, curr_sum
        nodes_to_visit = next_nodes_to_visit
        curr_level += 1
    return result_level


# https://leetcode.com/problems/root-equals-sum-of-children/
def checkTree(root: Optional[TreeNode]) -> bool:
    return root.val == root.left.val + root.right.val


# https://leetcode.com/problems/gas-station 
# Greedily add the running total of gas consumption as we go.
# If we drop below 0, then we can't make the jump to the next spot.
# This means that we can try reset and try from the current spot.
# Because if we got from A -> B but coult not get to C, then if 
# we can go from C -> A then we can go from C -> B.
def canCompleteCircuit(gas: List[int], cost: List[int]) -> int:
    total = start = used = 0
    for i in range(len(gas)):
        used += gas[i] - cost[i]
        total += gas[i] - cost[i]
        if total < 0:
            total = 0
            start = i + 1
    return -1 if used < 0 else start


# https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/
def removeDuplicates(nums: List[int]) -> int:
    n = len(nums)
    if n <= 2:
        return n
    write_index = 2
    i = 2
    while i < n:
        if nums[i] != nums[write_index-2]:
            nums[write_index] = nums[i]
            write_index += 1
        i += 1
    return write_index


# https://leetcode.com/problems/majority-element/
def majorityElement(nums: List[int]) -> int:
    majority_element = nums[0]
    count = 1
    for i in range(1, len(nums)):
        if nums[i] == majority_element:
            count += 1
        else:
            count -= 1
            if count == 0:
                majority_element = nums[i]
                count = 1
    return majority_element


# https://leetcode.com/problems/is-subsequence/submissions/
def isSubsequence(s: str, t: str) -> bool:
    if s is None or s == "":
        return True
    s_ptr = 0
    t_ptr = 0
    while t_ptr < len(t):
        if s[s_ptr] == t[t_ptr]:
            s_ptr += 1
            if s_ptr == len(s):
                return True
        t_ptr += 1
    return False


# https://leetcode.com/problems/ransom-note/
def canConstruct(ransomNote: str, magazine: str) -> bool:
    d = Counter(magazine)
    for c in ransomNote:
        if c not in d or d[c] == 0:
            return False
        d[c] -= 1
    return True


# https://leetcode.com/problems/isomorphic-strings/
def isIsomorphic(s: str, t: str) -> bool:
    """
        Add the first index of this character if it is not in either s or t
        Check the first index of the character and make sure it matches
    """
    s_len, t_len = len(s), len(t)
    if s_len != t_len:
        return False
    s_map, t_map = {}, {}
    for i in range(s_len):
        if s[i] not in s_map:
            s_map[s[i]] = i
        if t[i] not in t_map:
            t_map[t[i]]= i
        if s_map[s[i]] != t_map[t[i]]:
            return False
    return True


# https://leetcode.com/problems/word-pattern/
def wordPattern(pattern: str, s: str) -> bool:
    s = s.split()
    if len(pattern) != len(s) or len(set(pattern)) != len(set(s)):
        return False
    d = {}
    for i, word in enumerate(s):
        if word not in d:
            d[word] = pattern[i]
        elif d[word] != pattern[i]:
            return False
    return True


# https://leetcode.com/problems/word-pattern/
def wordPattern(pattern: str, s: str) -> bool:
    s = s.split()
    c_to_w = {}
    w_to_c = {}
    if len(pattern) != len(s):
        return False
    for char, word in zip(pattern, s):
        if char not in c_to_w and word not in w_to_c:
            c_to_w[char] = word
            w_to_c[word] = char
        elif char in c_to_w and c_to_w[char] != word:
            return False 
        elif word in w_to_c and w_to_c[word] != char:
            return False
    return True
    

# https://leetcode.com/problems/palindrome-number/
def isPalindrome(x: int) -> bool:
    if x < 0:
        return False

    # keep adding a digit to the end of the reversed number 
    y = x
    x_reverse = 0
    while y > 0:
        lowest_digit = y % 10    
        x_reverse = (x_reverse * 10) + lowest_digit
        y //= 10

    return x == x_reverse


# https://leetcode.com/problems/evaluate-division/
def calcEquation(equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:

    def bfs(start, end):
        visited = set()
        queue = deque([(start, 1)])
        while queue:
            node = queue.popleft()
            equation_letter, total = node[0], node[1]
            if equation_letter in visited:
                continue
            visited.add(equation_letter)
            for neighbor in graph[equation_letter]:
                if neighbor[0] == end:
                    return total * neighbor[1]
                queue.append((neighbor[0], total*neighbor[1]))
        return -1

    graph = defaultdict(list)
    for equation, value in zip(equations, values):
        fst, scd = equation[0], equation[1]
        graph[fst].append((scd, value))
        graph[scd].append((fst, 1/value))

    result = []
    for query in queries:
        fst, scd = query[0], query[1]
        if fst not in graph or scd not in graph:
            result.append(-1)
        else:
            result.append(bfs(fst, scd))
    return result


# https://leetcode.com/problems/minimum-size-subarray-sum/
def minSubArrayLen(target: int, nums: List[int]) -> int:
    left = 0
    subarray_running_total = 0
    subarray_minimum_length = float('inf')
    for right in range(len(nums)):
        subarray_running_total += nums[right]
        if subarray_running_total >= target:
            while subarray_running_total >= target:
                subarray_minimum_length = min(subarray_minimum_length, right - left + 1)
                subarray_running_total -= nums[left]
                left += 1
    return 0 if subarray_minimum_length == float('inf') else subarray_minimum_length


# https://leetcode.com/problems/binary-watch/
def readBinaryWatch(turnedOn: int) -> List[str]:
    def dfs(led, hour, minute, count):
        if hour > 11 or minute > 59:
            return
        if count == turnedOn:
            result.append(f"{hour}:{minute:02d}")
            return
        for i in range(led, 10):
            if i < 4: # hours
                dfs(i+1, hour + 2**i, minute, count+1)
            else: # minutes
                dfs(i+1, hour, minute + 2**(i-4), count+1)
    result = []
    dfs(0, 0, 0, 0)
    return result


# https://leetcode.com/problems/find-k-pairs-with-smallest-sums/
def kSmallestPairs(nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
    if not nums1 or not nums2 or k == 0:
        return []
    heap = []
    result = []

    # push up to k pairs of the smallest number in nums2 with smallest numbers of nums1
    smallest_num_in_nums2 = nums2[0]
    for i in range(min(k, len(nums1))):
        # (sum, i, j) -> i in nums1, j in nums2
        heapq.heappush(heap, (nums1[i] + smallest_num_in_nums2, i, 0))

    # build the result by taking this minimum element and then adding the smallest
    # pair using the smallest numbers in nums1
    while heap and len(result) < k:
        _, i, j = heapq.heappop(heap)
        result.append([nums1[i], nums2[j]])
        # if nums1[i] and nums[j+1] is currently the smallest,
        # the next smallest using a value at j from nums2 would be
        # the value at j+1
        if j+1 < len(nums2):
            heapq.heappush(heap, (nums1[i] + nums2[j+1], i, j+1))

    return result


# https://leetcode.com/problems/beautiful-arrangement/
def countArrangement(n: int) -> int:
    def backtrack(pos: int, used: list[bool]):
        if pos == n+1:
            return 1
        count = 0
        for num in range(1, n+1):
            if not used[num]:
                if num % pos == 0 or pos % num == 0:
                    used[num] = True
                    count += backtrack(pos+1, used)
                    used[num] = False
        return count

    used = [False] * (n+1)
    return backtrack(1, used)


# https://leetcode.com/problems/find-unique-binary-string/
def findDifferentBinaryString(nums: List[str]) -> str:
    # Idea: Create a new binary string that differs from the i-th string at the i-th position.
    # For each index i, the code looks at nums[i][i] — this is the diagonal element.
    # It flips that bit:
    #   - If it's '1', it becomes '0'
    #   - If it's '0', it becomes '1'
    result = []
    for i in range(len(nums)):
        digit = nums[i][i]
        if digit == "1":
            result.append("0")
        else:
            result.append("1")
    return "".join(result)


# https://leetcode.com/problems/gray-code/
def grayCode(n: int) -> List[int]:
    result = []
    for i in range(2**n):
        result.append(i ^ (i >> 1))
    return result


# https://leetcode.com/problems/letter-tile-possibilities/
def numTilePossibilities(tiles: str) -> int:
    def dfs(counter: Counter):
        total = 0
        for char, freq in counter.items():
            if freq > 0:
                total += 1
                counter[char] -= 1
                total += dfs(counter)
                counter[char] += 1
        return total
    counter = Counter(tiles)
    return dfs(counter)
