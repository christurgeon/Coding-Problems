##################################################
# A collection of additional LeetCode problems 
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


# https://leetcode.com/problems/iterator-for-combination/
class CombinationIterator:

    def __init__(self, characters: str, localCombinationLength: int):
        self.characters = characters
        self.localCombinationLength = localCombinationLength
        self.queue = deque()
        self.combinations = self.generate_combinations(0, "", self.queue)

    def generate_combinations(self, index, path, result) -> List[str]:
        if len(path) == self.localCombinationLength:
            result.append(path)
            return
        for i in range(index, len(self.characters)):
            next_index = i + 1
            next_path = path + self.characters[i]
            self.generate_combinations(next_index, next_path, result)
 
    def next(self) -> str:
        return self.queue.popleft()

    def hasNext(self) -> bool:
        return len(self.queue) > 0


# https://leetcode.com/problems/find-the-punishment-number-of-an-integer/
def punishmentNumber(n: int) -> int:
    def satisfies_constraint(target: int, value: str, index: int, current_sum: int) -> bool:
        if index == len(value):
            return current_sum == target
        for i in range(index, len(value) + 1):
            num = int(value[index:i+1])
            if satisfies_constraint(target, value, i+1, current_sum + num):
                return True
        return False

    sum_of_squares = 0
    for i in range(1, n + 1):
        square = i * i
        if satisfies_constraint(i, str(square), 0, 0):
            sum_of_squares += square
    return sum_of_squares


# https://leetcode.com/problems/odd-even-linked-list/
def oddEvenList(head: Optional[ListNode]) -> Optional[ListNode]:
    if not head:
        return None
    dummy_even_head, dummy_odd_head = ListNode(), ListNode()
    curr_even, curr_odd = dummy_even_head, dummy_odd_head
    turn = 0
    while head:
        if turn:
            curr_even.next = head
            curr_even = curr_even.next
        else:
            curr_odd.next = head
            curr_odd = curr_odd.next
        head = head.next
        turn ^= 1
    curr_even.next = None
    curr_odd.next = dummy_even_head.next
    return dummy_odd_head.next


# https://leetcode.com/problems/course-schedule-ii/
def findOrderII(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    graph = defaultdict(set)
    for course, prereq in prerequisites:
        graph[course].add(prereq)

    visited, visiting = set(), set()
    result = []
    
    def cycle(course):
        if course in visited:
            return False
        if course in visiting:
            return True
        visiting.add(course)
        for prereq in graph[course]:
            if cycle(prereq):
                return True
        visited.add(course)
        visiting.remove(course)
        result.append(course)
        return False            
            
    for course in range(numCourses):
        if cycle(course):
            return []
    return result


# https://leetcode.com/problems/snakes-and-ladders/
def snakesAndLadders(board: List[List[int]]) -> int:
    n = len(board)

    def get_next_placement(pos):
        quotient, remainder = divmod(pos - 1, n)
        row = n - 1 - quotient
        if quotient % 2 == 0:
            return row, remainder
        else:
            return row, n - 1 - remainder

    visited = set()
    queue = deque( [(1, 0)] ) # (current_place, move_count)
    while queue:
        current_place = queue.popleft()
        move_count = current_place[1]
        for i in range(1, 7):
            next_pos = current_place[0] + i
            if next_pos > n * n:
                continue
            r, c = get_next_placement(next_pos)
            if board[r][c] != -1:
                next_pos = board[r][c]
            if next_pos == n * n:
                return move_count + 1
            if next_pos in visited:
                continue
            visited.add(next_pos)
            queue.append((next_pos, move_count + 1))
    return -1


# https://leetcode.com/problems/minimum-genetic-mutation/
def minMutation(startGene: str, endGene: str, bank: List[str]) -> int:
    choices = {"A","C","G","T"}
    path_to_mutation = {}
    for gene in bank:
        path_to_mutation[gene] = float("inf")
    path_to_mutation[startGene] = 0
    if endGene not in path_to_mutation:
        return -1

    q = deque([(startGene, 0)]) # (gene, score)
    while q:
        gene, count = q.popleft()
        for i in range(len(gene)):
            for choice in choices:
                mutation = gene[:i] + choice + gene[i+1:]
                if mutation in path_to_mutation and path_to_mutation[mutation] > count + 1:
                    path_to_mutation[mutation] = count + 1
                    q.append((mutation, count + 1))
    
    return path_to_mutation[endGene] if path_to_mutation[endGene] != float("inf") else -1


# https://leetcode.com/problems/house-robber/
def rob(nums: List[int]) -> int:
    # Time - O(n)
    # Space - O(n)
    n = len(nums)
    if n == 1:
        return nums[0]
    A = [nums[0]] + ([0] * (n - 1))
    for i in range(1, n):
        A[i] = max(A[i-1], A[i-2] + nums[i])
    return A[-1]


# https://leetcode.com/problems/3sum/
def threeSum(nums: List[int]) -> List[List[int]]:
    # Time - O(n^2) sorting the array first: O(n log n) but 2-pointer for each value bumps it up
    # (n - 1) + (n - 2) + (n - 3) + ... + 1 = n(n - 1)/2 = O(n^2)
    # Space - O(1) (ignoring output)
    nums.sort()
    result = []
    n = len(nums)
    for i in range(n):
        # because it's sorted, advance to the end of this sequence
        # if i+1 < n and nums[i+1] == nums[i]:
        #     continue
        if i > 0 and nums[i] == nums[i-1]:
            continue
        value = nums[i]
        # treat current value as the lowest
        left, right = i+1, n-1
        while left < right:
            total = value + nums[left] + nums[right]
            if total == 0:
                result.append([value, nums[left], nums[right]])
                # advance left and right to get away from duplicates
                while left < right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1
                # advance one more time to get new values
                left += 1
                right -= 1
            elif total < 0: # get a larger number
                left += 1
            else: # get a smaller number
                right -= 1
    return result


# https://leetcode.com/problems/word-break/
def wordBreak(s: str, wordDict: List[str]) -> bool:
    # Time - O(n*m*k)
    #  n in len(s)
    #  m is len(wordDict)
    #  k is avg size of substrings
    # Space - O(n) 
    dp = [False] * (len(s)+1)
    dp[0] = True
    for i in range(1, len(s)+1):
        for word in wordDict:
            word_start_idx = i - len(word)
            if word_start_idx < 0:
                continue
            if dp[word_start_idx] and s[word_start_idx:i] == word:
                dp[i] = True
                break
    return dp[-1]


# https://leetcode.com/problems/maximum-subarray/
def maxSubArray(nums: List[int]) -> int:
    max_sum = float('-inf')
    curr_sum = float('-inf')
    for num in nums:
        # either add the new number or reset curr_sum
        curr_sum = max(num + curr_sum, num)
        # max_sum is just updated whenever curr_sum happens to be larger than what we've seen
        max_sum = max(max_sum, curr_sum)
    return max_sum


# https://leetcode.com/problems/maximum-sum-circular-subarray/
def maxSubarraySumCircular(nums: List[int]) -> int:
    # Time - O(n)
    # Space - O(1)
    curr_max_sum, max_sum = float("-inf"), float("-inf")
    curr_min_sum, min_sum = float("inf"), float("inf")
    total = 0
    for num in nums:
        curr_max_sum = max(num, num + curr_max_sum)
        max_sum = max(max_sum, curr_max_sum)
        curr_min_sum = min(num, num + curr_min_sum)
        min_sum = min(min_sum, curr_min_sum)
        total += num
    # the max subarray could be circular, if this is the case it can be computed
    # by taking the total sum and removing the minimum subarray
    circular_sum = total - min_sum
    # edge case: if all of the values are negative
    # so, return the maximum value in the whole nums array
    return max_sum if circular_sum == 0 else max(max_sum, circular_sum)


# https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/
def kthSmallest(matrix: List[List[int]], k: int) -> int:        
    """
    We want to use a min_heap to greedily explore the smallest number, and
    then adding its neighbor to the right and neighbor below, both of which
    are candidates for next smallest number.
    """
    n = len(matrix)
    seen = set()
    seen.add((0, 0)) # (row, col)
    min_heap = [ [matrix[0][0], 0, 0] ] # [value, row, col]

    while k > 1:
        curr = heapq.heappop(min_heap) 
        row, col = curr[1], curr[2]
        if col+1 < n and (row, col+1) not in seen:
            heapq.heappush(min_heap, [matrix[row][col+1], row, col+1])
            seen.add((row, col+1))
        if row+1 < n and (row+1, col) not in seen:
            heapq.heappush(min_heap, [matrix[row+1][col], row+1, col])
            seen.add((row+1, col))
        k -= 1
        
    return heapq.heappop(min_heap)[0]


# https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/
def kthSmallestMoreEfficient(matrix: List[List[int]], k: int) -> int:        
    """
    We can do this in O(1) space and O(2n * log(max-min)) time complexity
    by binary searching through the values and moving up or down from the middle
    depending on how many values are less than the middle value.

    e.g.

    If max = 50 and min = 10: 
        * (max - min) = (40 // 2 + min) = 30
        * We then find how many values are less than or equal to 30, if that value
          is less than k than we must now look between 10 and 30; otherwise, 
          between 30 and 50...
    """
    def countLessThanNumber(mid):
        count = 0
        row, col = 0, n - 1
        while row < n and col >= 0:
            if matrix[row][col] <= mid:
                count += col + 1
                row += 1
            else:
                col -= 1
        return count

    n = len(matrix)
    low, high = matrix[0][0], matrix[-1][-1]
    while low < high:
        diff = (high - low) // 2
        mid = diff + low
        if countLessThanNumber(mid) < k:
            low = mid + 1
        else:
            high = mid
    return low
        

# https://leetcode.com/problems/maximum-gap/
def maximumGap(nums: List[int]) -> int:
    """
    You want to divide the range [low, high] into n - 1 or n evenly spaced buckets, 
    such that each number falls into one of them. Why?
    Because if we have n elements, the maximum possible number of gaps between elements 
    in sorted order is n - 1, and the minimum possible maximum gap is: high - low // n - 1
    """
    n = len(nums)
    if n < 2:
        return 0

    # num - low: shifts the number to start from 0
    # (num - low) / (high - low): normalizes the value to a range between 0 and 1
    # Multiply by n - 1: scales it to an index in [0, n-1]
    # Use integer division (//) to round down to a valid bucket index
    low, high = min(nums), max(nums)
    bucket = defaultdict(list)
    for num in nums:
        if num == high:
            idx = n-1
        else:
            idx = (num-low) * (n-1) // (high-low)
        bucket[idx].append(num)

    # build bucket min/max summaries
    buckets = []
    for i in range(n):
        if bucket[i]:
            buckets.append( (min(bucket[i]), max(bucket[i])) )

    # compare previous bucket max to current bucket min
    result = 0
    for i in range(1, len(buckets)):
        result = max(result, buckets[i][0] - buckets[i-1][1])
    return result


# https://leetcode.com/problems/array-partition/
def arrayPairSum(nums: List[int]) -> int:
    nums.sort()
    return sum([ min(nums[i], nums[i-1]) for i in range(1, len(nums), 2)])


"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""
# https://leetcode.com/problems/populating-next-right-pointers-in-each-node/
def connect(root: 'Optional[Node]') -> 'Optional[Node]':
    if not root:
        return None
    level = deque([root])
    while level:
        next_queue = deque([])
        while level:
            node = level.popleft()
            node.next = level[0] if level else None
            if node.left:
                next_queue.append(node.left)
            if node.right:
                next_queue.append(node.right)
        level = next_queue
    return root


# https://leetcode.com/problems/populating-next-right-pointers-in-each-node/
def connectV2(root: 'Optional[Node]') -> 'Optional[Node]':
    """
    This implementation is slightly more efficient because it doesn't have two queues.
    """
    if not root:
        return None
    q = deque([root])
    while q:
        size = len(q)
        prev = None
        for _ in range(size):
            node = q.popleft()
            if prev:
                prev.next = node
            prev = node
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
    return root


# https://leetcode.com/problems/construct-string-from-binary-tree/ 
def tree2str(root: Optional[TreeNode]) -> str:
    def dfs(node):
        if not node:
            return ""
        if not node.left and not node.right:
            return str(node.val)
        if not node.right:
            return f"{node.val}({dfs(node.left)})"
        return f"{node.val}({dfs(node.left)})({dfs(node.right)})"
    return dfs(root)
        

# https://leetcode.com/problems/non-decreasing-subsequences/
def findSubsequences(nums: List[int]) -> List[List[int]]:
    n = len(nums)
    result = set()

    def dfs(index, path):
        if len(path) >= 2:
            result.add(tuple(path))
        for i in range(index, n):
            if not path or nums[i] >= path[-1]:
                dfs(i + 1, path + [nums[i]])
        
    dfs(0, [])
    return [list(sequence) for sequence in result]


# https://leetcode.com/problems/smallest-string-starting-from-leaf/
def smallestFromLeaf(root: Optional[TreeNode]) -> str:
    def itostr(value):
        return chr(value + 97)

    def backtrack(node, path):
        if not node:
            return "~" # lexicographically larger than any lowercase string
        path = itostr(node.val) + path
        if not node.left and not node.right:
            return path
        left = backtrack(node.left, path)
        right = backtrack(node.right, path)
        return min(left, right)

    return backtrack(root, "")


# https://leetcode.com/problems/the-number-of-beautiful-subsets/
def beautifulSubsets(nums: List[int], k: int) -> int:
    """
    How to build the result:
    1) count the current subset at the end (including empty)
    2) consider skipping the current value
    3) consider taking the current value (if it is valid)

    TC:
    O(nlog(n)) for the initial sort
    At each node in the recursion tree, you do an O(n) check over the current subset.
    The recursion tree can have up to O(2^n) nodes.
    """
    nums.sort()
    n = len(nums)
    def backtrack(index, subset):
        if index == n:
            return 1
        total = backtrack(index + 1, subset)
        if all(abs(nums[index] - x) != k for x in subset):
            total += backtrack(index + 1, subset + [nums[index]])
        return total
    return backtrack(0, []) - 1


# https://leetcode.com/problems/the-number-of-beautiful-subsets/
def beautifulSubsetsImprovedRuntime(nums: List[int], k: int) -> int:
    """
    How to build the result:
    1) count the current subset at the end (including empty)
    2) consider skipping the current value
    3) consider taking the current value only if value - k is not present

    TC:
    Each recusion only does O(1) extra work instead of scanning the subset
    Time: O(2^n) still exponential because you explore subsets, but no extra O(n) scanning per call.
    Space: O(n) for recursion stack and O(n) for frequence map
    """
    nums.sort()
    n = len(nums)
    frequency = defaultdict(int)
    def backtrack(index):
        if index == n:
            return 1
        total = backtrack(index + 1)
        if frequency[nums[index] - k] == 0:
            frequency[nums[index]] += 1
            total += backtrack(index + 1)
            frequency[nums[index]] -= 1
        return total
    return backtrack(0) - 1


# https://leetcode.com/problems/make-array-empty/
# This solution is not excepted due to TLE. 
# It needs to be optimized since it is O(n^2) and O(n) extra space.
def countOperationsToEmptyArray(nums: List[int]) -> int:
    operation_count = 0
    minimum_heap = nums.copy()
    heapq.heapify(minimum_heap)
    nums = deque(nums)
    while nums:
        while nums and nums[0] != minimum_heap[0]:
            nums.append(nums.popleft())
            operation_count += 1
        if nums:
            nums.popleft()
            heapq.heappop(minimum_heap)
            operation_count += 1
    return operation_count


# https://leetcode.com/problems/make-array-empty/
def countOperationsToEmptyArrayV2(nums: List[int]) -> int:
    """
    We can solve this efficiently by sorting the list while also including the index of the number in the original list.
    Then, when we iterate through we can calculate how many pops there must be before this element would be evicted from the original list.
    TC: O(nlogn)
    """
    nums = list(enumerate(nums)) # [(index, num)]
    nums.sort(key=lambda x: x[1]) # sort by num, not index
    operation_count = 0
    n = len(nums)
    for i, (original_index, _) in enumerate(nums):
        # This implies a wrap-around (rotation is needed to bring the element to the front).
        # We need to do one full rotation over remaining elements which costs.
        if i > 0 and original_index < nums[i-1][0]:
            # When a "rotation" is needed, the number of pops is n - i
            operation_count += n - i
    # Still need to account for the cost to pop each element (n in total).
    return operation_count + n


# https://leetcode.com/problems/find-the-distance-value-between-two-arrays/ 
def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
    # # O(n log m) where n = len(arr1)
    arr2.sort()
    result = 0
    arr2_len = len(arr2)
    for searchVal in arr1:
        startingIdx = bisect.bisect_left(arr2, searchVal)
        # If startingIdx < arr2_len then we've found it
        # bisect_left(arr2, searchVal) returns the index of the first 
        # element in arr2 that is not less than searchVal
        if startingIdx < arr2_len and abs(arr2[startingIdx] - searchVal) <= d:
            continue
        if startingIdx > 0 and abs(arr2[startingIdx - 1] - searchVal) <= d:
            continue
        result += 1            
    return result 


# https://leetcode.com/problems/count-and-say/
def countAndSay(n: int) -> str:
    def countAndSayHelper(iteration, string):
        if iteration == n:
            return string
        new_string, count = "", 1
        for i in range(1, len(string)):
            if string[i] != string[i-1]:
                new_string += f"{count}{string[i-1]}"
                count = 1
            else:
                count += 1
        new_string += f"{count}{string[-1]}"
        return countAndSayHelper(iteration + 1, new_string)
        
    if n == 1:
        return "1"
    return countAndSayHelper(1, "1")


# https://leetcode.com/problems/sort-integers-by-the-power-value/
def getKth(lo: int, hi: int, k: int) -> int:
    memo = {1: 0}

    def dfs(value):
        if value in memo:
            return memo[value]
        if value % 2 == 0:
            x = value // 2
        else:
            x = 3 * value + 1
        memo[value] = 1 + dfs(x)
        return memo[value]
    
    values = []
    for i in range(lo, hi+1):
        memo[i] = dfs(i)
        values.append((memo[i], i))
    values.sort()
    return values[k-1][1]


# https://leetcode.com/problems/minimum-number-of-operations-to-make-x-and-y-equal/
def minimumOperationsToMakeEqual(x: int, y: int) -> int:
    """
    When you're looking for the minimum number of operations, 
    you're essentially solving a shortest path problem in an implicit graph
    Hence, a BFS approach is lends itself natually here because with DFS we may
    geto on bad paths paths where we are just getting further and further away 
    from the solution.
    """
    if x <= y:
        return y - x
    seen = set()
    queue = deque([(x, 0)])
    while queue:
        value, steps = queue.popleft()
        if value == y:
            return steps
        if value in seen:
            continue
        seen.add(value)
        queue.append((value - 1, steps+1))
        # optional bound to prevent exploring pointless paths
        if value + 1 <= 2 * x: 
            queue.append((value + 1, steps+1))
        if value % 5 == 0:
            queue.append((value // 5, steps+1))
        if value % 11 == 0:
            queue.append((value // 11, steps+1))

    raise Exception("Failed to find a solution!")


# https://leetcode.com/problems/unique-binary-search-trees-ii/
def generateTrees(n: int) -> List[Optional[TreeNode]]:
    if n == 0:
        return []
    memo = {}

    def generate(start, end):
        if start > end:
            return [None]
        if (start, end) in memo:
            return memo[(start, end)]

        result = []
        for i in range(start, end+1):
            left = generate(start, i-1)
            right = generate(i+1, end)
            for left_node in left:
                for right_node in right:
                    node = TreeNode(i)
                    node.left = left_node
                    node.right = right_node
                    result.append(node)
        memo[(start, end)] = result
        return result

    return generate(1, n)


# https://leetcode.com/problems/binary-search-tree-iterator/
#     next(): Amortized O(1) per operation.
#     hasNext(): O(1)
#     Space: O(h), where h is the height of the tree.
class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.stack = []
        self._build_leftmost_inorder(root)

    def _build_leftmost_inorder(self, node: TreeNode):
        while node:
            self.stack.append(node)
            node = node.left

    def next(self) -> int:
        top = self.stack.pop()
        if top.right:
            self._build_leftmost_inorder(top.right)
        return top.val

    def hasNext(self) -> bool:
        return len(self.stack) > 0

# https://leetcode.com/problems/unique-paths/
# TC: O(n*m)
# Space: O(n*m)
def uniquePaths(m: int, n: int) -> int:
    if m == 1 and n == 1:
        return 1
    dp = [[0 for _ in range(n)] for _ in range(m)]
    # compute the bottom row
    for i in range(n-2, -1, -1):
        dp[m-1][i] = 1
    # compute rightmost column
    for i in range(m-2, -1, -1):
        dp[i][n-1] = 1
    # compute the middle        
    for i in range(m-2, -1, -1):
        for j in range(n-2, -1, -1):
            dp[i][j] = dp[i+1][j] + dp[i][j+1]
    return dp[0][0]

# TC: O(n*m)
# Space: O(n)
def uniquePathsOptimized(m: int, n: int) -> int:
    dp = [1] * n
    for _ in range(m-1):
        for j in range(n-2, -1, -1):
            dp[j] += dp[j+1]
    return dp[0]


# https://leetcode.com/problems/unique-paths-ii/
def uniquePathsWithObstacles(obstacleGrid: List[List[int]]) -> int:
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    dp = [[0 for _ in range(n)] for _ in range(m)]
    # build the bottom dp row
    for i in range(n-1, -1, -1):
        if obstacleGrid[m-1][i] == 1:
            dp[m-1][i] = 0
        elif i < n-1 and dp[m-1][i+1] == 0:
            dp[m-1][i] = 0
        else:
            dp[m-1][i] = 1
    # build the rightmost dp column
    for i in range(m-1, -1, -1):
        if obstacleGrid[i][n-1] == 1:
            dp[i][n-1] = 0
        elif i < m-1 and dp[i+1][n-1] == 0:
            dp[i][n-1] = 0
        else:
            dp[i][n-1] = 1
    # construct the rest of the board
    for i in range(m-2, -1, -1):
        for j in range(n-2, -1, -1):
            if obstacleGrid[i][j] == 1:
                dp[i][j] = 0
            else:
                dp[i][j] = dp[i+1][j] + dp[i][j+1]
    return dp[0][0]


# https://leetcode.com/problems/missing-number/
def missingNumber(nums: List[int]) -> int:
    n = len(nums)
    value = n * (n + 1) // 2
    for n in nums:
        value -= n
    return value


# https://leetcode.com/problems/find-the-duplicate-number/
# TC: O(n)
# Space: O(1)
def findDuplicate(nums: List[int]) -> int:
    slow, fast = nums[0], nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]] # move two steps
        if slow == fast:
            break
    # find the start of the cycle
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    return slow


# https://leetcode.com/problems/first-missing-positive/
def firstMissingPositive(nums: List[int]) -> int:
    n = len(nums)
    # replace numbers that are not candidates with a default value
    for i in range(n):
        if nums[i] <= 0 or nums[i] > n:
            nums[i] = n + 1
    # mark indices with a negative number
    for num in nums:
        val = abs(num)
        if 1 <= val <= n and nums[val-1] > 0:
            nums[val-1] *= -1
    # loop through all valid values
    for i in range(n):
        if nums[i] > 0:
            return i + 1
    return n + 1


# https://leetcode.com/problems/insert-delete-getrandom-o1/
class RandomizedSet:

    def __init__(self):
        self.map = dict()
        self.list = []

    def insert(self, val: int) -> bool:
        if val in self.map:
            return False
        self.list.append(val)
        self.map[val] = len(self.list) - 1
        return True

    def remove(self, val: int) -> bool:
        if val not in self.map:
            return False
        index = self.map[val]
        last_element = self.list[-1]
        self.list[index] = last_element
        self.map[last_element] = index
        self.list.pop()
        del self.map[val]
        return True

    def getRandom(self) -> int:
        return random.choice(self.list)


# The rand7() API is already defined for you.
# def rand7():
# @return a random integer in the range 1 to 7
def rand10():
    while True:
        # Generate a uniform distribution from [1, 49]
        result = (rand7() - 1) * 7 + rand7()
        # [1, 40] is also uniform and we can easily reduce it to [1, 10]
        if result <= 40:
            return (result - 1) % 10 + 1
        

# https://leetcode.com/problems/course-schedule-iii/
def scheduleCourse(courses: List[List[int]]) -> int:
    """
    1. Sort courses by deadlines to prioritize earlier deadlines
    2. Use a max-heap to keep track of durations of selected courses
    Sorting: O(n log n)
    """
    courses.sort(key=lambda x: x[1])
    total_time, max_heap = 0, []
    for duration, lastDay in courses:
        total_time += duration
        heapq.heappush(max_heap, -duration)
        # if we exceed deadline, drop the longest course
        if total_time > lastDay:
            longest_duration = -heapq.heappop(max_heap)
            total_time -= longest_duration
    return len(max_heap)


# https://leetcode.com/problems/power-of-three/
def isPowerOfThree(n: int) -> bool:
    if n == 1:
        return True
    if n <= 0 or n % 3 != 0:
        return False
    return self.isPowerOfThree(n // 3)


# https://leetcode.com/problems/minimum-path-sum/
def minPathSum(grid: List[List[int]]) -> int:
    n, m = len(grid), len(grid[0])
    dp = [[0 for _ in range(m)] for _ in range(n)]
    dp[-1][-1] = grid[-1][-1]
    # Compute the bottom row
    for i in range(m-2, -1, -1):
        dp[-1][i] = dp[-1][i+1] + grid[-1][i]
    # Compute the rightmost column
    for i in range(n-2, -1, -1):
        dp[i][-1] = dp[i+1][-1] + grid[i][-1]
    # Work backwards to the beginning 
    for row in range(n-2, -1, -1):
        for col in range(m-2, -1, -1):
            move_down_cost = dp[row+1][col]
            move_right_cost = dp[row][col+1]
            dp[row][col] = min(move_down_cost, move_right_cost) + grid[row][col]
    return dp[0][0]
        

# https://leetcode.com/problems/contiguous-array/
def findMaxLength(nums: List[int]) -> int:
    """
    Treat 0 as -1 and 1 as 1, whenever the balance is 0 
    then there are an equal number of 0s and 1s. Balance 
    tracks the cumulative count of 0s and 1s encountered.

    Initialize the tracker map with a balance of 0 mapped to 
    the index -1 to special case where the valid subarray
    starts at index 0.

    @returns result, the longest contiguous subarray w/ an equal number
             of 0s and 1s
    TC - O(n)
    Space - O(n)
    """
    result = balance = 0
    tracker = {0: -1}
    for i, num in enumerate(nums):
        balance = balance+1 if num == 1 else balance-1
        if balance in tracker:
            result = max(result, i - tracker[balance])
        else:
            tracker[balance] = i
        return result


# https://leetcode.com/problems/random-pick-with-weight/
class Solution:
    
    def __init__(self, w: List[int]):
        self._total = 0
        self._running_sum = []
        for weight in w:
            self._total += weight
            self._running_sum.append(self._total)

    def pickIndex(self) -> int:
        choice = random.randint(1, self._total)
        return self._bisect_left(choice)

    def _bisect_left(self, choice):
        lo, hi = 0, len(self._running_sum)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._running_sum[mid] < choice:
                lo = mid + 1
            else:
                hi = mid
        return lo
        
# https://leetcode.com/problems/sort-list/
class Solution:
    def sortList_nsquared(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy_head = ListNode(0)
        curr = head
        while curr:
            # At each step, remove curr and insert it into the sorted list
            prev = dummy_head
            while prev.next and prev.next.val < curr.val:
                prev = prev.next
            temp = curr.next
            curr.next = prev.next
            prev.next = curr
            curr = temp
        return dummy_head.next

    def sortList_nlogn(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        # Split list into two halves
        mid = self.getMiddleNode(head)
        left, right = head, mid.next
        mid.next = None
        # Sort the two sublits and merge them
        sorted_left, sorted_right = self.sortList(left), self.sortList(right)
        return self.mergeTwoLists(sorted_left, sorted_right)

    def getMiddleNode(self, node):
        slow, fast = node, node.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def mergeTwoLists(self, l1, l2):
        dummy_head = ListNode()
        tail = dummy_head
        while l1 and l2:
            if l1.val < l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next
        tail.next = l1 if l1 else l2
        return dummy_head.next
        

# https://leetcode.com/problems/minimum-number-of-flips-to-convert-binary-matrix-to-zero-matrix/
def minFlips(mat: List[List[int]]) -> int:
    """
    TC: O(2^(n*m)) because DFS is trying trying every combination of flips for each cell: and there are 2 combinations
    Space: O(mn), for the depth of the DFS recursion.
    """
    n, m = len(mat), len(mat[0])
    INFINITE = float('inf')
    
    def flip(x, y) -> None:
        for i, j in ((x,y), (x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if 0 <= i < n and 0 <= j < m: 
                mat[i][j] ^= 1

    def isZeroMatrix() -> bool:
        return all(sum(row) == 0 for row in mat)

    def dfs(x, y, step_count):
        if x == n:
            return step_count if isZeroMatrix() else INFINITE
        if y == m:
            return dfs(x+1, 0, step_count)

        # Explore flipping and not flipping
        no_flip_count = dfs(x, y+1, step_count)
        flip(x, y) # flip
        yes_flip_count = dfs(x, y+1, step_count + 1)
        flip(x, y) # flip back to reset the matrix
        return min(no_flip_count, yes_flip_count)

    result = dfs(0, 0, 0)
    return result if result != float('inf') else -1

# Ensures we never recompute the same thing because we're using BFS
def minFlipsWithBitMask(self, mat: list[list[int]]) -> int:
    """
    Space: O(2^mn)
    """
    n, m = len(mat), len(mat[0])
    
    # Convert matrix to initial bitmask
    start = 0
    for i in range(n):
        for j in range(m):
            if mat[i][j]:
                start |= 1 << (i*m + j)
    if start == 0:
        return 0
    
    # Precompute flip masks for each cell
    flip_masks = []
    for i in range(n):
        for j in range(m):
            mask = 0
            for x, y in [(i,j), (i+1,j), (i-1,j), (i,j+1), (i,j-1)]:
                if 0 <= x < n and 0 <= y < m:
                    mask |= 1 << (x*m + y)
            flip_masks.append(mask)
    
    # BFS
    queue = deque([(start, 0)]) # (matrix state, steps)
    visited = {start}
    while queue:
        state, steps = queue.popleft()
        if state == 0:
            return steps
        for mask in flip_masks:
            new_state = state ^ mask
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, steps + 1))
    return -1


# https://leetcode.com/problems/substring-with-concatenation-of-all-words/
def findSubstring(s: str, words: List[str]) -> List[int]:
    """
    This solution works, but it has O(N*M*L) time complexity in the worst case, where:
        N = len(s)
        M = len(words)
        L = len(words[0])
    """
    result = []
    if not s or not words:
        return result

    # Count the occurrences of each word
    lookup = defaultdict(int)
    for w in words:
        lookup[w] += 1

    # All words are the same length
    word_len = len(words[0])
    substr_len = word_len * len(words)

    # The words has to start from the beginning of the substring
    # "abcde" and substr_len = 2
    #     ^ start here 
    for i in range(len(s) - substr_len + 1):
        used = defaultdict(int)
        valid = True
        for j in range(i, i + substr_len, word_len):
            word = s[j:j+word_len]
            used[word] += 1
            # Either this word doesn't exist in our words list 
            # or we've used a word from the words list too many times...
            if word not in lookup or used[word] > lookup[word]:
                valid = False
                break
        if valid:
            result.append(i)
    return result


# https://leetcode.com/problems/text-justification/
def fullJustify(words: List[str], maxWidth: int) -> List[str]:
    result = []
    wordHolder = []
    aggWordLength = 0

    def writeLine(wordHolder, aggWordLength, result, isLastLine=False):
        availableSpaces = maxWidth - aggWordLength
        # Left-justify if we have one word or we are the last line
        if len(wordHolder) == 1 or isLastLine:
            newLine = " ".join(wordHolder)
            newLine += " " * (maxWidth - len(newLine))
        else:
            numGaps = len(wordHolder) - 1
            spacesPerLine = availableSpaces // numGaps
            extraSpaces = availableSpaces % numGaps
            newLine = ""
            for i, word in enumerate(wordHolder):
                newLine += word
                if i < numGaps:
                    newLine += " " * (spacesPerLine + (1 if i < extraSpaces else 0))
        result.append(newLine)

    for i, w in enumerate(words):
        newWordSize = len(w)
        newAggSize = aggWordLength + newWordSize + len(wordHolder)
        if newAggSize <= maxWidth:
            wordHolder.append(w)
            aggWordLength += newWordSize
        else:
            writeLine(wordHolder, aggWordLength, result)
            wordHolder = [w]
            aggWordLength = newWordSize

    # Handle the last line
    if wordHolder:
        writeLine(wordHolder, aggWordLength, result, isLastLine=True)
    return result


# https://leetcode.com/problems/candy/
def candy(ratings: List[int]) -> int:
    n = len(ratings)
    candies = [1] * n    
    # Pass 1: left to right, if we are higher than previous, increase candy
    for i in range(1, n):
        if ratings[i] > ratings[i-1]:
            candies[i] = candies[i-1] + 1
    # Pass 2: right to left, if we are higher than previous, increase candy
    for i in range(n-2, -1, -1):
        if ratings[i] > ratings[i+1]:
            candies[i] = max(candies[i], candies[i+1] + 1)
    return sum(candies)


# https://leetcode.com/problems/number-of-digit-one
@lru_cache(maxsize=None) # cache results for all inputs
def countDigitOne(n: int) -> int:
    """
    Time: O(log n)
    Space: O(log n) (recursion depth + cache).
    """
    if n <= 0:
        return 0
    if n < 10:
        return 1

    # Find the largest power of 10 <= n (i.e., div)
    div = 1
    while div * 10 <= n:
        div *= 10
    leftmost_digit = n // div   # most significant digit
    rem = n % div               # remainder after removing high digit

    # count of '1's contributed by the most significant digit
    # if n is 134: then leftmost_digit == 1 so count([100, 134]) == 35
    # if n is 234: then leftmost_digit == 2 so count([100, 199]) == 100
    high_contribution = rem + 1 if leftmost_digit == 1 else div
    # count of '1's contributed by lower digits for all numbers with high digit set
    # if n is 134: then mult 1 by recursive result of 99
    # if n is 234: then mult 2 (bc we need to get 100s and 200s) by recursive result of 99
    lower_contribution = leftmost_digit * self.countDigitOne(div-1)
    # count of '1's contributed by the remainder
    # if n is 134:  134 % 100 == 34
    # if n is 1134: 1234 % 1000 == 134
    remainder_contribution = self.countDigitOne(rem)
    return high_contribution + lower_contribution + remainder_contribution


# https://leetcode.com/problems/perfect-rectangle/
def isRectangleCover(rectangles: List[List[int]]) -> bool:
    # Each corner should appear 1 time, so 4 in total.
    # Other corners should apear an even number of times, meaning
    # it has a match, if it's an odd number of times.
    #
    # But we still have the issue of overlaps to deal with.
    # Another insight is that the sum of all small rectangles should equal big rectangle.
    # If this isn't this case, then there are gaps or an overlap...
    subrectangle_total_area = 0
    corners = set()
    for x, y, X, Y in rectangles:
        corners ^= {(x, y), (x, Y), (X, y), (X, Y)}
        subrectangle_total_area += (X-x) * (Y-y)
    if len(corners) != 4:
        return False
    min_x = min(c[0] for c in corners)
    min_y = min(c[1] for c in corners)
    max_x = max(c[0] for c in corners)
    max_y = max(c[1] for c in corners)
    global_rectangle_area = (max_x-min_x) * (max_y-min_y)
    return subrectangle_total_area == global_rectangle_area


# https://leetcode.com/problems/frog-jump/
def canCross(stones: List[int]) -> bool:
    if not stones or len(stones) == 1:
        return True
    # It always requires 1 jump to get to the second stone
    if len(stones) > 1 and stones[1] > 1:
        return False

    n = len(stones)
    dp = dict()

    def dfs(i, k):
        if (i, k) in dp:
            return dp[(i, k)]
        if k == 0:
            return False
        if i == n - 1:
            return True
        start = i+1
        end = i+k+1
        for j in range(start, end+1):
            # We've gone too far
            if j >= n:
                break
            cost_to_jump = stones[j] - stones[i]
            # Next k is not valid, end the loop here
            if cost_to_jump > k+1:
                break
            # Recurse on the next valid move
            if cost_to_jump in (k-1, k, k+1):
                if dfs(j, cost_to_jump):
                    return True
        # Record the result for the index and k
        dp[(i, k)] = False
        return False

    return dfs(1, 1)


# https://leetcode.com/problems/longest-increasing-path-in-a-matrix/
def longestIncreasingPath(matrix: List[List[int]]) -> int:
    n, m, memo = len(matrix), len(matrix[0]), dict()

    def getNextMoves(x, y):
        result = []
        for nx, ny in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if 0 <= nx < n and 0 <= ny < m:
                result.append((nx, ny))
        return result
    
    def dfs(x, y):
        if (x, y) in memo:
            return memo[(x, y)]
        longest_path = 0
        for nx, ny in getNextMoves(x, y):
            if matrix[nx][ny] > matrix[x][y]:
                longest_path = max(longest_path, dfs(nx, ny))
        memo[(x, y)] = longest_path + 1
        return longest_path + 1

    result = 0
    for x in range(n):
        for y in range(m):
            result = max(result, dfs(x, y))
        return result


# https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii
def maxProfit(prices: List[int]) -> int:
    n = len(prices)
    if n == 0:
        return 0

    @lru_cache(None)
    def dfs(i, transactions_left, holding):
        """
        We have 3 options:
        1. Do nothing
        2. Sell today
        3. Buy today (if transactions allow it)
        """
        if i == n or transactions_left == 0:
            return 0
        profit = dfs(i+1, transactions_left, holding)
        if holding:
            profit = max(profit, prices[i] + dfs(i+1, transactions_left-1, False))
        else:
            profit = max(profit, -prices[i] + dfs(i+1, transactions_left, True))
        return profit

    return dfs(0, 2, False)


# https://leetcode.com/problems/sliding-window-median/
def medianSlidingWindow(nums: List[int], k: int) -> List[float]:
    """
    O(nlogk) solution
    O(k) space complexity
    """
    from sortedcontainers import SortedList
    L = SortedList()
    result = []
    isMiddle = k % 2 == 1
    for i in range(len(nums)):
        L.add(nums[i]) # O(logk) bc uses binary search to insert
        if len(L) > k:
            L.remove(nums[i-k]) # O(logk) removal
        if len(L) == k:
            result.append(L[k//2] if isMiddle else (L[k//2-1] + L[k//2]) / 2)
    return result


# https://leetcode.com/problems/patching-array/
def minPatches(nums: List[int], n: int) -> int:
    result = 0
    nextNumberToForm = 1
    i = 0
    while nextNumberToForm <= n:
        # If the number we are on is less than the number we can form.
        # Advance by the number itself.
        if i < len(nums) and nums[i] <= nextNumberToForm:
            nextNumberToForm += nums[i]
            i += 1
        # Add this number. 
        # Therefore the next value we can reach is now double.
        else:
            nextNumberToForm += nextNumberToForm
            result += 1
    return result


# https://leetcode.com/problems/sort-an-array/
def sortArray(nums: List[int]) -> List[int]:
    """
    Perform a bucket sort
    TC O(n)
    SC O(max - min + 1)
    """
    NINF = float('-inf')
    maximum, minimum = max(nums), min(nums)
    bucket = [(NINF, 0) for _ in range(maximum - minimum + 1)]
    offset = 0
    bucket = [0] * (maximum - minimum + 1)
    for num in nums:
        bucket[num - minimum] += 1
    result = []
    for i, count in enumerate(bucket):
        if count > 0:
            result += [i + minimum] * count
    return result


# https://leetcode.com/problems/contains-duplicate-iii/
def containsNearbyAlmostDuplicate(nums: List[int], indexDiff: int, valueDiff: int) -> bool:
    """
    This solution exceeds the timeout...
    TC: O(nlogn) - this solution times out
    SC: O(n)
    """
    lookup = sorted((val, i) for i, val in enumerate(nums))
    n = len(lookup)
    i = 0
    for j in range(n):
        while i < j and lookup[j][0] - lookup[i][0] > valueDiff:
            i += 1
        for k in range(i, j):
            if abs(lookup[j][1] - lookup[k][1]) <= indexDiff:
                return True
        return False

def containsNearbyAlmostDuplicate(nums: List[int], indexDiff: int, valueDiff: int) -> bool:
    """
    This solution exceeds the timeout...
    TC: O(n*valueDiff)
    SC: O(n)
    """
    n = len(nums)
    if n <= 1:
        return False
    lookup = {} # value -> latest index
    for i, num in enumerate(nums):
        # Check num itself (duplicate value case)
        if num in lookup and i - lookup[num] <= indexDiff:
            return True
       # Check neighbors within valueDiff
        for diff in range(1, valueDiff+1):
            if num+diff in lookup and i-lookup[num+diff] <= indexDiff:
                return True
            if num-diff in lookup and i-lookup[num-diff] <= indexDiff:
                return True
        lookup[num] = i
    return False

def containsNearbyAlmostDuplicate(nums: List[int], indexDiff: int, valueDiff: int) -> bool:
    """
    Optimal solution for TC:
    TC: O(n)
    SC: O(indexDiff)

    valueDiff = 2
    indexDiff = 2
     
    -1 => {-1,-2,-3}
    0  => {0,1,2}
    1  => {3,4,5}
    """
    bucket = dict()
    bucketSize = valueDiff + 1
    for i, num in enumerate(nums):
        bucketId = num // bucketSize
        # Check the current and neighboring buckets
        if bucketId in bucket and abs(num - bucket[bucketId]) <= valueDiff:
            return True
        if bucketId + 1 in bucket and abs(num - bucket[bucketId + 1]) <= valueDiff:
            return True
        if bucketId - 1 in bucket and abs(num - bucket[bucketId - 1]) <= valueDiff:
            return True
        # Insert current num into its bucket
        bucket[bucketId] = num
        # Delete a bucket that is outside of our range
        if i >= indexDiff:
            valueToRemove = nums[i - indexDiff]
            oldBucketId = valueToRemove // bucketSize
            del bucket[oldBucketId]
    return False


# https://leetcode.com/problems/integer-to-english-words/
def numberToWords(num: int) -> str:
    lookup_digit = {
        "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"
    }
    lookup_teens = {
        "11": "Eleven", "12": "Twelve", "13": "Thirteen", "14": "Fourteen", "15": "Fifteen", "16": "Sixteen", "17": "Seventeen", "18": "Eighteen", "19": "Nineteen"
    }
    lookup_tens = {
        "2": "Twenty", "3": "Thirty", "4": "Forty", "5": "Fifty", "6": "Sixty", "7": "Seventy", "8": "Eighty", "9": "Ninety"
    }
    lookup_larger = ["", "Thousand", "Million", "Billion", "Trillion"]

    if num < 10:
        return lookup_digit[str(num)]

    def processThreeDigitChunk(chunk):
        chunk_result = []
        if len(chunk) == 3 and chunk[0] != "0":
            chunk_result.append(lookup_digit[chunk[0]])
            chunk_result.append("Hundred")
        if len(chunk) >= 2 and chunk[-2] != "0":
            if chunk[-2:] == "10":
                chunk_result.append("Ten")
                return chunk_result
            if chunk[-2] == "1":
                chunk_result.append(lookup_teens[chunk[-2:]])
                return chunk_result
            chunk_result.append(lookup_tens[chunk[-2]])
        if len(chunk) >= 1 and chunk[-1] != "0":
            chunk_result.append(lookup_digit[chunk[-1]])
        return chunk_result

    result = deque([])
    value = str(num)
    end = len(value) - 1
    placeholder = 0
    while end >= 0:
        chunk = value[max(0,end-2):end+1]
        chunk_result = processThreeDigitChunk(chunk)
        # Determine if we should add a "thousand", "million", etc.
        if chunk_result:
            if placeholder > 0:
                result.appendleft(lookup_larger[placeholder])
            # Add our next chunk to the result
            for s in reversed(chunk_result):
                result.appendleft(s)
        end -= 3
        placeholder += 1

    return " ".join(result)


# https://leetcode.com/problems/minimum-number-of-vertices-to-reach-all-nodes/
def findSmallestSetOfVertices(n: int, edges: List[List[int]]) -> List[int]:
    degree = [0] * n
    for u, v in edges:
        degree[v] += 1
    # A node will have degree 0 if it has no edges to it.
    # Therefore, in order for all nodes to be reachable, it
    # must be included. For any node with an edge to it, 
    # we can just include the "from node" if we wish to include
    # the smallest set of vertices...
    result = []
    for node in range(n):
        if degree[node] == 0:
            result.append(node)
    return result


# https://leetcode.com/problems/path-sum-ii/description/ 
def pathSum(root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
    solutions = []

    def helper(root: Optional[TreeNode], currSum: int, currSolution: []):
        if not root:
            return
        newCurrSum = currSum + root.val
        currSolution.append(root.val)
        if not root.left and not root.right:
            if newCurrSum == targetSum:
                solutions.append(currSolution)
        if root.left:
            helper(root.left, newCurrSum, currSolution.copy())
        if root.right:
            helper(root.right, newCurrSum, currSolution.copy())

    helper(root, 0, [])
    return solutions


# https://leetcode.com/problems/triangle/ 
def minimumTotal(triangle: List[List[int]]) -> int:
    # compute the solution in place using the triangle list
    #
    # [[2],
    #  [3,4],
    #  [6,5,7], <- we start here and loop through every digit in
    #  [4,1,8,3]]  the column and then we set it to the minimum of the  
    #              sum of the value below and the value below and to the right...
    #              therefore, by the time we reach the top, voila that's out answer
    size = len(triangle)
    for row in range(size - 2, -1, -1):
        for col in range(len(triangle[row])):
            triangle[row][col] = min(triangle[row][col] + triangle[row + 1][col], triangle[row][col] + triangle[row + 1][col + 1])
    return triangle[0][0]


