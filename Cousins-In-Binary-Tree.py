# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def helper(self, root, x, y, parent_value, depth):
        if root:
            if root.val == x or root.val == y:
                return [(parent_value, depth)]
            else:
                val = root.val
                new_depth = depth + 1
                return self.helper(root.left, x, y, val, new_depth) + self.helper(root.right, x, y, val, new_depth)
        else:
            return []
        
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        if root:
            vals = self.helper(root, x, y, root.val, 0)
            if len(vals) == 2:
                return vals[0][0] != vals[1][0] and vals[0][1] == vals[1][1]
            else:
                return False
        else:
            return False
        