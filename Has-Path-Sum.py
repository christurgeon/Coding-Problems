# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    
    def helper(self, node, current_sum, target_sum):
        # print(current_sum, node.val if node else "-")
        if node and not node.left and not node.right and current_sum + node.val == target_sum:
            return True
        elif node and (node.left or node.right):
            return self.helper(node.left, current_sum + node.val, target_sum) or self.helper(node.right, current_sum + node.val, target_sum)
        return False
    
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        return self.helper(root, 0, targetSum) 
        
