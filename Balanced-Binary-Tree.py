class BinaryTreeNode:
    def __init__(self, data=None, left=None, right=None):
        self.data = data 
        self.left = left 
        self.right = right 

def isBalancedBinaryTree(tree: BinaryTreeNode) -> bool:
    """
    if for each node in the tree, the difference in height of its 
    left and right subtrees is at most 1
    """
    BalancedStatusWithHeight = collections.namedtuple('BalancedStatusWithHeight', ('balanced', 'height'))

    # first value of the return value indicates if it is balanced
    # second value is the height of the tree
    def checkBalanced(tree):
        if not tree:
            return BalancedStatusWithHeight(balanced=True, height=-1)
        
        left_result = checkBalanced(tree.left)
        if not left_result.balanced:
            return left_result
        right_result = checkBalanced(tree.right)
        if not right_result.balanced:
            return right_result

        is_balanced = abs(left_result.height - right_result.height) <= 1
        height = max(left_result.height, right_result.height) + 1
        return BalancedStatusWithHeight(is_balanced, height)

    return checkBalanced(tree)

# preorder traversal: O(n)
# space bounded by height: O(h)
