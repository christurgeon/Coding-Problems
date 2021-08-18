class BinaryTreeNode:
    def __init__(self, data=None, left=None, right=None):
        self.data = data 
        self.left = left 
        self.right = right 

def isSymmetric(tree: BinaryTreeNode) -> bool:
    def checkSymmetric(subtree0, subtree1):
        if not subtree0 and not subtree1:
            return True 
        elif subtree0 and subtree1:
            return (subtree0.data == subtree1.data 
                    and checkSymmetric(subtree0.left, subtree1.right) 
                    and checkSymmetric(subtree0.right, subtree1.left))
        return False

    return not tree or checkSymmetric(tree.left, tree.right)
