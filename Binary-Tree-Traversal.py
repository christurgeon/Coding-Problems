class BinaryTreeNode:
    def __init__(self, data=None, left=None, right=None):
        self.data = data 
        self.left = left 
        self.right = right 

def treeTraversal(root: BinaryTreeNode) -> None:
    if root:
        # preorder: root, left, right
        print("Preorder", root.val)
        treeTraversal(root.left)

        # inorder: left, root, right
        print("Inorder", root.val)
        treeTraversal(root.right)

        # postorder: left, right, root
        print("Postorder", root.val)
