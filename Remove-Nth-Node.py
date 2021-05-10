# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        size = 0
        pointer = head
        while pointer:
            pointer = pointer.next
            size += 1    
        idx = size - n + 1
        if n == size:
            head = head.next
            return head
        pointer = head
        for i in range(idx-2):
            pointer = pointer.next
        node_to_remove = pointer.next
        pointer.next = node_to_remove.next
        return head
            
        
            
        