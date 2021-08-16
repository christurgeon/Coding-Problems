# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head or not head.next or k == 0:
            return head
        
        size = 1
        temp = head
        while temp.next:
            temp = temp.next
            size += 1
        if k % size == 0:
            return head
            
        first, second = head, head
        for _ in range(k % size):
            second = second.next
        while second.next:
            first, second = first.next, second.next
        temp = first.next    
        first.next = None
        second.next = head
        return temp
