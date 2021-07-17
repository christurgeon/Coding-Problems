# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head
        
        dummy_head = head.next
        prior = None
        first = head
        second = head.next
        while first and second:
            first_cache = first
            second_cache = second

            temp = second.next
            second.next = first
            first.next = temp
            first = temp
            
            if temp is not None:
                second = temp.next
            else:
                second = None
                
            if prior:
                prior.next = second_cache
            prior = first_cache
            
        return dummy_head