# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy_head = it = ListNode()
        while l1 and l2:
            if l1.val < l2.val:
                it.next = l1
                l1 = l1.next
            else:
                it.next = l2
                l2 = l2.next
            it = it.next
        if l1:
            it.next = l1
        if l2:
            it.next = l2
        return dummy_head.next
        