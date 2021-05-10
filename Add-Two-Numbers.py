# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

        
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        result_list = ListNode()
        pointer = result_list
        carrier = 0
        longer = None
        while True:
            if l1 is None:
                longer = l2
                break
            if l2 is None:
                longer = l1
                break
            added = l1.val + l2.val + carrier
            if added >= 10:
                pointer.val = added % 10
                carrier = 1
            else:
                pointer.val = added
                carrier = 0
            l1 = l1.next
            l2 = l2.next 
            if l1 or l2:
                pointer.next = ListNode()
                pointer = pointer.next
            
        if not longer and carrier == 1:
            pointer.next = ListNode()
            pointer = pointer.next
            pointer.val = 1
            return result_list
            
        while longer:
            added = longer.val + carrier
            if added >= 10:
                pointer.val = added % 10
                carrier = 1
            else:
                pointer.val = added
                carrier = 0
            longer = longer.next
            if longer:
                pointer.next = ListNode()
                pointer = pointer.next
            elif carrier == 1:
                pointer.next = ListNode()
                pointer = pointer.next
                pointer.val = 1
        
        return result_list
        