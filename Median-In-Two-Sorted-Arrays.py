class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        M, N = len(nums1), len(nums2)
        total = (M + N + 1) // 2
         
        # let N be the larger array
        if M > N:
            nums1, nums2 = nums2, nums1
            M, N = N, M
        
        # bin search through smaller array M
        low, high = 0, M
        while low <= high:
            mid_M = low + (high - low) // 2
            mid_N = total - mid_M
            M_left  = float('-inf') if mid_M == 0 else nums1[mid_M-1]
            N_left  = float('-inf') if mid_N == 0 else nums2[mid_N-1]
            M_right = float('inf')  if mid_M == M else nums1[mid_M]
            N_right = float('inf')  if mid_N == N else nums2[mid_N]

            if M_left <= N_right and N_left <= M_right:
                if (M + N) % 2 == 1:
                    return max(M_left, N_left)
                else:
                    return (max(M_left, N_left) + min(M_right, N_right)) / 2.0

            if M_left > N_right:
                high = mid_M - 1
            else:
                low = mid_M + 1
        
        return -1
