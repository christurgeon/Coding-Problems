import math
import heapq
import itertools
from typing import Iterator, List, Tuple


def topKLargest(k: int, stream: Iterator[str]):
    """
    Given a stream of elements, return the top k longest
    """
    min_heap = [(len(s), s) for s in itertools.islice(stream, k)] # store size and string
    heapq.heapify(min_heap)
    for next_string in stream:
        # push the next one and pop the smallest one
        heapq.heappushpop(min_heap, (len(next_string), next_string))
    return [s[1] for s in heapq.nsmallest(k, min_heap)]


def mergeSortedArrays(sorted_arrays: List[List[int]]):
    """
    Given a list of sorted arrays, merge them into a single list
    """
    min_heap = []
    sorted_arrays_iters = [iter(x) for x in sorted_arrays] 

    # put first element and index in the min_heap from each array
    for i, it in enumerate(sorted_arrays_iters):
        first = next(it, None) 
        if first is not None:
            heapq.heappush(min_heap, (first, i))

    result = []
    while min_heap:
        smallest, smallest_array_idx = heapq.heappop(min_heap)
        smallest_array_iter = sorted_arrays_iters[smallest_array_idx]
        result.append(smallest)
        next_elem = next(smallest_array_iter, None)
        if next_elem is not None:
            heapq.heappush(min_heap, (next_elem, smallest_array_idx))
    return result


def sortKIncreasingDecreasingArray(A: List[int]):
    """
    Keep track of starting and ending indexes for sorted subarrays, 
    Add the sorted ones to a list, and reverse if descending order, 
    Then merge sort them all using helper
    """
    # breaks down A into a set of sorted arrays
    sorted_subarrays = []
    increasing, decreasing = range(2)
    subarray_type = increasing
    start_index = 0
    for i in range(1, len(A) + 1):
        if (i == len(A) or # A is ended, adds last subarray
            (A[i-1] < A[i] and subarray_type == decreasing) or 
            (A[i-1] >= A[i] and subarray_type == increasing)):
            sorted_subarrays.append(A[start_index:i] if subarray_type == increasing else A[i-1:start_index-1:-1])
            start_index = i
            subarray_type = (decreasing if subarray_type == increasing else increasing)
    return mergeSortedArrays(sorted_subarrays)


def sortApproximateSortedArray(seq: Iterator[int], k: int):
    """
    Given array that is approx sorted, where inproper value is at most k spaces
    from where it should be, sort the array, use min_heap to store k values 
    """
    min_heap: List[int] = [] 
    # Add the first k elements, stop if fewer
    for x in itertools.islice(seq, k):
        heapq.heappush(min_heap, x)

    result = []
    # for every new element, add to min_heap, and extract smallest and put it to result
    for x in seq:
        result.append( heapq.heappushpop(min_heap, x) )

    # add the rest
    while min_heap:
        smallest = heapq.heappop(min_heap)
        result.append(smallest)
    return result


class Star:
    def __init__(self, x: float, y: float, z: float):
        self.x = x 
        self.y = y 
        self.z = z 
    
    @property
    def distance(self) -> float:
        return math.sqrt(self.x**2 + self.y**2, self.z**2)

    def __lt__(self, rhs: 'Star') -> bool:
        return self.distance < rhs.distance


def findClosestKStars(stars: Iterator[Star], k: int):
    """
    Given a sequence of star types and an int k, find the 
    k closest stars to the 0,0,0 coordinate (your location)
    """
    max_heap: List[Tuple[float, Star]] = []
    for s in stars:
        # add each star to heap, if size greater than k, pop one
        # insert with - since python only supports min heap
        heapq.heappush(max_heap, (-s.distance, s))
        if len(max_heap) == k + 1:
            heapq.heappop(max_heap)
        
    return [s[1] for s in heapq.nlargest(k, max_heap)]
    # time = O(nlogk) space = O(k)


def onlineMedian(seq: Iterator[int]):
    """
    Keep track of the median at all positions within the sequence
    1. min heap stores larger half seen so far
    2. max heap stores smaller half seen so far (negate them)
    3. keep them similar in size
    """
    min_heap = []
    max_heap = []
    result = []
    for x in seq:
        """
        ensure min_heap and max_heap have same number of elments if even number is read, 
        otherwise min_heap must have one more element than the max_heap
        """
        heapq.heappush(max_heap, -heapq.heappushpop(min_heap, x))
        if len(max_heap) > len(min_heap):
            heapq.heappush(min_heap, -heapq.heappop(max_heap))

        # median is average of two or just the middle one
        result.append(0.5 * (min_heap[0] + (-max_heap[0])) if len(min_heap) == len(max_heap) else min_heap[0])

    return result


def kLargestInBinaryHeap(A: List[int], k: int):
    if k <= 0:
        return 0
    
    # stores tha value (-value, index)-pair in candidate_max_heap.
    # this heap is ordered by value field, uses negative to get max heap
    # Each time we extract out max, we add the children to the heap, 
    # and those children's value is ordered 
    candidate_max_heap = []
    candidate_max_heap.append( (-A[0], 0) )
    result = []
    for _ in range(k):
        candidate_index = candidate_max_heap[0][1]
        result.append(-heapq.heappop(candidate_max_heap)[0])
        
        left_child_index = 2*candidate_index + 1
        if left_child_index < len(A):
            heapq.heappush(candidate_max_heap, (-A[left_child_index], left_child_index))
        right_child_index = 2*candidate_index + 2
        if right_child_index < len(A):
            heapq.heappush(candidate_max_heap, (-A[right_child_index], right_child_index))
        
    return result
