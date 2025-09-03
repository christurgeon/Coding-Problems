#!/bin/python3

import math
import os
import random
import re
import sys
import collections
from typing import List 
import numpy as np
import pandas as pd


# Complete the 'fizzBuzz' function below.
# The function accepts INTEGER n as parameter.
def FizzBuzz(n: int):
    for i in range(1, n + 1):
        mod3 = (i % 3 == 0)
        mod5 = (i % 5 == 0)
        if mod3 and mod5:
            print("FizzBuzz")
        elif mod3:
            print("Fizz")
        elif mod5:
            print("Buzz")
        else:
            print(i)


# A method to swap two variables without using a temp
def SwapTwoVariables(x: int, y: int):
    x = x + y   # x + y
    y = x - y   # x
    x = x - y   # x + y - x
    return x, y


def CountTheFrequencyOfAnElementInAnArray(A):
    d = dict()
    for element in A:
        if element not in d:
            d[element] = 1
        else:
            d[element] += 1
    return d


def CountTheFrequencyOfAnElementInAnArrayCounter(A):
    return collections.Counter(A)


def ConcatenateTwoLists(nums: List[int]) -> List[int]:
    return nums + nums


def BinarySearchAnArray(A, x):
    low, high = 0, len(A) - 1
    while low <= high:
        mid = (low + high) // 2 # average of low and high
        
        if A[mid] > x:
            high = mid - 1
        elif A[mid] < x:
            low = mid + 1
        else:
            return True
    return False
    

def MaxTickets(tickets: List[int]) -> int:
    t = sorted(tickets)
    longest = 1
    start, end = 0, 1
    while end < len(t):
        if abs(t[end] - t[end-1]) <= 1:
            end += 1 
        else:
            diff = end - start 
            longest = max(longest, diff)
            start = end
            end += 1
    return max(longest, end - start)


# Put M idenctical apples on N identical plates, 
# allow some plates to be left empty, 
# how many different methods are there?
#
# 0<=m<=10,1<=n<=10. 0<=n<=10<=m<=10
#
# Cases:
# (1) when number of plates is more than the number of apples, 
#     there must be n-m leftover plates 
# (2) when number of plates is less than the number of apples,
#     a. when there is an empty plate at least one plate is empty 
#     b. when there is no empty plate, all plates have apples,
#        and removing one from each plate has no effect
def MandNApples(m: int, n: int):
    if m == 0 or n == 0: # all apples on a plate, or no apples
        return 1
    if m < n:
        return MandNApples(m, m) # number of plates remaining is equal to the number of apples equal to m
    if m >= n:
        return MandNApples(m, n-1) + MandNApples(m-n, n)


def Wildcards(param):
  """
  +    represents single alphabetical character
  $    represents a single digit
  {X}  represents number of repetitions, should be at least one
  *    means 3 repetitions of the string
  """
  args = param.split(" ")
  if len(args) != 2:
    raise Exception("Invalid Input! [{}] not accepted...".format(param))
  lookup, target = args[0], args[1]
  lookup_iter = len(lookup) - 1
  target_iter = len(target) - 1
  
  try:
    while lookup_iter >= 0 and target_iter >= 0:
      char = lookup[lookup_iter]

      if char == "*":
        repeated_char = target[target_iter]
        for i in reversed(range(target_iter - 2, target_iter + 1)):
          if target[i] != repeated_char:
            return False
        lookup_iter -= 1
        target_iter -= 3
      elif char == "+":
        if not target[target_iter].isalpha():
          return False
        lookup_iter -= 1
        target_iter -= 1
      elif char == "$":
        if not target[target_iter].isdigit():
          return False
        lookup_iter -= 1
        target_iter -= 1
      elif char == "}":
        digit = []
        lookup_iter -= 1
        while lookup[lookup_iter] != "{":
          digit.append(lookup[lookup_iter])
          lookup_iter -= 1
        digit = "".join(reversed(digit))
        if not digit.isdigit() or lookup[lookup_iter - 1] != "*":
          return False
        digit = int(digit)
        repeated_char = target[target_iter]
        for i in reversed(range(target_iter - digit + 1, target_iter)):
          if target[i] != repeated_char:
            return False
        lookup_iter -= 2
        target_iter -= digit

      else:
        # unexpected character has appeared
        return False
  except IndexError:
    # wildcard pattern had mismatched expected size, we know it's invalid
    return False

  return target_iter == -1


def TowerOfHanoi(n, fromRod, toRod, auxRod):
    if n == 1:
        print("Moved disk 1 from rod", fromRod, "to rod", toRod)
    TowerOfHanoi(n-1, fromRod, auxRod, toRod)
    print("Moved disk", n, "from rod", fromRod, "to rod", toRod)
    TowerOfHanoi(n-1, auxRod, toRod, fromRod)


def ToolChanger(tools, startIndex, target):
    if tools[startIndex] == target:
        return startIndex
    size = len(tools)
    l, r = startIndex-1, startIndex+1
    while (l % size) != startIndex:
        if tools[l % size] == target:
            return abs(l - startIndex)
        if tools[r % size] == target:
            return abs(r - startIndex)
        l, r = l-1, r+1
    return -1


def HackerCards(collection, d):
    result = []
    collection = sorted(collection)
    previous = 0
    for i in range(len(collection)):
        diff = collection[i] - previous
        if diff > 1:
            for j in range(1, diff):
                new_card = previous + j
                result.append(new_card)
                d -= new_card
                if d <= new_card:
                    return result    
        previous = collection[i] 
    next_card = max(collection[-1], result[-1] if len(result) > 0 else -1) + 1
    while d >= next_card:
        result.append(next_card)
        next_card, d = next_card + 1, d - next_card
    return result


def GetMinimumDifferenceBetweenAnagrams(a, b):
    result = []
    for a_word, b_word in zip(a, b):
        if len(a_word) != len(b_word):
            result.append(-1)
        else:
            num_changes = 0
            counter = dict()
            for char in a_word:
                counter[char] = 1 if (char not in counter) else (counter[char] + 1)
            for char in b_word:
                if char in counter:
                    if counter[char] == 1:
                        del counter[char]
                    else:
                        counter[char] -= 1
                else:
                    num_changes += 1
            result.append(num_changes)
    return result


def FindBeforeMatrix(after):
    m = len(after)
    n = len(after[0])
    before = [[] for _ in range(m)]
    value = 0
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                value = after[0][0]
            elif i == 0:
                value = after[0][j] - after[0][j-1]
            elif j == 0:
                value = after[i][0] - after[i-1][0]
            else:
                value = after[i][j] + after[i-1][j-1] - after[i][j-1] - after[i-1][j]   
            before[i].append(value)
    return before


def PredictMissingPrices(knownTimestamps, pricesAtKnownTimeStamps, unknownTimestamps):
    all_prices = []
    for i, tstamp in enumerate(knownTimestamps):
        all_prices.append( (tstamp, pricesAtKnownTimeStamps[i]) )
    for i in unknownTimestamps:
        all_prices.append( (i, None) )
    all_prices = sorted(all_prices, key = lambda x : x[0])
    prices_missing_indices = set([i for i in range(len(all_prices)) if all_prices[i][1] is None])

    prices_refined = []
    for _, price in all_prices:
        prices_refined.append(np.nan if price is None else price)
    print(prices_refined)
    s = pd.Series(prices_refined)
    s = s.interpolate()
    
    revised = [val for idx, val in enumerate(s) if (idx in prices_missing_indices)]
    return revised


def DroppedRequests(requestTime):
    
    # helper function to avoid code duplication
    def handleDrop(current_request, dropped_tracker, dropped_count, i):
        if current_request not in dropped_tracker or dropped_tracker[current_request] != i:
                dropped_tracker[current_request] = i
                return dropped_count + 1
        return dropped_count 
                
    dropped_count = 0
    dropped_tracker = dict()

    # use the fact of the array being sorted to check
    for i in range(len(requestTime)):
        current_request = requestTime[i]
        if i >= 3 and current_request == requestTime[i-3]:
            dropped_count = handleDrop(current_request, dropped_tracker, dropped_count, i)
        if i >= 20 and current_request - requestTime[i-20] < 10:
            dropped_count = handleDrop(current_request, dropped_tracker, dropped_count, i)
        if i >= 60 and current_request - requestTime[i-60] < 60:
            dropped_count= handleDrop(current_request, dropped_tracker, dropped_count, i)            
    return dropped_count


def ProcessLogs(logs, threshold):

    def recordTransacton(transaction_tracker, user_id):
        if user_id in transaction_tracker:
            transaction_tracker[user_id] += 1
        else:
            transaction_tracker[user_id] = 1   

    transaction_tracker = dict()
    for log in logs:
        log = log.split(" ")
        sender, receiver = log[0], log[1]
        if sender == receiver:
            recordTransacton(transaction_tracker, sender)
        else:
            recordTransacton(transaction_tracker, sender)
            recordTransacton(transaction_tracker, receiver)
    # optimal solution might be inserting using array bisection algorithm, python has bisect
    # but it doesnt support insertion while specifying a key such as int ...
    result = [user_id for user_id, transaction_count in transaction_tracker.items() if transaction_count >= threshold]
    return sorted(result, key=int)

class RecentCounter:

    def __init__(self):
        self.__tracker = collections.deque()
        
    def ping(self, t: int) -> int:
        self.__tracker.appendleft(t)
        while abs(t - self.__tracker[-1]) > 3000:
            self.__tracker.pop()
        return len(self.__tracker)

        
def FinalInstances(instances, averageUtil):
    index = 0
    system_sleep = False 
    while index < len(averageUtil):
        rate = averageUtil[index]
        print("Index:",index, "Instances:",instances, "Rate:",rate)
        if rate > 60:
            doubled_instances = instances * 2
            if doubled_instances <= 2 * (10**8):
                instances = doubled_instances
                system_sleep = True
        elif rate < 25 and instances > 1:
            instances = math.ceil(instances / 2)
            system_sleep = True
        
        # Adjust index value
        index = index + 11 if system_sleep else index + 1
        system_sleep = False 
    
    return instances


def BreakPalindrome(palindromeStr):
    size = len(palindromeStr)
    if size > 1:
        returnStr = ""
        for idx, char in enumerate(palindromeStr):
            if char.isalnum() and char > "a" and size // 2 != idx:
                return returnStr + "a" + palindromeStr[idx+1:]
            else:
                returnStr += char
    return "IMPOSSIBLE"      
    

def MaxPerformances(arrivals, durations):
    idx = list(range(len(arrivals)))
    finish_times = [arrivals[i] + durations[i] for i in range(len(durations))]
    idx.sort(key = lambda x: finish_times[x])
    s = set()
    previous_finish = 0
    for i in idx:
        if arrivals[i] >= previous_finish:
            s.add(i)
            previous_finish = finish_times[i]
    return len(s)


def CalculateCost(arr):
    minimum_value = 0
    while len(arr) >= 2:
        minimum = 9999999
        index = -1
        for i in range(1, len(arr)):
            subp = arr[i] * arr[i-1]
            if subp < minimum:
                minimum = subp # dp soln
                index = i
        minimum_value += minimum
        middle = max(arr[index-1], arr[index])
        arr = arr[:index-1] + [middle] + arr[index+1:] # resize
    return minimum_value


def BestTrio(friends_from, friends_to):
    trios = dict()
    graph = dict()
    for from_, to_ in zip(friends_from, friends_to):
        if from_ not in graph:
            graph[from_] = [to_]
        else:
            graph[from_].append(to_)
        if to_ not in graph:
            graph[to_] = [from_]
        else:
            graph[to_].append(from_)
            
    for f1, f1_friends in graph.items():
        if len(f1_friends) > 2:
            for f2 in f1_friends:
                f2_friends = graph[f2]
                for f3 in f2_friends:
                    f3_friends = graph[f3]
                    trio_key = tuple( sorted([f1, f2, f3]) ) 
                    if trio_key in trios:
                        continue
                    if f3 != f1 and f1 in f3_friends and f2 in f3_friends:
                        trios[trio_key] = len(f1_friends)-2 + len(f2_friends)-2 + len(f3_friends)-2
        
    values = trios.values()
    if len(values) > 0:
        return min(values)
    return -1

    
def PopularPlace(data): # [(d, p, t), ...]
    counter = dict()
    for _, place, _ in data:
        if place in counter:
            counter[place] += 1
        else:
            counter[place] = 0
    place = None
    max_count = float('-inf')
    for place_id, count in counter:
        if count > max_count:
            place = place_id
            max_count = count
    return place


def DistinctVisits(data):
    place_count = dict()
    device_at_place_tracker = dict()
    for device_id, place_id, timestamp in data:
        if place_id in device_at_place_tracker:
            devices = device_at_place_tracker[place_id]
            device_is_present = device_id in devices
            if device_is_present and timestamp - devices[device_id] > 10000:
                place_count[place_id] += 1
            elif device_is_present: 
                devices[device_id] = timestamp 
            else:
                device_at_place_tracker[place_id][device_id] = timestamp
        else:
            place_count[place_id] = 1
            device_at_place_tracker[place_id] = {device_id : timestamp}


def FindShortestSubArray(nums: List[int]) -> int:
    max_so_far = float('-inf')
    tracker = dict()

    for idx, num in enumerate(nums):
        if num in tracker:
            count, start, end = tracker[num]
            tracker[num] = (count + 1, start, idx + 1)
            current_max = count + 1
        else:
            tracker[num] = (1, idx, idx + 1) 
            current_max = 1
        max_so_far = max(max_so_far, current_max)
    
    start, end = 0, len(nums)
    for _, v in tracker.items():
        count, element_start, element_end = v
        if count == max_so_far:
            diff = element_end - element_start
            if diff < end - start:
                start, end = element_start, element_end
        
    return end - start


def MaxInversions(arr):
    inversions = 0
    size = len(arr)
    for i in range(1, size-1): # treat each element as the middle
        smaller, larger = 0, 0
        for j in range(i+1, size): 
            if arr[j] < arr[i]: 
                smaller += 1
        for j in range(i): 
            if arr[j] > arr[i]: 
                larger += 1
        inversions += smaller * larger
    return inversions   


#######################################################################################
###   RANDOM INTERVIEW QUESTION
#######################################################################################


import json
import requests
import sys 
import mailgunDatapoints as dp

# the api key, auth tuple, base url
MAILGUN_API_KEY = 'XXX'
AUTHORIZATION   = ("api", MAILGUN_API_KEY)
BASE_URL        = "https://api.mailgun.net/v3/"

# enum for requests
class RequestType:
    get    = "GET"
    post   = "POST"
    delete = "DELETE"


# check response and propagate exception if needed
def handle_request(url, request_type=RequestType.get, params=None, propagate=True):
    response, status_code = None, None
    try:
        if request_type == RequestType.get:
            response = requests.get(url=url, auth=AUTHORIZATION, params=params)
        elif request_type == RequestType.post:
            response = requests.post(url=url, auth=AUTHORIZATION, params=params)
        else: 
            response = requests.delete(url=url, auth=AUTHORIZATION, params=params)
        response.raise_for_status()
    except Exception as e:
        print(e)
        if response:
            status_code = response.status_code
        if propagate:
            raise e
    return (response, status_code)


# helper function to get mailgun domain
def acquire_domain():
    print("Beginning to determine domain...")
    url = BASE_URL + "domains"
    response = requests.get(url=url, auth=("api", MAILGUN_API_KEY))
    try:
        response.raise_for_status()
        data = json.loads(response.text)
        domain = data["items"][0]["name"]
        print("Parsed", domain)
        return domain
    except Exception as e:
        print(e)
        raise e

# constant for mailgun domain
DOMAIN = acquire_domain()
print("\n\n\n")


# get all the mailing lists the user belongs to
def access(identifier):
    url = BASE_URL + "lists/pages"
    response, _ = handle_request(url)
    all_mailing_lists = [i["address"] for i in json.loads(response.text)["items"]]
    mailing_lists_with_user = []
    # all_mailing_lists = all_mailing_lists[30:]
    for mailing_list in all_mailing_lists:
        url = BASE_URL + "lists/{}/members/{}".format(mailing_list, identifier)
        response, _ = handle_request(url, propagate=False)
        if "not found" in response.text: 
            print("Email {} not a member of {}".format(identifier, mailing_list))
        else: 
            mailing_lists_with_user.append(mailing_list)

    # unsure what difference between data/context should be in runIntegration.py
    # returning both for now, that way mailing lists can be passed to erasure
    return {"data" : mailing_lists_with_user, "context": mailing_lists_with_user} 


# create a mailing address with a given address
def create_mailing_list(address):
    url = BASE_URL + "lists"
    payload = { "address" : "{}@{}".format(address, DOMAIN) }
    response, _ = handle_request(url, request_type=RequestType.post, params=payload)
    print(response.text)


# remove the user from all mailing lists
def erasure(identifier, context):
    print("Preparing to remove {} from {} email lists".format(identifier, len(context)))
    deleted_count = 0
    for mailing_list in context:
        url = BASE_URL + "lists/{}/members/{}".format(mailing_list, identifier)
        response, _ = handle_request(url, request_type=RequestType.delete, propagate=False)
        if "member has been deleted" in response.text:
            deleted_count += 1
        else:
            print("Error deleting {} from {}... {}".format(identifier, mailing_list, response.text))
    print("Successfully deleted email {}/{} mailing lists...".format(deleted_count, len(context)))


# create mailing lists and seed users into them
def seed(identifier):
    raise NotImplementedError("Seed not implemented!")


# Modify this list to add the identifiers you want to use.
sample_identifiers_list = [
    'spongebob@transcend.io',
    'squidward@transcend.io',
    'patrick_star@transcend.io',
    'sandy_cheeks@transcend.io'
]

class ActionType:
    # Fetch data for a given identifier
    # from the remote system, e.g. Mailgun.
    Access = 'ACCESS'
    # Delete data for a given identifier
    # from the remote system.
    Erasure = 'ERASURE'
    # Seed data into the remote system
    # creatine a profile with the given identifier.
    Seed = 'SEED'

def verify_action_args(args):
    """
    Validate arguments.
    """
    valid_actions = [ActionType.Seed, ActionType.Erasure, ActionType.Access]
    if len(args) != 2:
        raise ValueError('This module accepts a single argument: python3 runIntegration.py <action>, where <action> can be one of: {}'.format(", ".join(valid_actions)))
    action = args[1]
    if action not in valid_actions:
        raise ValueError("Action argument must be one of {}".format(", ".join(valid_actions)))
    return action


def run_integration(identifier, action_type):
    """
    Run the ACCESS and/or ERASURE flows for the given identifier.
    """
    print('Running access...\n')
    access_result = dp.access(identifier)
    data = access_result['data']
    print('Data retrieved for ' + identifier + ':')
    print(json.dumps(data, indent=2))

    if action_type == ActionType.Access:
        return

    context = access_result['context']
    print('Context for the erasure: ', context)
    print('\nRunning erasure...')
    dp.erasure(identifier, context)
    print('All done!')


def main():
    action = verify_action_args(sys.argv)
    data = sample_identifiers_list

    # Run the functions for all the identifiers we want to test
    for identifier in data:
        if action == ActionType.Seed:
            dp.seed(identifier)
        elif action == ActionType.Access or action == ActionType.Erasure:
            run_integration(identifier, action)
    return

if __name__ == "__main__":
    main()

#######################################################################################
###   RANDOM INTERVIEW QUESTION
#######################################################################################

# Given a list of `events` that consists of a <name, action, start_time, end_time> and an 
# integer k. Find the first time in the day between "00:00" and "23:59", when all people
# mentioned in at least one event are avilable for a meeting of k minutes. Return the answer
# in the "HH:MM" format and return "-5" if it's not possible.

# Convert "HH:MM" to an integer of minutes for ease of comparison
def convertToMinutes(time):
    hour, minute = time.split(":")
    return int(hour)*60 + int(minute)

def getMeetingTime(events, k):
    # Map the event interval to the person (remember intervals here are inclusive)
    interval_map = collections.defaultdict(list)
    for event in events:
        person_name, action, start, end = event.split()
        interval_map[person_name].append((start, end))
    # Sort the event intervals and merge them in case there are any that intersect
    for person_name, interval in interval_map.items():
        sorted_interval_time_range = sorted((convertToMinutes(start), convertToMinutes(end)) for start, end in interval)
        merged_intervals = []
        for start, end in sorted_interval_time_range:
            if not merged_intervals or merged_intervals[-1][1] < start:
                merged_intervals.append([start, end])
            else:
                # We sorted by starting range, so it's possible the current interval we are processing is within 
                # the time range of the last interval added to `merged_intervals`
                merged_intervals[-1][1] = max(merged_intervals[-1][1], end)
        interval_map[person_name] = merged_intervals
    print(f"Sorted interval map: {interval_map}")
    # Locate the available free intervals
    day_start = 0        # 00:00
    day_end = 24*60 - 1  # 23:59
    free_times = []
    for person_name, interval in interval_map.items():
        person_free = []
        prev_end = day_start - 1
        for start, end in interval:
            free_start = prev_end + 1
            free_end = start - 1
            if free_end - free_start + 1 >= k:
                person_free.append((free_start, free_end))
            prev_end = end
        # Check from last interval to the end of the day
        free_start = prev_end + 1
        free_end = day_end
        if free_end - free_start + 1 >= k:
            person_free.append((free_start, free_end))
        free_times.append(person_free)
    print(f"Free times: {free_times}")
    shared_free_time = free_times[0] # Start with the first person’s free times
    for person in free_times[1:]:    # Intersect with each subsequent person
        result = []
        i, j = 0, 0
        while i < len(shared_free_time) and j < len(person):
            start = max(shared_free_time[i][0], person[j][0])
            end = min(shared_free_time[i][1], person[j][1])
            if end - start >= k:
                result.append((start, end))
            if shared_free_time[i][1] < person[j][1]:
                i += 1
            else:
                j += 1
        shared_free_time = result
        if not shared_free_time:
            break
    # Compute the intersection
    if shared_free_time:
        minutes = shared_free_time[0][0]
        return f"{minutes // 60:02d}:{minutes % 60:02d}"
    else:
        return "-5"

#######################################################################################
###   RANDOM INTERVIEW QUESTIONS
#######################################################################################

def oddNumbers(l, r):
    if l > r:
        return
    offset = 1 if l % 2 == 0 else 0
    return [odd_number for odd_number in range(l+offset, r+1, 2)]

# O(n) space complexity
# O(max_neighbor-min_neighbor)
def smoothArray(A):
    while True:
        edited = False
        new_scores = A.copy()
        for i in range(1, len(A) - 1):
            if A[i] < A[i-1] and A[i] < A[i+1]:
                new_scores[i] = A[i] + 1
                edited = True
            elif A[i] > A[i-1] and A[i] > A[i+1]:
                new_scores[i] = A[i] - 1
                edited = True
        A = new_scores
        if not edited:
            break
    return A

def stepCountToNearestFibonacci(x):
    """
    Given a number x return the number of steps it takes to move to 
    the nearest Fibonacci number if you can only add or subtract one
    each move.
    """
    a, b = 0, 1
    while b < x:
        a, b = b, a + b
    return min(b-x, x-a)

def computeNumberOfItemsToFillLeftoverSpace(A):
    """
    First element of the array is the total weight of items in a box.
    The maximum storage for the box is 5000.
    If A[1:] has random items with weights, return the most number of
    items we can fit.
    """
    result = 0
    existing_weight = A[0]
    if existing_weight >= 5000 or len(A) == 1:
        return result
    # Adjust the first value to a maximum value and heapify the input array
    A[0] = float('inf')
    heapq.heapify(A)
    remaining_weight = 5000 - k
    for _ in range(len(A) - 1):
        # We can't take the item, break here
        if remaining_weight - A[0] < 0:
            break
        remaining_weight -= heapq.heappop(A)
        result += 1
    return result

def encodeHexSpeak(S):
    """
    Given a base-10 number as a string, from 1 to 1,000,000,000,000.
    Convert it to hexadecimal and display its HexSpeak, if valid, 
    otherwise return "ERROR"
    """
    hexspeak_mapping = {
        "0": "O",
        "1": "I",
        "A": "A",
        "B": "B",
        "C": "C",
        "D": "D",
        "E": "E",
        "F": "F"
    }
    hexadecimal_str = hex(int(S))[2:].upper() # omit 0x prefix
    try:
        "".join(hexspeak_mapping[c] for c in hexadecimal_str)
    except KeyError:
        return "ERROR"

def getNumberOfUniqueSubarraysOfDistinctElements(S):
    """
    Given a string S, return an integer that represents the number of
    ways we can select a non-empty substring of S in which all of the
    characters are identical.

    e.g. "zzzyz" == 8 bc 4x "z", 1x "y", 2x "zz", 1x "zzz"

    We solve it in O(n) time and space complexity using a 2-pointer approach
    using n * (n + 1) // 2 to find total number of subarrays which is also
    the sum of natural numbers...
    """
    i = j = result = 0
    while j < len(S):
        while j < len(S) and S[j] == S[i]:
            j += 1
        diff = j - i
        result += diff * (diff+1) // 2
        i = j
    return result    


##################################################
##################################################
##################################################

def findKthLargestInFile(file_path, k):
    """
    Best for small files!
    TC: O(nlogn)
    SC: O(n)
    """
    if k < 1:
        raise ValueError(f"Illegal argument: k because {k} < 1")
    with open(file_path, "r") as file:
        values = [int(line.strip()) for line in file.readlines()]
        values.sort()
    if k > len(values):
        raise ValueError(f"Illegal argument: k because {k} > number of digits in the file")
    return values[-k]

def findKthLargestInFileBetter(file_path, k):
    """
    Best for larger files and a smaller value of k!
    TC: O(nlogk)
    SC: O(k)
    """
    if k < 1:
        raise ValueError(f"Illegal argument: k because {k} < 1")
    heap = []
    with open(file_path, "r") as file:
        for line in file:
            heapq.heappush(heap, int(line.strip())
            if len(heap) > k:
                heapq.heappop(heap)
    if k > len(heap):
        raise ValueError(f"Illegal argument: k because {k} > number of values in the file")
    return values[0] # smallest in the heap, k-th largest overall
