class Solution:
  def isPalindromic(self, s: str): 
    half = len(s) // 2
    for i in range(half):
      if s[i] != s[~i]:
        return False
    return True
