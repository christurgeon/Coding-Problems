class Solution:
  
  def intToString(self, x: int):
    negative = False
    if x < 0:
      x, negative = -x, True
     
    s = []
    while True:
      s.append(char(ord('0') + x % 10))
      x = x // 10
      if x == 0:
        break
        
    # add negative sign back if needed
    return ('-' if negative else '') + ''.join(reversed(s))
