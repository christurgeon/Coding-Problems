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

print(Wildcards(input()))