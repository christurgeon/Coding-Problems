class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        prefix = [""]
        for index, char in enumerate(strs[0]):
            for j in range(1, len(strs)):
                if index == len(strs[j]) or char != strs[j][index]:
                    return "".join(prefix)
            prefix.append(char)
        return "".join(prefix)

        