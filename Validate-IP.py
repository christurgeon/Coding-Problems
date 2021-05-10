class Solution(object):
    def validIPAddress(self, IP):
        """
        :type IP: str
        :rtype: str
        """
        
        
        def IPv4(IP):
            def starts_with(s, match):
                return s.startswith(match)

            tokens = IP.split(".")
            if len(tokens) != 4:
                return "Neither"
            try:
                for token in tokens:
                    if starts_with(token, "0") and len(token) > 1:
                        return "Neither"
                    if not token.isdigit():
                        return "Neither"
                    res = int(token)
                    if res < 0 or res > 255:
                        return "Neither"
            except Exception as e:
                return "Neither"
            return "IPv4"
        
        
        def IPv6(IP):
            def good_hex(hex_):
                match = re.match("^[A-Fa-f0-9]+$", hex_)
                return True if match else False
                    
            tokens = IP.split(":")
            if len(tokens) != 8:
                return "Neither"
            for i in tokens:
                size = len(i)
                if size < 1 or size > 4:
                    return "Neither"
                if not good_hex(i):
                    return "Neither"
            return "IPv6"
        
        
        return_one = IPv4(IP)
        if return_one != "Neither":
            return return_one
        else:
            return IPv6(IP)
        
            
            
            
            