class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        
        
        def validPlacement(row, col, value):
            # check row and column
            for i in range(9):
                if board[row][i] == value or board[i][col] == value:
                    return False
            # check square
            row_bound = (row // 3) * 3
            col_bound = (col // 3) * 3
            for i in range(3):
                for j in range(3):
                    if board[i+row_bound][j+col_bound] == value:
                        return False 
            return True
        
        
        def solve(row, col):
            # start a new col
            if col == 9:
                col = 0
                row += 1
                
                # we have filled all rows
                if row == 9:
                    return True
                
            if board[row][col] != '.':
                return solve(row, col+1)
            
            # if we have space available and we can place
            for value in range(1, 10):
                if validPlacement(row, col, str(value)):
                    board[row][col] = str(value)
                    if solve(row, col+1):
                        return True
            
            # backtracking
            board[row][col] = '.'
            return False
        
        solve(0, 0)
                
        