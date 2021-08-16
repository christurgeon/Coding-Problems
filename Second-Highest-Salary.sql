/* Write your T-SQL query statement below */
SELECT MAX(Salary) AS SecondHighestSalary 
FROM Employee t
WHERE t.Salary NOT IN (
    SELECT MAX(Salary) FROM Employee
)
