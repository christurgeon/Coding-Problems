/* Write your T-SQL query statement below */
UPDATE Salary
SET sex = CASE WHEN sex = 'f' THEN 'm' ELSE 'f' END
