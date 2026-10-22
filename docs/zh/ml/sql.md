!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# SQL

### 🧩 **1. Core SQL Concepts**

- **SQL Definition**: Structured Query Language, used to interact with relational databases.
- **Common Commands (CRUD)**:
    - `SELECT` – retrieve data
    - `INSERT` – add data
    - `UPDATE` – modify data
    - `DELETE` – remove data
- **Joins**:
    - `INNER JOIN` → only matching records
    - `LEFT JOIN` → all from left, matching from right
    - `RIGHT JOIN` → all from right, matching from left
    - `FULL JOIN` → all records from both

### 📊 **2. Key Interview Topics**

- **WHERE vs HAVING**:

    `WHERE` filters rows *before aggregation*, `HAVING` filters *after aggregation*.

- **GROUP BY and Aggregates**:

    `GROUP BY` is used with functions like `COUNT()`, `SUM()`, `AVG()`, `MAX()`, `MIN()`.

- **Subqueries**:

    Queries inside queries — used in `SELECT`, `FROM`, or `WHERE`.

- **Window Functions**:

    `ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`, `OVER(PARTITION BY … ORDER BY …)`

    → useful for ranking, moving averages, cumulative sums.

- **CASE WHEN**: conditional logic within SQL.
- **NULL Handling**:

    `IS NULL`, `COALESCE()`, `IFNULL()` — replaces nulls with defaults.


# FQA

### Q2: What are the main types of SQL commands?

- **DDL (Data Definition Language):** `CREATE`, `ALTER`, `DROP`, `TRUNCATE`
- **DML (Data Manipulation Language):** `SELECT`, `INSERT`, `UPDATE`, `DELETE`
- **DCL (Data Control Language):** `GRANT`, `REVOKE`
- **TCL (Transaction Control Language):** `COMMIT`, `ROLLBACK`, `SAVEPOINT`

### Q3: What is the difference between `WHERE` and `HAVING`?

- `WHERE` filters rows **before aggregation**.
- `HAVING` filters **after aggregation** (with `GROUP BY`).

```sql
SELECT department, COUNT(*)
FROM employees
WHERE salary > 3000
GROUP BY department
HAVING COUNT(*) > 5;
```

### Q4: What is the difference between `INNER JOIN` and `OUTER JOIN`?

- `INNER JOIN`: returns only matching rows.
- `LEFT JOIN`: all rows from left + matching from right.
- `RIGHT JOIN`: all rows from right + matching from left.
- `FULL OUTER JOIN`: all rows from both sides.

### Q5: What are aggregate functions?

`COUNT()`, `SUM()`, `AVG()`, `MAX()`, `MIN()` — perform calculations over groups of rows.

### Q6: What is the difference between `UNION` and `UNION ALL`?

- `UNION` removes duplicates.
- `UNION ALL` keeps duplicates and is faster.

### Q7: What is a subquery?

A query nested inside another query.

Used in `SELECT`, `FROM`, or `WHERE`:

```sql
SELECT name
FROM employee
WHERE salary > (SELECT AVG(salary) FROM employee);
```

### Q8: What is a CTE (Common Table Expression)?

A temporary named result set that can be referenced within a query.

```sql
WITH high_salary AS (
  SELECT name, salary FROM employee WHERE salary > 5000
)
SELECT * FROM high_salary WHERE name LIKE 'A%';
```

### Q9: What are indexes?

Indexes speed up read operations but slow down writes.

They store references to data like a book’s index for faster lookup.

### Q10: What are primary and foreign keys?

- **Primary key:** uniquely identifies a row in a table.
- **Foreign key:** links one table to another (enforces referential integrity).

### Q11: How do you find duplicate records?

```sql
SELECT name, COUNT(*)
FROM employee
GROUP BY name
HAVING COUNT(*) > 1;
```

### Q12: How do you find the nth highest salary?

```sql
SELECT DISTINCT salary
FROM employee e1
WHERE (SELECT COUNT(DISTINCT salary) FROM employee e2 WHERE e2.salary > e1.salary) = n - 1;
```

### Q13: How do you select top N records?

```sql
SELECT * FROM employee ORDER BY salary DESC LIMIT 5;   -- MySQL / PostgreSQL
```

or

```sql
SELECT TOP 5 * FROM employee ORDER BY salary DESC;     -- SQL Server
```

### Q14: How do you find employees not in a department?

```sql
SELECT *
FROM employees
WHERE department_id NOT IN (SELECT id FROM departments);
```

### Q15: How to find customers who made more than 3 purchases last month?

```sql
SELECT customer_id, COUNT(*) AS orders
FROM orders
WHERE order_date >= '2025-09-01' AND order_date < '2025-10-01'
GROUP BY customer_id
HAVING COUNT(*) > 3;
```

### Q16: What are window functions?

They perform calculations across rows related to the current row.

```sql
SELECT
  employee_id,
  salary,
  RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dept_rank
FROM employees;
```

**Common window functions:**

- `ROW_NUMBER()` – unique rank
- `RANK()` – allows gaps
- `DENSE_RANK()` – no gaps
- `SUM() OVER()` – running total
- `AVG() OVER()` – moving average

### Q17: Difference between `GROUP BY` and `OVER()`?

`GROUP BY` collapses rows into one per group.

`OVER()` keeps original rows and adds aggregated info.

### Q18: What are constraints?

Rules applied to columns:

- `NOT NULL` – must have a value
- `UNIQUE` – values must be unique
- `CHECK` – must meet a condition
- `DEFAULT` – assigns default value
- `PRIMARY KEY` – unique + not null
- `FOREIGN KEY` – references another table

### Q19: What is normalization?

The process of structuring data to reduce redundancy and improve integrity.

- **1NF:** no repeating groups
- **2NF:** all non-key attributes depend on primary key
- **3NF:** no transitive dependency

### Q20: What is denormalization?

Combining tables to improve read performance (at the cost of redundancy).

### Q24: Difference between `DELETE`, `TRUNCATE`, and `DROP`?

| Command | Removes Data | Removes Table | Rollback | Speed |
| --- | --- | --- | --- | --- |
| DELETE | ✅ | ❌ | ✅ | Slow |
| TRUNCATE | ✅ | ❌ | ❌ | Fast |
| DROP | ✅ | ✅ | ❌ | Fastest |

### Q29: What is a self-join?

A table joined with itself — useful for hierarchical data (e.g., employees and managers).

```sql
SELECT e.name, m.name AS manager
FROM employee e
JOIN employee m ON e.manager_id = m.id;
```
