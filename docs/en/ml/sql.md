# 🗃️ SQL: reason about rows before memorizing syntax

SQL expresses operations over relational data. Good queries begin by defining one row, its key, and the intended output grain. A query can be syntactically correct while double-counting revenue after a many-to-many join.

## 🧭 The main operations

`SELECT` retrieves data; `INSERT`, `UPDATE`, and `DELETE` change rows. `CREATE`, `ALTER`, and `DROP` define objects. `GRANT`/`REVOKE` manage privileges, while `COMMIT`, `ROLLBACK`, and `SAVEPOINT` manage transactions. DDL/DML/DCL/TCL are useful teaching categories; naming conventions and transaction behavior vary by database.

For query reasoning, think `FROM/JOIN → WHERE → GROUP BY → HAVING → window calculations → SELECT/DISTINCT → ORDER BY → limit`. This is a conceptual processing order, not a promise about the optimizer's physical execution plan.

## 🧩 Filtering, aggregation, and nulls

```sql
SELECT department_id, COUNT(*) AS employees, AVG(salary) AS mean_salary
FROM employees
WHERE salary > 3000
GROUP BY department_id
HAVING COUNT(*) > 5;
```

`WHERE` filters individual rows before grouping; `HAVING` filters groups. `COUNT(*)` counts rows, while `COUNT(salary)` counts non-null salary values. `SUM`, `AVG`, `MIN`, and `MAX` generally ignore null inputs. An all-null group and an empty result need explicit interpretation.

Use `IS NULL`, not `= NULL`. SQL comparisons can evaluate to unknown, and `WHERE` retains only true rows. `COALESCE` selects the first non-null value; choosing zero as a replacement is a business assumption. `CASE WHEN` expresses conditional values, for example `SUM(CASE WHEN status = 'paid' THEN amount ELSE 0 END)`.

## 📌 Joins and missing matches

An inner join retains matches; a left join retains every left row; right and full joins preserve the corresponding unmatched sides. If one left key matches three right rows, it produces three output rows. A filter on a right-table column in `WHERE` can remove null-extended rows from a left join; put the appropriate match restriction in `ON` when unmatched left rows should remain.

To find employees without a matching department:

```sql
SELECT e.*
FROM employees AS e
WHERE NOT EXISTS (
    SELECT 1 FROM departments AS d WHERE d.id = e.department_id
);
```

This includes employees with a null department ID. Add `e.department_id IS NOT NULL` if those should be excluded. `NOT IN` can yield unknown if its subquery contains null, so it is not an interchangeable anti-join shortcut. See [subquery semantics](https://www.postgresql.org/docs/current/functions-subquery.html).

A self-join uses two aliases for one table: join `employees e` to `employees m` on `e.manager_id = m.id`. Use a left join if employees without managers should remain. `UNION` removes duplicate result rows; `UNION ALL` retains them and avoids that deduplication work. Neither guarantees order without `ORDER BY`.

## 🚀 Subqueries, CTEs, ranking, and ties

A scalar subquery can compare salary with `(SELECT AVG(salary) FROM employees)`. A CTE names a query result for readability and reuse within the statement; it is not automatically a stored table or an optimization barrier in every engine.

```sql
WITH ranked AS (
    SELECT employee_id, salary,
           DENSE_RANK() OVER (ORDER BY salary DESC) AS salary_rank
    FROM employees
    WHERE salary IS NOT NULL
)
SELECT employee_id, salary FROM ranked WHERE salary_rank = 3;
```

This returns all employees with the third-highest distinct salary. With salaries 100, 100, 90, 80, `ROW_NUMBER` assigns distinct positions, `RANK` gives 1,1,3,4, and `DENSE_RANK` gives 1,1,2,3. To select exactly N rows, include a deterministic tie-breaker in `ORDER BY` and use dialect-specific `LIMIT`, `FETCH`, or `TOP`. The original correlated-subquery approach also works by counting distinct greater salaries, but assess its plan on large tables.

Windows keep row identities while computing over related rows. `PARTITION BY department_id` restarts a calculation per department. For running totals, specify a frame such as `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`; default frames can include tied peers. See [window functions](https://www.postgresql.org/docs/current/tutorial-window.html).

## 🔍 Dates, duplicates, and integrity

```sql
SELECT customer_id, COUNT(*) AS purchases
FROM orders
WHERE order_date >= '2025-09-01' AND order_date < '2025-10-01'
GROUP BY customer_id
HAVING COUNT(*) > 3;
```

This is a fixed September 2025 example, not a permanently correct definition of “last month.” Compute month boundaries in the intended business timezone. A half-open interval avoids guessing the final day's last representable timestamp.

Find repeated keys with `GROUP BY key HAVING COUNT(*) > 1`, then decide whether they are duplicates or legitimate events. A primary key uniquely identifies rows and is non-null; foreign keys enforce supported referential relationships. `NOT NULL`, `UNIQUE`, and `CHECK` constrain values. A default supplies an omitted value; it does not itself validate all supplied values.

Normalization reduces update anomalies: 1NF avoids repeating groups under the chosen relational representation; 2NF removes partial dependencies of non-key attributes on candidate keys; 3NF removes inappropriate transitive dependencies, with a formal definition in terms of functional dependencies. Denormalization trades redundant storage and maintenance for selected read paths. It requires an update-consistency strategy.

## 📊 Performance and transaction boundaries

Indexes can accelerate suitable filters, joins, and ordering, but cost storage and maintenance; a scan may be preferable for a large fraction of a table. Inspect query plans and actual cardinalities. An index does not repair a logically multiplying join.

`DELETE` removes selected rows, `TRUNCATE` empties tables, and `DROP` removes objects. Locking, triggers, identity handling, and rollback behavior depend on the engine. PostgreSQL supports transaction rollback of `TRUNCATE`; “TRUNCATE can never roll back” is incorrect. See [PostgreSQL TRUNCATE](https://www.postgresql.org/docs/current/sql-truncate.html) and [transactions](https://www.postgresql.org/docs/15/tutorial-transactions.html).

Exercise: create a tiny dataset with ties, nulls, duplicate join keys, and unmatched departments. Predict query results before execution. Those edge cases teach more than testing only clean, unique rows. Continue with [Pandas](pandas.md).
