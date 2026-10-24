# 🗃️ SQL：先明确每行代表什么，再写语法

SQL 表达对关系数据的操作。好的查询应该首先定义一行、其键以及预期的输出粒度。一个查询可以语法正确，但在多对多连接后重复计算收入。

## 🧭 主要操作

`SELECT` 用于检索数据；`INSERT`、`UPDATE` 和 `DELETE` 用于更改行。`CREATE`、`ALTER` 和 `DROP` 用于定义对象。`GRANT`/`REVOKE` 管理权限，而 `COMMIT`、`ROLLBACK` 和 `SAVEPOINT` 管理事务。DDL/DML/DCL/TCL 是有用的教学类别；命名约定和事务行为因数据库而异。

分析查询时，思考 `FROM/JOIN → WHERE → GROUP BY → HAVING → 窗口计算 → SELECT/DISTINCT → ORDER BY → LIMIT`。这是理解查询的逻辑顺序，而不是对优化器物理执行计划的承诺。

## 🧩 过滤、聚合和空值

```sql
SELECT department_id, COUNT(*) AS employees, AVG(salary) AS mean_salary
FROM employees
WHERE salary > 3000
GROUP BY department_id
HAVING COUNT(*) > 5;
```

`WHERE` 在分组之前过滤单行；`HAVING` 过滤组。`COUNT(*)` 计算行数，而 `COUNT(salary)` 计算非空工资值。`SUM`、`AVG`、`MIN` 和 `MAX` 通常忽略空输入。全为空的组和空结果需要显式解释。

使用 `IS NULL`，而不是 `= NULL`。SQL 比较可以评估为未知，而 `WHERE` 只保留真行。`COALESCE` 选择第一个非空值；将零作为替换是一种业务假设。`CASE WHEN` 表达条件值，例如 `SUM(CASE WHEN status = 'paid' THEN amount ELSE 0 END)`。

## 📌 连接和未匹配记录

内连接保留匹配项；左连接保留所有左侧行；右连接和全连接保留相应的未匹配侧。如果一个左侧键匹配三个右侧行，则产生三行输出。在 `WHERE` 中对右侧表列的过滤可以移除左连接中的右侧为空的结果行；当需要保留未匹配的左侧行时，在 `ON` 子句中添加适当的匹配限制。

要查找没有匹配部门的员工：

```sql
SELECT e.*
FROM employees AS e
WHERE NOT EXISTS (
    SELECT 1 FROM departments AS d WHERE d.id = e.department_id
);
```

这包括具有空部门 ID 的员工。如果需要排除这些员工，则添加 `e.department_id IS NOT NULL`。`NOT IN` 在其子查询包含空值时可能会产生未知，因此它不是可互换的反连接快捷方式。请参阅 [子查询语义](https://www.postgresql.org/docs/current/functions-subquery.html)。

自连接使用一个表的两个别名：将 `employees e` 连接到 `employees m` 上 `e.manager_id = m.id`。如果需要保留没有经理的员工，则使用左连接。`UNION` 移除重复的结果行；`UNION ALL` 保留它们并避免去重工作。两者在没有 `ORDER BY` 的情况下不保证顺序。

## 🚀 子查询、CTE、排名和平局

标量子查询可以将工资与 `(SELECT AVG(salary) FROM employees)` 进行比较。CTE（公共表表达式）为可读性和在语句内重用命名查询结果；它不是每个引擎中的自动存储表或优化屏障。

```sql
WITH ranked AS (
    SELECT employee_id, salary,
           DENSE_RANK() OVER (ORDER BY salary DESC) AS salary_rank
    FROM employees
    WHERE salary IS NOT NULL
)
SELECT employee_id, salary FROM ranked WHERE salary_rank = 3;
```

这返回所有具有第三高不同工资的员工。对于工资为 100、100、90、80 的情况，`ROW_NUMBER` 分配不同的位置，`RANK` 给出 1,1,3,4，而 `DENSE_RANK` 给出 1,1,2,3。要选择恰好 N 行，在 `ORDER BY` 中包含确定性的并列值的确定性排序字段，并使用特定于方言的 `LIMIT`、`FETCH` 或 `TOP`。原始相关子查询方法也通过计算不同的大于工资来工作，但在大型表上评估其计划。

窗口保持行标识符的同时对相关行组成的窗口进行计算。`PARTITION BY department_id` 每个部门重新启动计算。对于运行总计，指定一个框架，如 `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`；默认框架可以包括排序键相同的行。请参阅 [窗口函数](https://www.postgresql.org/docs/current/tutorial-window.html)。

## 🔍 日期、重复项和完整性

```sql
SELECT customer_id, COUNT(*) AS purchases
FROM orders
WHERE order_date >= '2025-09-01' AND order_date < '2025-10-01'
GROUP BY customer_id
HAVING COUNT(*) > 3;
```

这是一个固定的 2025 年 9 月示例，而不是永久正确的“上个月”的定义。按业务采用的时区计算月份边界。半开区间避免猜测最后一天可表示的时间戳。

使用 `GROUP BY key HAVING COUNT(*) > 1` 找到重复键，然后决定它们是重复项还是合法事件。主键唯一标识行且非空；外键强制支持的引用关系。`NOT NULL`、`UNIQUE` 和 `CHECK` 约束值。默认值提供省略的值；它本身不验证所有提供的值。

规范化减少更新异常：1NF 避免在所选的关系表示下重复组；2NF 要求非主属性不对候选键的真子集产生部分依赖；3NF 对非主属性通过其他非键属性形成的传递依赖作进一步约束。严格判断应以函数依赖和候选键的正式定义为准。反规范化通过冗余存储和维护换取选定的读路径。它需要一个更新一致性策略。

## 📊 性能和事务边界

索引可以加速合适的过滤、连接和排序，但会消耗存储和维护；对于表的一部分扫描可能更优。检查查询计划和实际基数。索引不能修复因多对多匹配而使行数膨胀的连接。

`DELETE` 移除选定行，`TRUNCATE` 清空表，`DROP` 移除对象。锁定、触发器、标识处理和回滚行为取决于引擎。PostgreSQL 支持 `TRUNCATE` 的事务回滚；“TRUNCATE 从不回滚”是错误的。请参阅 [PostgreSQL TRUNCATE](https://www.postgresql.org/docs/current/sql-truncate.html) 和 [事务](https://www.postgresql.org/docs/15/tutorial-transactions.html)。

练习：创建一个包含平局、空值、重复连接键和未匹配部门的小数据集。在执行前预测查询结果。这些边缘情况比仅测试干净且唯一的行教得更多。继续使用 [Pandas](pandas.md)。
