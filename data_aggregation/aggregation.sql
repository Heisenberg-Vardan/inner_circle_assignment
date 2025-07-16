-- Aggregates percentage of likes, dislikes, and total interactions per day, gender, and city of the sender
SELECT
    DATE(i.timestamp) AS day,
    u.gender AS gender,
    u.city AS city,
    COUNT(*) AS total_interactions,
    ROUND(100.0 * SUM(CASE WHEN i.like_type IN (1,2) THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_likes,
    ROUND(100.0 * SUM(CASE WHEN i.like_type = 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_dislikes
FROM
    interactions i
JOIN
    users u
ON
    i.user_id = u.user_id
GROUP BY
    day, gender, city
ORDER BY
    day, gender, city;
