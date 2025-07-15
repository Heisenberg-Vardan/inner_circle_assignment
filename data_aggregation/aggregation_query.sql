SELECT
    DATE(i.timestamp) AS interaction_date,
    u.gender,
    u.city,
    COUNT(i.user_id) AS total_interactions,
    (SUM(CASE WHEN i.like_type IN (1, 2) THEN 1 ELSE 0 END) * 100.0) / COUNT(i.user_id) AS percentage_likes,
    (SUM(CASE WHEN i.like_type = 0 THEN 1 ELSE 0 END) * 100.0) / COUNT(i.user_id) AS percentage_dislikes
FROM
    interactions i
JOIN
    users u ON i.user_id = u.user_id
GROUP BY
    interaction_date,
    u.gender,
    u.city
ORDER BY
    interaction_date,
    u.city;