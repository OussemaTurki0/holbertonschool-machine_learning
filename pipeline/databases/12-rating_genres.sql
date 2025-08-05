-- List genres by sum of ratings from shows linked to them, sorted descending
SELECT tv_genres.name, SUM(tv_show_ratings.rating) AS rating
FROM tv_genres
JOIN tv_show_genres ON tv_genres.id = tv_show_genres.genre_id
JOIN tv_show_ratings ON tv_show_genres.tv_show_id = tv_show_ratings.tv_show_id
GROUP BY tv_genres.name
ORDER BY rating DESC;
