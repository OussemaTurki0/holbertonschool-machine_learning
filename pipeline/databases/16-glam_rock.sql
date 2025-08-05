-- List bands with Glam rock as main style, ranked by lifespan until 2020
SELECT band_name,
       CASE
         WHEN split IS NULL OR split > 2020 THEN 2020
         ELSE split
       END - formed AS lifespan
FROM metal_bands
WHERE main_style = 'Glam rock'
ORDER BY lifespan DESC;
