DELIMITER //

CREATE PROCEDURE AddBonus(
    IN p_user_id INT,
    IN p_project_name VARCHAR(255),
    IN p_score INT
)
BEGIN
    DECLARE projectId INT;

    -- Check if project exists
    SELECT id INTO projectId FROM projects WHERE name = p_project_name LIMIT 1;

    -- If project doesn't exist, insert it and get the id
    IF projectId IS NULL THEN
        INSERT INTO projects (name) VALUES (p_project_name);
        SET projectId = LAST_INSERT_ID();
    END IF;

    -- Insert correction
    INSERT INTO corrections (user_id, project_id, score) VALUES (p_user_id, projectId, p_score);
END;
//

DELIMITER ;
