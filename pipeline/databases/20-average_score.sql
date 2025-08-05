DELIMITER //

CREATE PROCEDURE ComputeAverageScoreForUser(
    IN p_user_id INT
)
BEGIN
    DECLARE avgScore FLOAT;

    SELECT AVG(score) INTO avgScore
    FROM corrections
    WHERE user_id = p_user_id;

    UPDATE users SET average_score = IFNULL(avgScore, 0)
    WHERE id = p_user_id;
END;
//

DELIMITER ;
