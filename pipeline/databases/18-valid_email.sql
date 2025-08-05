-- Trigger to reset valid_email to 0 only when email is updated
DELIMITER //
CREATE TRIGGER reset_valid_email_on_email_change
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
    IF NOT (OLD.email <=> NEW.email) THEN
        SET NEW.valid_email = 0;
    END IF;
END;
//
DELIMITER ;
