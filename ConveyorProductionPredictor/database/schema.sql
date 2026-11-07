-- ============================================================
-- Conveyor Belt Production Predictor
-- Base de données MySQL pour le suivi et la prédiction industrielle
-- Serveur cible : localhost:3306 | Utilisateur : root
-- ============================================================

CREATE DATABASE IF NOT EXISTS conveyor_production_predictor_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE conveyor_production_predictor_db;

-- Relevés opérationnels importés depuis les fichiers CSV
CREATE TABLE IF NOT EXISTS production_records (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    record_time DATETIME NOT NULL COMMENT 'Date et heure du relevé',
    belt VARCHAR(80) NOT NULL COMMENT 'Identifiant du convoyeur',
    conveyor_speed DECIMAL(6,2) NOT NULL COMMENT 'Vitesse de la bande (m/s)',
    active_trucks SMALLINT UNSIGNED NOT NULL COMMENT 'Nombre de camions actifs',
    humidity DECIMAL(5,2) NOT NULL COMMENT 'Humidité (%)',
    temperature DECIMAL(5,2) NOT NULL COMMENT 'Température (°C)',
    operating_hours DECIMAL(5,2) NOT NULL COMMENT 'Heures de fonctionnement',
    shift_name VARCHAR(30) NOT NULL COMMENT 'Équipe de travail',
    production DECIMAL(10,2) NOT NULL COMMENT 'Production observée (t/h)',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_production_record_time (record_time),
    INDEX idx_production_belt (belt),
    INDEX idx_production_shift (shift_name)
) ENGINE=InnoDB COMMENT='Historique des mesures de production des convoyeurs';

-- Résultats calculés depuis le formulaire de prédiction
CREATE TABLE IF NOT EXISTS prediction_history (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    predicted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Date et heure de la prédiction',
    belt VARCHAR(80) NOT NULL,
    conveyor_speed DECIMAL(6,2) NOT NULL COMMENT 'Vitesse de la bande (m/s)',
    active_trucks SMALLINT UNSIGNED NOT NULL,
    humidity DECIMAL(5,2) NOT NULL COMMENT 'Humidité (%)',
    temperature DECIMAL(5,2) NOT NULL COMMENT 'Température (°C)',
    operating_hours DECIMAL(5,2) NOT NULL COMMENT 'Heures de fonctionnement',
    shift_name VARCHAR(30) NOT NULL,
    prediction DECIMAL(10,2) NOT NULL COMMENT 'Production prédite (t/h)',
    INDEX idx_prediction_date (predicted_at),
    INDEX idx_prediction_belt (belt)
) ENGINE=InnoDB COMMENT='Historique des prédictions réalisées par l application';

-- Contrôle rapide après exécution :
-- SHOW TABLES;
-- DESCRIBE production_records;
-- DESCRIBE prediction_history;
