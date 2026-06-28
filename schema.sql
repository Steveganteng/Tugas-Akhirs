-- Skema MySQL untuk aplikasi rekomendasi restok obat (database: TugasAkhir)
-- Catatan: `python seed.py` membuat tabel via SQLAlchemy create_all();
-- file ini disediakan untuk setup manual / dokumentasi.

CREATE DATABASE IF NOT EXISTS TugasAkhir
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE TugasAkhir;

CREATE TABLE IF NOT EXISTS users (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  username      VARCHAR(64) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  nama          VARCHAR(128),
  role          VARCHAR(32) DEFAULT 'apoteker',
  is_active     BOOLEAN DEFAULT TRUE,
  created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
  last_login    DATETIME NULL,
  INDEX (username), INDEX (role)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS upload_log (
  upload_id        INT AUTO_INCREMENT PRIMARY KEY,
  filename         VARCHAR(255),
  n_baris          INT DEFAULT 0,
  n_valid          INT DEFAULT 0,
  n_ditolak        INT DEFAULT 0,
  status           ENUM('diproses','selesai','gagal') DEFAULT 'diproses',
  error_text       TEXT,
  model_dilatih    VARCHAR(64),
  candidate_version INT,
  uploaded_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX (status)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS transaksi_raw (
  id               BIGINT AUTO_INCREMENT PRIMARY KEY,
  tanggal_masuk    DATE,
  register         VARCHAR(64),
  kode_diagnosa    VARCHAR(64),
  diagnosa_primer  VARCHAR(255),
  resep_obat       VARCHAR(255),
  jumlah           FLOAT,
  sisa_stok        FLOAT,
  satuan           VARCHAR(32),
  upload_id        INT,
  created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX (resep_obat), INDEX (upload_id),
  FOREIGN KEY (upload_id) REFERENCES upload_log(upload_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS panel_bulanan (
  id           BIGINT AUTO_INCREMENT PRIMARY KEY,
  obat         VARCHAR(255) NOT NULL,
  periode      VARCHAR(7) NOT NULL,
  demand       FLOAT DEFAULT 0,
  stok         FLOAT DEFAULT 0,
  satuan       VARCHAR(32),
  is_observasi BOOLEAN DEFAULT TRUE,
  sumber       VARCHAR(32) DEFAULT 'init',
  UNIQUE KEY uq_obat_periode (obat, periode),
  INDEX (obat), INDEX (periode)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS rekomendasi (
  id                 BIGINT AUTO_INCREMENT PRIMARY KEY,
  nama_obat          VARCHAR(255) NOT NULL,
  periode            VARCHAR(32),
  prediksi_demand    FLOAT,
  rop                FLOAT,
  safety_stock       FLOAT,
  jumlah_rekomendasi FLOAT,
  stok_saat_ini      FLOAT,
  status             VARCHAR(32),
  segmen             VARCHAR(64),
  cluster            INT,
  model_version      INT,
  generated_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX (nama_obat), INDEX (status), INDEX (model_version)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS model_registry (
  version_id           INT AUTO_INCREMENT PRIMARY KEY,
  nama_model           VARCHAR(64),
  path_artefak         VARCHAR(512),
  metrics_json         TEXT,
  trained_at           DATETIME DEFAULT CURRENT_TIMESTAMP,
  trained_on_upload_id INT,
  status               ENUM('active','candidate','rejected','archived') DEFAULT 'candidate',
  catatan              VARCHAR(255),
  INDEX (status)
) ENGINE=InnoDB;
