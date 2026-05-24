-- PharmaLink GCN - Database Setup Script for PostgreSQL
-- Run this in pgAdmin 4 or psql to set up the database

-- 1. Create database (run as postgres superuser)
-- CREATE DATABASE pharmalink;

-- 2. Connect to pharmalink database, then run:

-- The 'users' table will be auto-created by TypeORM (synchronize: true)
-- But here is the manual DDL for reference:

CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         VARCHAR(80)  UNIQUE NOT NULL,
    username      VARCHAR(50)  UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name     VARCHAR(120),
    role          VARCHAR(20)  NOT NULL DEFAULT 'user',
    is_active     BOOLEAN NOT NULL DEFAULT TRUE,
    avatar_url    TEXT,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- 3. Create default admin account (password: Admin@12345)
-- bcrypt hash of 'Admin@12345' with 12 rounds:
INSERT INTO users (email, username, password_hash, full_name, role, is_active)
VALUES (
    'admin@pharmalink.local',
    'admin',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/o.k2vXjCy',
    'System Administrator',
    'admin',
    TRUE
) ON CONFLICT DO NOTHING;

-- 4. Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_users_email    ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_role     ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_active   ON users(is_active);

-- Verify:
SELECT id, email, username, role, is_active, created_at FROM users;
