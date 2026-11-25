-- Runs once, when the dev volume is first created (compose.dev.yml).
-- The opt-in database tests empty this database; never point them at "aity".
CREATE DATABASE aity_test OWNER aity;
