CREATE TABLE IF NOT EXISTS replay_events (
    event_id text PRIMARY KEY,
    user_session text NOT NULL,
    source_event_time timestamp NOT NULL,
    replay_time timestamp NOT NULL,
    event_type text NOT NULL,
    product_id text NOT NULL,
    user_id text NOT NULL,
    category_id text NOT NULL,
    category_code text NULL,
    brand text NULL,
    price double precision NULL,
    source text NOT NULL,
    created_at timestamp NOT NULL DEFAULT current_timestamp
);
