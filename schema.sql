CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,

    google_sub TEXT UNIQUE NOT NULL,
    email TEXT,
    name TEXT DEFAULT '',

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);


CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users (id) ON DELETE CASCADE,

    title TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated
ON chat_sessions (user_id, updated_at DESC);


CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY,

    session_id UUID NOT NULL
        REFERENCES chat_sessions (id)
        ON DELETE CASCADE,

    role TEXT NOT NULL
        CHECK (role IN ('user', 'asssistant')),

    content TEXT NOT NULL,

    status TEXT NOT NULL DEFAULT 'completed'
        CHECK (status IN ('pending', 'completed', 'failed')),

    metadata JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
ON chat_messages (session_id, created_at ASC, id ASC);
