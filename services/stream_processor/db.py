"""PostgreSQL replay event append store."""

from __future__ import annotations

from typing import Any


class ReplayEventStore:
    def __init__(self, connection):
        self.connection = connection

    def append(self, event: dict[str, Any]) -> None:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO replay_events (
                        event_id,
                        user_session,
                        source_event_time,
                        replay_time,
                        event_type,
                        product_id,
                        user_id,
                        category_id,
                        category_code,
                        brand,
                        price,
                        source
                    )
                    VALUES (
                        %(event_id)s,
                        %(user_session)s,
                        %(source_event_time)s,
                        %(replay_time)s,
                        %(event_type)s,
                        %(product_id)s,
                        %(user_id)s,
                        %(category_id)s,
                        %(category_code)s,
                        %(brand)s,
                        %(price)s,
                        %(source)s
                    )
                    ON CONFLICT (event_id) DO NOTHING
                    """,
                    event,
                )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
