"""Small helpers for Sprint 2a gold feature computation."""


def compute_cart_to_view_ratio(total_views: int, total_carts: int) -> float:
    if total_views == 0:
        return 0.0
    return total_carts / total_views


def normalize_category_value(category_code: str | None, category_id: str) -> str:
    return category_code if category_code is not None else category_id
