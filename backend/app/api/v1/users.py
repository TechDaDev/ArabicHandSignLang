from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from app.api.deps import DbSession, get_current_active_user
from app.models.user import User
from app.schemas.user import UserPublic, UserUpdateRequest


router = APIRouter(prefix="/users", tags=["users"])


@router.patch("/me", response_model=UserPublic)
def update_current_user(
    payload: UserUpdateRequest,
    db: DbSession,
    current_user: User = Depends(get_current_active_user),
) -> User:
    """Update the authenticated user's editable profile fields."""
    updates = payload.model_dump(exclude_unset=True)

    new_username = updates.get("username")
    if "username" in updates and new_username != current_user.username and new_username is not None:
        username_taken = db.scalar(select(User).where(User.username == new_username, User.id != current_user.id))
        if username_taken is not None:
            raise HTTPException(status_code=409, detail="Username is already in use")

    for field_name in ("username", "full_name"):
        if field_name in updates:
            setattr(current_user, field_name, updates[field_name])

    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    return current_user


@router.get("/me", response_model=UserPublic)
def read_current_user(current_user: User = Depends(get_current_active_user)) -> User:
    """Return the current authenticated user profile."""
    return current_user
