from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from app.models.db import SessionLocal, Company, Topic, GithubLink, Base, engine

Base.metadata.create_all(bind=engine)

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ItemCreate(BaseModel):
    name: str

class GithubLinkCreate(BaseModel):
    url: str
    company_id: Optional[int] = None
    topic_id: Optional[int] = None

@router.get("/companies")
def get_companies(db: Session = Depends(get_db)):
    return db.query(Company).all()

@router.post("/companies")
def create_company(item: ItemCreate, db: Session = Depends(get_db)):
    db_item = Company(name=item.name)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

@router.get("/topics")
def get_topics(db: Session = Depends(get_db)):
    return db.query(Topic).all()

@router.post("/topics")
def create_topic(item: ItemCreate, db: Session = Depends(get_db)):
    db_item = Topic(name=item.name)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

@router.get("/github-links")
def get_github_links(db: Session = Depends(get_db)):
    return db.query(GithubLink).all()

@router.post("/github-links")
def create_github_link(item: GithubLinkCreate, db: Session = Depends(get_db)):
    if not item.company_id and not item.topic_id:
        raise HTTPException(status_code=400, detail="Must provide company_id or topic_id")
    db_item = GithubLink(url=item.url, company_id=item.company_id, topic_id=item.topic_id)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item
