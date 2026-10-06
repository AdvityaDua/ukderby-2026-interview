from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from app.core.config import settings

engine = create_engine(settings.DATABASE_URL.replace("postgres://", "postgresql://"), pool_pre_ping=True, pool_recycle=300)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Company(Base):
    __tablename__ = 'companies'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    logo_icon = Column(String, nullable=True)
    description = Column(String, nullable=True)
    github_links = relationship("GithubLink", back_populates="company")

class Topic(Base):
    __tablename__ = 'topics'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    logo_icon = Column(String, nullable=True)
    description = Column(String, nullable=True)
    github_links = relationship("GithubLink", back_populates="topic")

class GithubLink(Base):
    __tablename__ = 'github_links'
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, unique=True, index=True)
    company_id = Column(Integer, ForeignKey('companies.id'), nullable=True)
    topic_id = Column(Integer, ForeignKey('topics.id'), nullable=True)

    company = relationship("Company", back_populates="github_links")
    topic = relationship("Topic", back_populates="github_links")

from sqlalchemy import JSON, DateTime
import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    interviews = relationship("Interview", back_populates="user")

class Interview(Base):
    __tablename__ = "interviews"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    company = Column(String)
    role = Column(String)
    status = Column(String, default="completed") # processing, completed, error
    feedback_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="interviews")

# Create tables
Base.metadata.create_all(bind=engine)
