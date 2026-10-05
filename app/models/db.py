from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from app.core.config import settings

engine = create_engine(settings.DATABASE_URL.replace("postgres://", "postgresql://"))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Company(Base):
    __tablename__ = 'companies'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    github_links = relationship("GithubLink", back_populates="company")

class Topic(Base):
    __tablename__ = 'topics'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    github_links = relationship("GithubLink", back_populates="topic")

class GithubLink(Base):
    __tablename__ = 'github_links'
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, unique=True, index=True)
    company_id = Column(Integer, ForeignKey('companies.id'), nullable=True)
    topic_id = Column(Integer, ForeignKey('topics.id'), nullable=True)

    company = relationship("Company", back_populates="github_links")
    topic = relationship("Topic", back_populates="github_links")
