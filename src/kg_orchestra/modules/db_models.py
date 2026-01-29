from sqlalchemy import Column, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class Paragraph(Base):
    __tablename__ = "paragraphs"
    paragraph_id = Column(String, primary_key=True)
    pmcid = Column(String, nullable=False)
    paragraph_text = Column(String, nullable=False)

# SQLite setup
engine = create_engine("sqlite:////home/bio/groupshare/amohamed/workspace/alzminer/databases/ndd_paragraphs_db/ndd_para_paragraphs.db") # Change this path as needed.
Session = sessionmaker(bind=engine)

def create_tables():
    Base.metadata.create_all(engine)