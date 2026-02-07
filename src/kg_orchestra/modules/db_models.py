from sqlalchemy import Column, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class Paragraph(Base):
    __tablename__ = "paragraphs"
    paragraph_id = Column(String, primary_key=True)
    pmcid = Column(String, nullable=False)
    paragraph_text = Column(String, nullable=False)

# SQLite setup
db_name = input("Please provide a name for your sql database without spaces: ") or "ndd_paragraphs"
db_path = input ("Please provide a path for the database folder: ") or "db"
engine = create_engine(f"sqlite:///{db_path}/{db_name}.db") # Change this path as needed.
Session = sessionmaker(bind=engine)

def create_tables():
    Base.metadata.create_all(engine)
