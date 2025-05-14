# scripts/build_index.py
import csv, pickle, faiss, numpy as np
import logging 
import sys 

# --- logging conf---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__) 

def main():
    from src.settings import settings
    from src.db.base import Base, engine, SessionLocal
    from src.db.models import Document as DbDocument
    from src.db.crud import add_documents
    from src.adapters.embeddings.sentence_transformers import SentenceTransformerEmbedder

    logger.info("Initializing database and creating tables if they don't exist...")
    Base.metadata.create_all(bind=engine)
    
    session = SessionLocal()
    
    logger.info(f"Loading documents from CSV: {settings.faq_csv}")
    with open(settings.faq_csv, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        texts = []
        if settings.csv_has_header: 
            try:
                next(reader)
                logger.info("CSV header skipped.")
            except StopIteration:
                logger.warning("CSV seems to be empty or only has a header.")
        
        for i, row in enumerate(reader):
            if len(row) >= 2: 
                content = f"{row[0]} {row[1]}" # Q + A 
                texts.append(content)
            else:
                logger.warning(f"Skipping malformed row {i+1} (o {i+2} if header) in CSV: {row}")


    if texts:
        logger.info(f"Adding {len(texts)} documents to the database...")
        from src.utils import preprocess_text
        processed_texts = [preprocess_text(t) for t in texts]
        add_documents(session, processed_texts)
        logger.info(f"Successfully inserted {len(texts)} documents.")
    else:
        logger.warning("No texts extracted from CSV to add to the database.")


    if settings.retrieval_mode == "dense":
        logger.info("Dense retrieval mode enabled. Building FAISS index...")
        docs_from_db = session.query(DbDocument.id, DbDocument.content).order_by(DbDocument.id).all()
        if not docs_from_db:
            logger.error("No documents found in database to build FAISS index. Aborting FAISS build.")
        else:
            db_ids = [doc.id for doc in docs_from_db]
            db_contents = [doc.content for doc in docs_from_db]

            logger.info(f"Generating embeddings for {len(db_contents)} documents...")
            embedder = SentenceTransformerEmbedder()
            vectors = np.array([embedder.embed(t) for t in db_contents]).astype("float32")
            
            # IndexFlatIP for interal product (adequate for normalized embeddings)
            index_dim = vectors.shape[1]
            index = faiss.IndexFlatIP(index_dim) 
            index.add(vectors)
            
            faiss.write_index(index, settings.index_path)
            logger.info(f"FAISS index written to {settings.index_path}")
            
            with open(settings.id_map_path, "wb") as fh_pickle:
                pickle.dump(db_ids, fh_pickle) # Save real IDs into db
            logger.info(f"FAISS ID map written to {settings.id_map_path}")
    
    session.close()
    logger.info("Build index process completed.")

if __name__ == "__main__":
    main()