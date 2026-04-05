import os
from rag_utility import process_document_to_chroma_db, answer_question

# FOR TESTING PURPOSE 
# CHEKCING WHETHER FUNCTIONS WORK PROPERLY OR NOT

def main():
    if not os.path.exists("doc_vectorstore"):
        process_document_to_chroma_db("doc/Pollution_Presentation.pdf")
    
    response = answer_question("What is pollution?")
    print(response)

if __name__ == "__main__":
    main()