import os
from rag_pipeline import HindiRAGSystem

PDF_FILE = "Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf"

test_queries = [
    {"id": 1, "query": "SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष?", "lang": "Hindi"},
    {"id": 2, "query": "कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?", "lang": "Hindi"},
    {"id": 3, "query": "पोखरण-II में कलाम की क्या भूमिका थी?", "lang": "Hindi"},
    {"id": 4, "query": "Which institution did Kalam attend to study aeronautical engineering?", "lang": "English"},
    {"id": 5, "query": "Who co-wrote the autobiography, and in what year was it published?", "lang": "English"},
    {"id": 6, "query": "How and where did Kalam die in 2015?", "lang": "English"}
]

def main():
    rag = HindiRAGSystem()
    rag.index_document(PDF_FILE)
    print("\n" + "="*70 + "\n")

    for item in test_queries:
        q = item["query"]
        docs = rag.retrieve(q, top_k=3)
        answer = rag.generate_answer(q, docs)
        top = docs[0]

        print(f"Query: {q}")
        print(f"Answer: {answer}")
        print(f'Source (citation): section: "{top["section"]}", chunk_id: {top["chunk_id"]}, page: {top["page"]}, score: {top["score"]}')
        print("-" * 70)

if __name__ == "__main__":
    main()