from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('8.Document_loaders\dl-curriculum.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=90,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_documents(docs)

print(result[0].page_content)