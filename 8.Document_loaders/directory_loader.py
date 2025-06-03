from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = '8.Document_loaders\directory',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()

print('Docs: ', docs)

print('1st Doc: ', docs[0].page_content)
print('1st Doc metadta: ', docs[0].metadata)