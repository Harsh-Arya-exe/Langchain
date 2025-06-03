from langchain_community.document_loaders import CSVLoader

loader = CSVLoader('8.Document_loaders\Social_Network_Ads.csv')

docs = loader.load()

print("\nLength: ", len(docs))
print("\n1st row: ", docs[0].page_content)