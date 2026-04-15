import os, warnings
warnings.filterwarnings("ignore")
os.chdir(r"c:\Java yazılanlar\rag-tubitak-project")
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyPDFLoader, silent_errors=True, show_progress=False)
docs = loader.load()

with open("chunk_results.txt", "w") as f:
    f.write(f"Sayfa: {len(docs)}\n")
    for cs, co in [(4000,200), (5000,200), (6000,200), (8000,200)]:
        ts = RecursiveCharacterTextSplitter(chunk_size=cs, chunk_overlap=co)
        c = ts.split_documents(docs)
        f.write(f"  {cs}/{co} -> {len(c)} chunk\n")
    f.write("DONE\n")
