"""
TÜBİTAK RAG Projesi - Ana Uygulama
Groq API ile Belge Tabanlı Soru-Cevap Sistemi
"""

import streamlit as st
import os
import time
import json
import re
import datetime
from dotenv import load_dotenv
from pathlib import Path

# Çevre değişkenlerini yükle (.env dosyasından)
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Sayfa yapılandırması
st.set_page_config(
    page_title="Nexus AI - TÜBİTAK Bilgi Asistanı",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS ile profesyonel stil (Nexus AI - Modern Dark Mode)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Ana Arka Plan */
    .stApp {
        background: linear-gradient(135deg, #0F1419 0%, #1a1f2e 100%);
        color: #E3E3E3;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F1419 0%, #1a1f2e 100%);
        border-right: 1px solid #2d3748;
    }
    
    /* Sidebar Başlık */
    [data-testid="stSidebar"] h1 {
        color: #0ea5e9 !important;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 1.8em !important;
        margin-bottom: 2rem;
    }
    
    /* Başlıklar */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
    }
    
    /* Metinler */
    p, span, div, label {
        color: #cbd5e0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Mesaj Kutuları - Kullanıcı */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        border: 1px solid #334155;
        margin: 12px 0;
        padding: 16px;
    }
    
    /* Mesaj Kutuları - Asistan */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background: linear-gradient(135deg, #0ea5e933 0%, #3b82f633 100%);
        border: 1px solid #0ea5e944;
        border-radius: 16px;
        margin: 12px 0;
        padding: 16px;
    }
    
    /* Kaynak Kartları */
    .source-card {
        background: linear-gradient(135deg, #1e2139 0%, #1a2d42 100%);
        border: 1px solid #2d3748;
        border-left: 4px solid #0ea5e9;
        border-radius: 12px;
        padding: 15px;
        margin-top: 10px;
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }
    .source-card:hover {
        border-left-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.2);
    }
    .source-header {
        color: #0ea5e9 !important;
        font-weight: 600;
        font-size: 0.95em;
        margin-bottom: 8px;
    }
    .transparency-score {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        margin-left: 10px;
    }
    .score-high { background-color: #0ea5e944; color: #7dd3fc !important; }
    .score-med { background-color: #f59e0b44; color: #fcd34d !important; }
    .score-low { background-color: #ef444444; color: #fca5a5 !important; }
    
    /* Butonlar */
    .stButton button {
        background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%) !important;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.8rem;
        transition: all 0.3s ease;
        font-weight: 600;
        color: #ffffff !important;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(14, 165, 233, 0.4);
    }
    .stButton button p {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Bilgi Kutucukları */
    .stAlert {
        background: linear-gradient(135deg, #1e2139 0%, #1a2d42 100%) !important;
        border: 1px solid #2d3748;
        border-left: 4px solid #0ea5e9;
        color: #e2e8f0 !important;
        border-radius: 8px;
    }
    .stAlert p {
        color: #cbd5e0 !important;
    }
    
    /* Input Alanı */
    .stTextInput input {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px;
        padding: 12px 16px;
        transition: all 0.2s ease;
    }
    .stTextInput input:focus {
        border-color: #0ea5e9 !important;
        box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.2) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e2139 !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
        border: 1px solid #334155 !important;
        transition: all 0.2s ease;
    }
    .streamlit-expanderHeader:hover {
        background-color: #2d3748 !important;
    }
    
    /* Select Box */
    .stSelectbox select {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
    }
    
    /* Main Content Area */
    .main {
        padding: 2rem;
    }
    
    /* Chat Input Container */
    .stChatInputContainer {
        background: transparent !important;
        border-top: 1px solid #2d3748;
        padding: 1.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Header ile başlık
st.markdown("<h1 style='font-size: 2.5em; margin: 0; color: #0ea5e9; display: flex; align-items: center;'>⚡ &nbsp;&nbsp;ZeKamu</h1>", unsafe_allow_html=True)
st.divider()


class RAGSystem:
    """RAG Sistemi - Belge yükleme, vektör oluşturma ve sorgulama"""
    
    def __init__(self, data_folder="data", vector_db_path="vectorstore"):
        self.data_folder = data_folder
        self.vector_db_path = vector_db_path
        self.vectorstore = None
        self.rag_chain = None
        self.retriever = None
        
        # API Key kontrolü
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable bulunamadı! Lütfen .env dosyasını kontrol edin.")
        
        # Embeddings - LOKAL model (bedava, sınırsız, Türkçe destekli)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # LLM - Groq API (hızlı ve uygun fiyatlı)
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            groq_api_key=self.api_key,
            temperature=0.3
        )
    
    def load_documents(self):
        """Data klasöründeki tüm PDF dosyalarını yükle"""
        
        # Data klasörü kontrolü
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
            return []
        
        # PDF dosyalarını tek tek yükle (hata toleransı için)
        documents = []
        pdf_files = list(Path(self.data_folder).rglob("*.pdf"))
        
        if not pdf_files:
            return []
        
        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_path))
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                st.warning(f"⚠️ {pdf_path.name} yüklenemedi: {e}")
                continue
        
        return documents
    
    def split_documents(self, documents):
        """Belgeleri küçük parçalara böl (chunking)"""
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,   # Büyük chunk → daha az embedding isteği → rate limit sorunu azalır
            chunk_overlap=300,  # Yeterli bağlam korunur
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        return chunks

    def create_vectorstore(self, chunks):
        """Vektör veritabanı oluştur - lokal embedding, rate limit yok"""
        
        total = len(chunks)
        st.info(f"📊 {total} parça lokal olarak embed ediliyor (rate limit yok)...")
        
        try:
            self.vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            self.vectorstore.save_local(self.vector_db_path)
            st.success(f"✅ {total} parça başarıyla embed edildi ve kaydedildi!")
        except Exception as e:
            raise Exception(f"Embedding hatası: {str(e)}")
        
        return self.vectorstore
    
    def load_vectorstore(self):
        """Kaydedilmiş vektör veritabanını yükle"""
        
        index_file = os.path.join(self.vector_db_path, "index.faiss")
        if os.path.exists(index_file):
            try:
                self.vectorstore = FAISS.load_local(
                    self.vector_db_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return True
            except Exception as e:
                print(f"Vektör DB yükleme hatası: {e}")
                return False
        return False
    
    def create_qa_chain(self):
        """Soru-Cevap zinciri oluştur (LCEL)"""
        
        if not self.vectorstore:
            raise ValueError("Vektör veritabanı yüklenmemiş!")
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Prompt şablonu
        prompt = ChatPromptTemplate.from_template(
            """Sen Türkiye'deki kamu mali yönetimi ve denetim konularında uzman bir asistansın.
Aşağıda sana verilen belge parçalarını dikkatlice oku ve soruyu bu bilgilere dayanarak yanıtla.
Cevabın belgelerden çıkarılabilecek bilgilere dayanmalıdır.
Eğer sorunun cevabı verilen belgelerde hiç geçmiyorsa, bunu belirt.

Belge Parçaları:
{context}

Soru: {question}

Yanıt:"""
        )
        
        # LCEL zinciri
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return self.rag_chain
    
    def initialize(self):
        """Sistemi başlat - belgeler yoksa yükle, varsa hazır"""
        
        # Vektör DB klasörü yoksa oluştur
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        # Tamamlanmış vektör DB var mı?
        if self.load_vectorstore():
            self.create_qa_chain()
            return "Sistem hazır (Mevcut vektör veritabanı yüklendi)"
        
        # Yoksa yeni oluştur
        documents = self.load_documents()
        
        if len(documents) == 0:
            return f"UYARI: {self.data_folder}/ klasöründe PDF dosyası bulunamadı!"
        
        st.info(f"📚 {len(documents)} sayfa yüklendi, parçalanıyor...")
        chunks = self.split_documents(documents)
        st.info(f"🔗 {len(chunks)} parça oluşturuldu, embedding başlıyor...")
        self.create_vectorstore(chunks)
        self.create_qa_chain()
        
        return f"Sistem hazır ({len(documents)} belge, {len(chunks)} parça işlendi)"
    
    def calculate_transparency_score(self, doc_content):
        """
        Gelişmiş Şeffaflık Puanı Hesaplama Algoritması
        4 Temel Kriter: Erişilebilirlik, Hesap Verebilirlik, Güncellik, Tutarlılık
        """
        
        scores = {
            "accessibility": 0,
            "accountability": 0,
            "recency": 0,
            "consistency": 0
        }
        
        # 1. ERİŞİLEBİLİRLİK (Accessibility) - %25
        # Metin yoğunluğu ve yapısal bütünlük kontrolü
        char_count = len(doc_content)
        if char_count > 500:
            # Çok kısa metinler (OCR hatası veya boş sayfa) düşük puan alır
            scores["accessibility"] = 25
        elif char_count > 200:
            scores["accessibility"] = 15
        else:
            scores["accessibility"] = 5
            
        # 2. HESAP VEREBİLİRLİK (Accountability) - %35
        # Sayısal veri, para birimi ve denetim terimlerinin yoğunluğu
        # Sayısal veri kontrolü (Regex ile sayıları bul)
        numeric_density = len(re.findall(r'\d+', doc_content)) / (len(doc_content.split()) + 1)
        
        # Anahtar kelimeler
        accountability_keywords = [
            "bütçe", "gider", "gelir", "harcama", "denetim", "faaliyet", 
            "performans", "hedef", "gerçekleşme", "sapma", "tl", "tutar", "%"
        ]
        keyword_count = sum(1 for k in accountability_keywords if k in doc_content.lower())
        
        if numeric_density > 0.05 and keyword_count > 3: # %5'ten fazla sayısal veri ve en az 3 anahtar kelime
            scores["accountability"] = 35
        elif numeric_density > 0.02 or keyword_count > 1:
            scores["accountability"] = 20
        else:
            scores["accountability"] = 5
            
        # 3. GÜNCELLİK (Recency) - %20
        # Metin içindeki yıl bilgilerini kontrol et
        current_year = datetime.datetime.now().year
        years_found = re.findall(r'20\d{2}', doc_content)
        
        if years_found:
            # En güncel yılı bul
            latest_year_in_doc = max([int(y) for y in years_found])
            year_diff = current_year - latest_year_in_doc
            
            if year_diff <= 1: # Bu yıl veya geçen yıl
                scores["recency"] = 20
            elif year_diff <= 3: # Son 3 yıl
                scores["recency"] = 10
            else: # Eski veri
                scores["recency"] = 5
        else:
            scores["recency"] = 0 # Tarih yoksa düşük puan
            
        # 4. TUTARLILIK (Consistency) - %20
        # "Hedef" ve "Sonuç" kavramlarının birlikte geçmesi
        has_target = any(x in doc_content.lower() for x in ["hedef", "amaç", "plan", "öngörü"])
        has_result = any(x in doc_content.lower() for x in ["sonuç", "gerçekleşme", "tamamlanma", "çıktı"])
        
        if has_target and has_result:
            scores["consistency"] = 20
        elif has_target or has_result:
            scores["consistency"] = 10
        else:
            scores["consistency"] = 5
            
        total_score = sum(scores.values())
        return total_score, scores

    def query(self, question):
        """Soru sor ve cevap al"""
        
        if not self.rag_chain:
            return {"answer": "Sistem henüz hazır değil!", "source_documents": []}
        
        try:
            # Kaynak belgeleri al
            source_docs = self.retriever.invoke(question)
            
            # Cevabı üret
            answer = self.rag_chain.invoke(question)
            
            # DEBUG: Eğer cevap bulunamadıysa, bağlamı kontrol et
            if not source_docs:
                answer = "Belgelerde bu konuyla ilgili parça bulunamadı. Lütfen farklı kelimelerle tekrar deneyin."
            
            # Kaynak belgeler için şeffaflık analizi
            processed_sources = []
            for doc in source_docs:
                total_score, details = self.calculate_transparency_score(doc.page_content)
                processed_sources.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Bilinmeyen Belge"),
                    "page": doc.metadata.get("page", 0),
                    "score": total_score,
                    "details": details
                })
            
            return {
                "answer": answer,
                "source_documents": processed_sources
            }
        except Exception as e:
            error_msg = str(e)
            if any(x in error_msg for x in ["429", "RESOURCE_EXHAUSTED", "quota"]):
                user_msg = "API istek limiti aşıldı. Lütfen birkaç dakika bekleyip tekrar deneyin."
            elif "API_KEY" in error_msg or "401" in error_msg or "403" in error_msg:
                user_msg = "API anahtarı geçersiz veya eksik. Lütfen .env dosyanızı kontrol edin."
            else:
                user_msg = f"Bir hata oluştu: {error_msg}"
            return {
                "answer": user_msg,
                "source_documents": []
            }


# Streamlit Session State başlatma
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.system_status = "Başlatılmadı"
    st.session_state.messages = []

# Sidebar - Sistem Durumu
with st.sidebar:
    st.markdown("<h1 style='color: #0ea5e9; margin: 0; font-size: 1.8rem;'>Belge Asistanı</h1>", unsafe_allow_html=True)
    st.divider()
    
    st.subheader("Sistem Durumu")
    status_placeholder = st.empty()
    
    # Sistem başlatma butonu
    if st.button("Sistemi Başlat", type="primary", use_container_width=True):
        with st.spinner("Başlatılıyor..."):
            try:
                rag = RAGSystem(data_folder="data", vector_db_path="vectorstore")
                status_message = rag.initialize()
                st.session_state.rag_system = rag
                st.session_state.system_status = status_message
                st.success(status_message)
            except Exception as e:
                st.error(f"Hata: {str(e)}")
                st.session_state.system_status = f"Hata: {str(e)}"
    
    # Durum gösterimi
    if st.session_state.system_status.startswith("Sistem hazır"):
        status_placeholder.success(st.session_state.system_status)
    elif st.session_state.system_status.startswith("UYARI"):
        status_placeholder.warning(st.session_state.system_status)
    elif st.session_state.system_status.startswith("Hata"):
        status_placeholder.error(st.session_state.system_status)
    else:
        status_placeholder.info(st.session_state.system_status)
    
    st.markdown("---")
    
    # Kullanım bilgisi
    st.subheader("Kullanım")
    st.markdown("""
    1. `data/` klasörüne PDF dosyalarınızı ekleyin
    2. "Sistemi Başlat" butonuna tıklayın
    3. Sorularınızı sorun
    """)
    
    st.markdown("---")
    st.caption("Powered by Google Gemini")

# Chat mesajlarını göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcı girişi
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    
    # Sistem kontrolü
    if not st.session_state.rag_system:
        st.error("Lütfen önce sistemi başlatın!")
    else:
        # Kullanıcı mesajını ekle
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Cevap üret
        with st.chat_message("assistant"):
            with st.spinner("Yanıt oluşturuluyor..."):
                try:
                    response = st.session_state.rag_system.query(prompt)
                    answer = response["answer"]
                    
                    st.markdown(answer)
                    
                    # Kaynak belgeleri göster (Gelişmiş)
                    with st.expander("Kaynaklar ve Şeffaflık Analizi"):
                        for i, doc in enumerate(response["source_documents"]):
                            score = doc['score']
                            details = doc['details']
                            score_class = "score-high" if score >= 75 else "score-med" if score >= 50 else "score-low"
                            
                            st.markdown(f"""
                            <div class="source-card">
                                <div class="source-header">
                                    KAYNAK {i+1}: {os.path.basename(doc['source'])} (Sayfa {doc['page']})
                                    <span class="transparency-score {score_class}">Şeffaflık: {score}/100</span>
                                </div>
                                <div style="font-size: 0.8em; color: #C4C7C5; margin-bottom: 10px;">
                                    <b>Detaylar:</b> 
                                    Erişilebilirlik: {details['accessibility']}/25 | 
                                    Hesap Verebilirlik: {details['accountability']}/35 | 
                                    Güncellik: {details['recency']}/20 | 
                                    Tutarlılık: {details['consistency']}/20
                                </div>
                                <div style="font-size: 0.9em; color: #C4C7C5;">
                                    {doc['content'][:300]}...
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Asistan cevabını kaydet
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                    
                except Exception as e:
                    st.error(f"Hata oluştu: {str(e)}")

# Sohbeti temizle butonu
if len(st.session_state.messages) > 0:
    if st.button("Sohbeti Temizle"):
        st.session_state.messages = []
        st.rerun()
