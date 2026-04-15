"""
TÜBİTAK RAG Projesi - Ana Uygulama
Google Gemini API ile Belge Tabanlı Soru-Cevap Sistemi
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

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Sayfa yapılandırması
st.set_page_config(
    page_title="TÜBİTAK RAG Sistemi",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS ile özel stil (Gemini Dark Mode - Gelişmiş)
st.markdown("""
    <style>
    /* Ana Arka Plan */
    .stApp {
        background-color: #131314;
        color: #E3E3E3;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1E1F20;
        border-right: 1px solid #444746;
    }
    
    /* Başlıklar */
    h1, h2, h3 {
        color: #E3E3E3 !important;
        font-family: 'Google Sans', sans-serif;
        font-weight: 500;
    }
    
    /* Metinler */
    p, span, div, label {
        color: #C4C7C5;
        font-family: 'Roboto', sans-serif;
    }
    
    /* Mesaj Kutuları - Kullanıcı */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #282A2C;
        border-radius: 20px;
        border: none;
        margin: 10px 0;
    }
    
    /* Mesaj Kutuları - Asistan */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: transparent;
        border: none;
        padding: 0;
        margin: 10px 0;
    }
    
    /* Kaynak Kartları */
    .source-card {
        background-color: #1E1F20;
        border: 1px solid #444746;
        border-radius: 12px;
        padding: 15px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .source-header {
        color: #A8C7FA !important;
        font-weight: bold;
        font-size: 0.9em;
        margin-bottom: 5px;
    }
    .transparency-score {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        margin-left: 10px;
    }
    .score-high { background-color: #0F5223; color: #6DD58C !important; }
    .score-med { background-color: #5B4300; color: #FFD666 !important; }
    .score-low { background-color: #601410; color: #FFB4AB !important; }
    
    /* Butonlar - İÇİNDEKİ YAZIYI SİYAH YAP */
    .stButton button {
        background-color: #A8C7FA !important;
        border-radius: 24px;
        border: none;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s;
    }
    .stButton button p {
        color: #040C19 !important; /* Buton yazısı kesinlikle koyu olacak */
        font-weight: 600 !important;
    }
    .stButton button:hover {
        background-color: #8AB4F8 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    
    /* Bilgi Kutucukları (Alerts) - Koyu Mod Uyumu */
    .stAlert {
        background-color: #1E1F20 !important;
        border: 1px solid #444746;
        color: #E3E3E3 !important;
    }
    .stAlert p {
        color: #E3E3E3 !important; /* Alert içindeki yazılar açık renk */
    }
    
    /* Input Alanı */
    .stTextInput input {
        background-color: #1E1F20 !important;
        color: #E3E3E3 !important;
        border: 1px solid #444746 !important;
        border-radius: 24px;
        padding: 12px 20px;
    }
    .stTextInput input:focus {
        border-color: #A8C7FA !important;
        box-shadow: 0 0 0 1px #A8C7FA;
    }
    
    /* Expander (Kaynaklar) */
    .streamlit-expanderHeader {
        background-color: #1E1F20 !important;
        color: #E3E3E3 !important;
        border-radius: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)


class RAGSystem:
    """RAG Sistemi - Belge yükleme, vektör oluşturma ve sorgulama"""
    
    def __init__(self, data_folder="data", vector_db_path="vectorstore"):
        self.data_folder = data_folder
        self.vector_db_path = vector_db_path
        self.vectorstore = None
        self.rag_chain = None
        self.retriever = None
        
        # API Key kontrolü
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable bulunamadı! Lütfen .env dosyasını kontrol edin.")
        
        # Gemini API yapılandırması
        genai.configure(api_key=self.api_key)
        
        # Embeddings - LOKAL model (bedava, sınırsız, Türkçe destekli)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # LLM - Gemini API (sadece soru-cevap için, çok az istek)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
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
            self.vectorstore = FAISS.load_local(
                self.vector_db_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return True
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
    st.title("TÜBİTAK RAG")
    
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

# Ana ekran - Chat arayüzü
st.title("Belge Asistanı")

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
