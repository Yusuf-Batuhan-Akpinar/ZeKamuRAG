"""
TÜBİTAK RAG Projesi - Ana Uygulama
Google Gemini API ile Belge Tabanlı Soru-Cevap Sistemi
"""

import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
import google.generativeai as genai

# Çevre değişkenlerini yükle (.env dosyasından)
load_dotenv()
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

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
        self.qa_chain = None
        
        # API Key kontrolü
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable bulunamadı! Lütfen .env dosyasını kontrol edin.")
        
        # Gemini API yapılandırması
        genai.configure(api_key=self.api_key)
        
        # Embeddings ve LLM modellerini başlat
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=self.api_key
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
            temperature=0.3,
            convert_system_message_to_human=True
        )
    
    def load_documents(self):
        """Data klasöründeki tüm PDF dosyalarını yükle"""
        
        # Data klasörü kontrolü
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
            return []
        
        # PDF dosyalarını yükle
        loader = DirectoryLoader(
            self.data_folder,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        
        documents = loader.load()
        return documents
    
    def split_documents(self, documents):
        """Belgeleri küçük parçalara böl (chunking)"""
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Her parça maksimum 1000 karakter
            chunk_overlap=200,  # Parçalar arası 200 karakter örtüşme
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    def create_vectorstore(self, chunks):
        """Vektör veritabanı oluştur ve kaydet"""
        
        # FAISS vektör veritabanı oluştur
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        # Veritabanını diske kaydet
        self.vectorstore.save_local(self.vector_db_path)
        
        return self.vectorstore
    
    def load_vectorstore(self):
        """Kaydedilmiş vektör veritabanını yükle"""
        
        if os.path.exists(self.vector_db_path):
            self.vectorstore = FAISS.load_local(
                self.vector_db_path,
                self.embeddings
            )
            return True
        return False
    
    def create_qa_chain(self):
        """Soru-Cevap zinciri oluştur"""
        
        # Özel prompt şablonu - Halüsinasyonu önlemek için
        prompt_template = """
        Sen bir uzman asistansın. Sadece verilen belgelerdeki bilgileri kullanarak soruları yanıtla.
        Eğer sorunun cevabı belgelerde yoksa, "Bu bilgi belgelerde mevcut değil" de.
        Asla bilgi uydurma veya belge dışı bilgi kullanma.
        
        Bağlam: {context}
        
        Soru: {question}
        
        Detaylı Cevap:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # RetrievalQA zinciri oluştur
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # En alakalı 3 belge parçasını getir
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return self.qa_chain
    
    def initialize(self):
        """Sistemi başlat - belgeler yoksa yükle, varsa hazır"""
        
        # Önce kaydedilmiş vektör DB'yi yüklemeyi dene
        if self.load_vectorstore():
            self.create_qa_chain()
            return "Sistem hazır (Mevcut vektör veritabanı yüklendi)"
        
        # Yoksa yeni oluştur
        documents = self.load_documents()
        
        if len(documents) == 0:
            return f"UYARI: {self.data_folder}/ klasöründe PDF dosyası bulunamadı!"
        
        chunks = self.split_documents(documents)
        self.create_vectorstore(chunks)
        self.create_qa_chain()
        
        return f"Sistem hazır ({len(documents)} belge, {len(chunks)} parça işlendi)"
    
    def calculate_transparency_score(self, doc_content):
        """
        Gelişmiş Şeffaflık Puanı Hesaplama Algoritması
        4 Temel Kriter: Erişilebilirlik, Hesap Verebilirlik, Güncellik, Tutarlılık
        """
        import re
        import datetime
        
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
        
        if not self.qa_chain:
            return {"answer": "Sistem henüz hazır değil!", "source_documents": []}
        
        try:
            response = self.qa_chain.invoke({"query": question})
            
            # Kaynak belgeler için şeffaflık analizi
            processed_sources = []
            for doc in response["source_documents"]:
                total_score, details = self.calculate_transparency_score(doc.page_content)
                processed_sources.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Bilinmeyen Belge"),
                    "page": doc.metadata.get("page", 0),
                    "score": total_score,
                    "details": details
                })
            
            return {
                "answer": response["result"],
                "source_documents": processed_sources
            }
        except Exception as e:
            return {
                "answer": f"Bir hata oluştu: {str(e)}\n\nLütfen API anahtarınızı ve internet bağlantınızı kontrol edin.",
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
