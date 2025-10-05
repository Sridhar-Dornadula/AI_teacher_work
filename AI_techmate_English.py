# created By DSR Sridhar Dornadula [SD50]
# it is used to work with AI avatars and provide interactive learning experiences for students giving them a virtual classroom environment knowledge.

import base64
import sys
import speech_recognition as sr
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QTextEdit, QMessageBox, QShortcut, QLabel, QSizePolicy, QSplitter,
    QFileDialog, QComboBox, QDialog, QInputDialog
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QEvent
from PyQt5.QtGui import QKeySequence, QImage, QPixmap, QIcon, QFont, QColor, QKeyEvent, QTextCharFormat, QPalette
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import webbrowser
from urllib.parse import quote
import re, pandas as pd , tempfile
import sys
import json
from datetime import datetime
import requests
import cv2
from openai import OpenAI
import os, faiss, numpy as np , PyPDF2
from sentence_transformers import SentenceTransformer
# from g4f.client import Client
import random , shutil , traceback
import pygame
import os
import asyncio
import edge_tts
#from TTS.api import TTS

data_replace = r"C:\Users\sridh\Downloads\viewing_templates\data_for_realistic.xlsx"
hearing_video_path = r"C:\Users\sridh\Downloads\viewing_templates\hearing_dsr.mp4"
thinking_video_path = [r"C:\Users\sridh\Downloads\viewing_templates\thinking_part_1.mp4",
                      r"C:\Users\sridh\Downloads\viewing_templates\thinking_part_2.mp4"]
startup_video_path = [r"C:\Users\sridh\Downloads\viewing_templates\static_position-Forward.mp4",
                      r"C:\Users\sridh\Downloads\viewing_templates\static_position-REVERSE.mp4"]
paths_for_read = [
    r"C:\Users\sridh\Downloads\viewing_templates\explanation_DSR_Forward.mp4",
    r"C:\Users\sridh\Downloads\viewing_templates\voice_input_speech_2-Forward.mp4",
    r"C:\Users\sridh\Downloads\viewing_templates\voice_input_speech_2-REVERSE.mp4",
    r"C:\Users\sridh\Downloads\viewing_templates\explanation_DSR_REVERSE.mp4"
]

DARK_COLORS = {
    'background': "#d8d0d0",
    'surface': '#2d2d30',
    'surface_alt': '#3e3e42',
    'primary': '#007acc',
    'primary_hover': '#1177bb',
    'primary_pressed': '#0e639c',
    'text_primary': "#ffffff",
    'text_secondary': '#ffffff', 
    'text_muted': '#ffffff',
    'border': '#3e3e42',
    'border_light': '#484848',
    'success': '#4caf50',
    'warning': '#ff9800',
    'error': '#f44336',
    'accent': '#bb86fc',
    'selection': '#264f78',
    'highlight': '#fff3cd'
}

def system_authentication():
    return os.getlogin()

def licence_work_by_server():
    """Fetch allowed usernames from GitHub repo and check with system user"""
    GITHUB_TOKEN = "*********************************************"
    REPO_OWNER = "Sridhar-Dornadula"
    REPO_NAME = "AI_teacher_work"
    FILE_PATH = "****************"

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}?ref=main"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "ngrok-skip-browser-warning": "1"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        content_base64 = response.json()['content']
        content = base64.b64decode(content_base64).decode('utf-8')

        allowed_users = [line.strip() for line in content.splitlines() if line.strip()]
        current_user = system_authentication()

        if current_user in allowed_users:
            print("✅ Access Granted")
            main_to_RUN_GUI()
            return True
        else:
            print(f"❌ Access Denied for {current_user}")
            return False
    else:
        print(f"⚠️ Error fetching file: {response.status_code}")
        return False

class TTSManager:
    def __init__(self):
        self.voice = "en-IN-PrabhatNeural"
        if getattr(sys, 'frozen', False):
            self.temp_dir = os.path.join(tempfile.gettempdir(), "ai_teachmate_audio")
        else:
            self.temp_dir = "temp_audio"
        
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
        except PermissionError:
            self.temp_dir = os.path.join(tempfile.gettempdir(), "ai_teachmate_audio")
            os.makedirs(self.temp_dir, exist_ok=True)

        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.mixer.init()

    async def speak_text_async(self, text):
        """Convert text to speech using edge-tts"""
        try:
            audio_file = os.path.join(
                self.temp_dir, f"speech_{random.randint(1000, 9999)}.mp3"
            )

            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(audio_file)

            if os.path.exists(audio_file):
                self.current_file = audio_file
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                self.is_playing = True
                return True
            return False

        except Exception as e:
            print(f"TTS Error: {e}")
            return False

    def speak_text_sync(self, text):
        """Synchronous wrapper - still blocks but used only in threads"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.speak_text_async(text))
            if result:
                pass
        finally:
            loop.close()
        return result

    def stop_speech(self):
        """Stop playback"""
        try:
            pygame.mixer.music.stop()
            self.is_playing = False
        except:
            pass

    def is_speaking(self):
        """Check if currently speaking"""
        try:
            return pygame.mixer.music.get_busy() and self.is_playing
        except:
            return False

    def cleanup(self):
        """Clean up temp files"""
        try:
            pygame.mixer.quit()
        except:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
tts_manager = TTSManager()

class ListeningThread(QThread):
    """Separate thread for handling speech recognition to avoid blocking UI"""
    listening_started = pyqtSignal()
    listening_finished = pyqtSignal(str)
    listening_error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.is_listening = False
    
    def run(self):
        self.is_listening = True
        self.listening_started.emit()
        
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                print("🎙️ Microphone is ON - Say something...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            
            print("🔄 Processing speech...")
            text = recognizer.recognize_google(audio)
            self.listening_finished.emit(text)
            
        except sr.WaitTimeoutError:
            self.listening_error.emit("Listening timeout - please try again")
        except sr.UnknownValueError:
            self.listening_error.emit("Sorry, I could not understand the audio")
        except sr.RequestError as e:
            self.listening_error.emit(f"Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            self.listening_error.emit(f"An error occurred: {str(e)}")
        finally:
            self.is_listening = False

class WorkerThread(QThread):
    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, prompt, docs=None):
        super().__init__()
        self.prompt = prompt
        self.docs = docs if docs else []

    def run(self):
        try:
            client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
            if self.docs:
                embedder = SentenceTransformer("all-MiniLM-L6-v2")
                query_emb = embedder.encode([self.prompt])
                doc_embs = np.array([d[1] for d in self.docs])

                index = faiss.IndexFlatL2(doc_embs.shape[1])
                index.add(doc_embs)
                D, I = index.search(query_emb, k=3)
                context = "\n\n".join([self.docs[i][0] for i in I[0]])

                full_prompt = f"""You are an AI teacher using RAG.
Context from reference document:
{context}

Question:
{self.prompt}

Provide a comprehensive answer based on the context above and your knowledge.
"""
            else:
                full_prompt = self.prompt

            response = client.chat.completions.create(
                model="google/gemma-3-1b",
                messages=[
                        {"role": "system", "content": "You are a helpful personal assistant."},
                        {"role": "user", "content": full_prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            answer = response.choices[0].message.content
            self.result_ready.emit(answer)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}\n\nMake sure LM Studio is running on localhost:1234"
            self.error_occurred.emit(error_msg)

class TTSThread(QThread):
    tts_started = pyqtSignal()
    tts_finished = pyqtSignal()
    tts_error = pyqtSignal(str)
    text_ready_for_display = pyqtSignal(str)
    
    def __init__(self, text, enable_streaming=False):
        super().__init__()
        self.text = text
        self.should_stop = False
        self.enable_streaming = enable_streaming
    
    def run(self):
        try:            
            if self.enable_streaming:
                self.text_ready_for_display.emit(self.text)
            self.msleep(100)
            success = tts_manager.speak_text_sync(self.text)
            
            if success and not self.should_stop:

                for _ in range(30):  
                    if tts_manager.is_speaking():
                        self.tts_started.emit()   
                        break
                    self.msleep(100)
                
                while tts_manager.is_speaking() and not self.should_stop:
                    self.msleep(100)  
                
                if not self.should_stop:
                    self.tts_finished.emit()
            elif not success:
                self.tts_error.emit("Failed to generate or play speech")
                
        except Exception as e:
            self.tts_error.emit(str(e))
    
    def stop(self):
        self.should_stop = True
        tts_manager.stop_speech()

class VideoWidget(QWidget):
    video_finished = pyqtSignal()
    def __init__(self, video_paths):
        super().__init__()
        self.video_paths = video_paths  
        self.current_video_index = 0
        self.cap = None
        self.is_looping = False  

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)  
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout()
        layout.addWidget(self.label, alignment=Qt.AlignCenter)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_video(self, loop=False):
        """Start video with optional looping"""
        self.is_looping = loop
        video_path = self.video_paths[self.current_video_index]
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.timer.start(33)  

    def stop_video(self):
        self.timer.stop()
        self.is_looping = False

    def update_frame(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(img))
        else:
            if self.is_looping:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                self.video_finished.emit()

class CodeEditor(QTextEdit):
    """VS Code-like editor with syntax highlighting and features"""
    
    def __init__(self):
        super().__init__()
        self.setup_editor()
        self.input_blocked = False  
        
    def setup_editor(self):
        font = QFont("Consolas", 14)
        font.setFixedPitch(True)
        self.setFont(font)

        self.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: none;
                selection-background-color: #264f78;
                selection-color: #ffffff;
            }
        """)
        self.setPlaceholderText("// Start taking notes...\n// Use this space for:\n - Key concepts\n - Important points")
    
    def keyPressEvent(self, event):
        """Override to handle key events properly"""
        if hasattr(self.parent(), 'mic_is_on') and self.parent().mic_is_on:
            return
        if not self.input_blocked:
            super().keyPressEvent(event)
    
    def block_input(self, blocked=True):
        """Method to temporarily block text input"""
        self.input_blocked = blocked

class TextInputGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI TeachMate – Mentor")
        screen = QApplication.primaryScreen()
        rect = screen.availableGeometry()
        self.setGeometry(100, 100, 2000, 900)

        url = "https://img.freepik.com/premium-vector/artificial-intelligence-icon-vector-image-can-be-used-cyberpunk_120816-404547.jpg"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            pixmap = QPixmap()
            pixmap.loadFromData(response.content)
            self.setWindowIcon(QIcon(pixmap))
        except Exception as e:
            print(f"⚠️ Could not load icon: {e}")

        main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(main_splitter)

        left_widget = QWidget()
        left_widget.setFixedWidth(600)  
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        
        header_layout = QHBoxLayout()
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Topic....")
        self.text_input.setStyleSheet("""
            QLineEdit {
                padding: 8px 12px;
                font-size: 14px;
                border: 2px solid #4a90e2;
                border-radius: 6px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #357abd;
            }
        """)
        header_layout.addWidget(self.text_input)
        
        self.submit_button = QPushButton("📻 Submit")
        self.submit_button.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2968a3;
            }
        """)
        header_layout.addWidget(self.submit_button)
        left_layout.addLayout(header_layout)
        self.submit_button.clicked.connect(self.submit_topic)

        self.load_pdf_button = QPushButton("Load PDF")
        self.load_pdf_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        self.load_pdf_button.clicked.connect(self.load_pdf_document)
        header_layout.addWidget(self.load_pdf_button)

        left_layout.addLayout(header_layout)
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)

        self.output_textedit = QTextEdit()
        self.output_textedit.setReadOnly(True)
        self.output_textedit.setStyleSheet("""
            QTextEdit {
                font-size: 16px;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 8px;
                background-color: #f9f9f9;
                line-height: 1.5;
            }
        """)
        self.content_layout.addWidget(self.output_textedit)

        self.video_widget = VideoWidget(startup_video_path)
        self.video_widget.video_finished.connect(self.handle_video_finished)
        self.content_layout.addWidget(self.video_widget)
        
        self.video_widget.start_video()
        self.is_waveform_visible = True
        self.content_layout.setStretch(0, 1)
        self.content_layout.setStretch(1, 1)
        
        left_layout.addWidget(self.content_widget)

        question_layout = QHBoxLayout()
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Ask anytime during the lesson...")
        self.question_input.setStyleSheet("""
            QLineEdit {
                padding: 8px 12px;
                font-size: 14px;
                border: 1px solid #ddd;
                border-radius: 6px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #4a90e2;
            }
        """)
        question_layout.addWidget(self.question_input)
        
        self.record_button = QPushButton("🎙️")
        self.record_button.setFixedSize(45, 35)
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ddd;
                border-radius: 6px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.record_button.clicked.connect(self.record_question)
        question_layout.addWidget(self.record_button)
        left_layout.addLayout(question_layout)

        bottom_layout = QHBoxLayout()
        self.bottom_label = QLabel("© DSR AI Scholar Engine - Sridhar Dornadula")
        self.bottom_label.setStyleSheet("color: #666; font-size: 11px;")
        bottom_layout.addWidget(self.bottom_label)
        bottom_layout.addStretch()
        left_layout.addLayout(bottom_layout)
        
        main_splitter.addWidget(left_widget)
        
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        toolbar = QWidget()
        toolbar.setStyleSheet("""
            QWidget {
                background-color: #2d2d30;
                border-bottom: 1px solid #3e3e42;
            }
        """)
        toolbar.setFixedHeight(35)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 0, 10, 0)
        
        tab_label = QLabel("📝 Black-Board notes")
        tab_label.setStyleSheet("""
            QLabel {
                color: #cccccc;
                background-color: #1e1e1e;
                padding: 8px 15px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-size: 13px;
            }
        """)
        toolbar_layout.addWidget(tab_label)
        toolbar_layout.addStretch()
        
        self.highlighted_text = []  
        self.text_snippets = []    
        self.highlight_color = QColor(255, 255, 0, 100)
        
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["Select", "Refresh", "Save", "Send via Email", "Highlighter", "Text Snippet"])
        self.lang_combo.currentTextChanged.connect(self.handle_dropdown_action)
        self.lang_combo.setStyleSheet("""
            QComboBox {
                background-color: #007acc;
                color: white;
                border: 1px solid #3e3e42;
                padding: 4px 8px;
                border-radius: 3px;
                font-size: 12px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid white;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d30;
                color: white;
                border: 1px solid #3e3e42;
                selection-background-color: #007acc;
                selection-color: white;
            }
            QComboBox QAbstractItemView::item {
                color: white;
                padding: 4px 8px;
                border-bottom: 1px solid #3e3e42;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #404040;
                color: white;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #007acc;
                color: white;
            }
        """)
        toolbar_layout.addWidget(self.lang_combo)
        
        right_layout.addWidget(toolbar)

        self.code_editor = CodeEditor()
        right_layout.addWidget(self.code_editor)
        status_bar = QWidget()
        status_bar.setFixedHeight(25)
        status_bar.setStyleSheet("""
            QWidget {
                background-color: #007acc;
                color: white;
            }
        """)
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(10, 0, 10, 0)
        
        self.line_col_label = QLabel("Ln 1, Col 1")
        self.line_col_label.setStyleSheet("font-size: 12px; color: white;")
        status_layout.addWidget(self.line_col_label)
        
        status_layout.addStretch()
        
        self.char_count_label = QLabel("0 characters")
        self.char_count_label.setStyleSheet("font-size: 12px; color: white;")
        status_layout.addWidget(self.char_count_label)
        
        lang_label = QLabel("Plain Text")
        lang_label.setStyleSheet("font-size: 12px; color: white;")
        status_layout.addWidget(lang_label)
        
        encoding_label = QLabel("UTF-8")
        encoding_label.setStyleSheet("font-size: 12px; color: white;")
        status_layout.addWidget(encoding_label)
        
        right_layout.addWidget(status_bar)
        right_bottom_layout = QHBoxLayout()
        self.right_bottom_label = QLabel("")
        self.right_bottom_label.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        right_bottom_layout.addWidget(self.right_bottom_label)
        right_bottom_layout.addStretch()
        
        right_bottom_widget = QWidget()
        right_bottom_widget.setFixedHeight(35)
        right_bottom_widget.setLayout(right_bottom_layout)
        
        right_layout.addWidget(right_bottom_widget)
        main_splitter.addWidget(right_widget)
        
        main_splitter.setSizes([600, 1000])

        self.code_editor.textChanged.connect(self.update_editor_stats)
        self.code_editor.cursorPositionChanged.connect(self.update_cursor_position)

        self.setup_timers_and_threads()
        self.setup_shortcuts()

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.docs = []  
        self.pdf_loaded = False
        
        self.question_input.setEnabled(False)
        self.record_button.setEnabled(False)
        self.first_chunk_displayed = False
        QTimer.singleShot(300, self.simulate_ctrl_s)

    def load_pdf_document(self):
        """Load and process PDF document for RAG"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select PDF Document", "", "PDF files (*.pdf)"
        )
        
        if not filename:
            return
            
        try:
            # Extract text from PDF
            pdf_text = self.extract_pdf_text(filename)
            
            if not pdf_text.strip():
                QMessageBox.warning(self, "Empty PDF", "No text could be extracted from the PDF!")
                return
            
            # Chunk the text
            chunks = self.chunk_text(pdf_text, chunk_size=500)
            
            # Create embeddings
            QMessageBox.information(self, "Processing", "Creating embeddings... This may take a moment.")
            embeddings = self.embedder.encode(chunks)
            
            # Store docs with embeddings
            self.docs = [(chunk, emb) for chunk, emb in zip(chunks, embeddings)]
            self.pdf_loaded = True
            
            QMessageBox.information(
                self, "Success", 
                f"PDF loaded successfully!\n{len(self.docs)} text chunks processed."
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load PDF:\n{str(e)}")

    def extract_pdf_text(self, pdf_path):
        """Extract text from PDF file"""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def chunk_text(self, text, chunk_size=500):
        """Split text into chunks for better RAG performance"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks          
    
    def handle_dropdown_action(self, action):
        """Handle dropdown menu selections"""
        if action == "Refresh":
            self.refresh_application()
        elif action == "Save":
            self.save_notes_enhanced()
        elif action == "Send via Email":
            self.send_via_email()
        elif action == "Highlighter":
            self.toggle_highlighter()
        elif action == "Text Snippet":
            self.create_text_snippet()

        QTimer.singleShot(100, lambda: self.lang_combo.setCurrentIndex(0))
        
    def setup_timers_and_threads(self):
        """Setup all timers and thread-related variables"""
        self.auto_submit_timer = QTimer()
        self.auto_submit_timer.setSingleShot(True)
        self.auto_submit_timer.timeout.connect(self.ask_question)
        self.question_input.textEdited.connect(self.reset_auto_submit_timer)

        self.listening_thread = None
        self.tts_thread = None
        self.mic_is_on = False
        
        self.is_speaking = False
        self.pending_function = None
        self.worker = None
        self.chunks = []
        self.current_chunk_index = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.display_next_chunk)

        self.transition_timer = QTimer()
        self.transition_timer.setSingleShot(True)
        self.transition_timer.timeout.connect(self.start_main_content_timer)

        self.main_explanation_paused = False
        self.paused_chunk_index = 0
        self.main_explanation_active = False
        self.interrupt_requested = False
        self.in_qa_mode = False
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.shortcut_toggle = QShortcut(QKeySequence("Ctrl+S"), self)
        self.shortcut_toggle.activated.connect(self.toggle_view)

        save_shortcut = QShortcut(QKeySequence("Ctrl+Shift+S"), self)
        save_shortcut.activated.connect(self.save_notes)
        
        QTimer.singleShot(100, self.shortcut_toggle.activated.emit)

    def simulate_ctrl_s(self):
        """Simulate pressing Ctrl+S after window opens"""
        event = QKeyEvent(QEvent.KeyPress, Qt.Key_S, Qt.ControlModifier)
        QApplication.postEvent(self, event)
        event = QKeyEvent(QEvent.KeyRelease, Qt.Key_S, Qt.ControlModifier)
        QApplication.postEvent(self, event)
        
    def update_editor_stats(self):
        """Update character count in status bar"""
        text = self.code_editor.toPlainText()
        char_count = len(text)
        self.char_count_label.setText(f"{char_count} characters")
        
    def update_cursor_position(self):
        """Update cursor position in status bar"""
        cursor = self.code_editor.textCursor()
        line = cursor.blockNumber() + 1
        col = cursor.columnNumber() + 1
        self.line_col_label.setText(f"Ln {line}, Col {col}")

    def save_notes(self):
        """Save notes from the code editor"""
        try:
            notes_text = self.code_editor.toPlainText()
            if not notes_text.strip():
                QMessageBox.information(self, "Save Notes", "No notes to save!")
                return
                
            filename, _ = QFileDialog.getSaveFileName(
                self, 
                "Save Notes", 
                f"notes_{self.text_input.text().replace(' ', '_') if self.text_input.text() else 'learning_notes'}.txt",
                "Text files (*.txt);;Markdown files (*.md);;All files (*.*)"
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(notes_text)
                QMessageBox.information(self, "Save Notes", f"Notes saved successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save notes:\n{str(e)}")

    def toggle_view(self):
        """Toggle between video and text view"""
        if self.is_waveform_visible:
            self.video_widget.hide()
            self.output_textedit.show()
        else:
            self.output_textedit.hide()
            self.video_widget.show()
        self.is_waveform_visible = not self.is_waveform_visible

    def handle_video_finished(self):
        """Handle video playback completion"""
        if self.video_widget.video_paths == startup_video_path:
            self.video_widget.current_video_index = (self.video_widget.current_video_index + 1) % len(self.video_widget.video_paths)
            self.video_widget.start_video()
        else:
            self.video_widget.current_video_index = (self.video_widget.current_video_index + 1) % len(self.video_widget.video_paths)
            self.video_widget.start_video()
    
    def reset_auto_submit_timer(self):
        self.auto_submit_timer.start(1000)

    def submit_topic(self):
        global text
        text = self.text_input.text().strip()
        if not text:
            QMessageBox.warning(self, "Warning", "Please enter a topic.")
            return

        if not self.pdf_loaded:
            reply = QMessageBox.question(
                self, "No PDF Loaded",
                "No PDF document loaded for reference. Continue without RAG?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        self.output_textedit.setPlainText(f"🎙️ Explaining topic: {text}\n\nPlease wait...")
        self.submit_button.setEnabled(False)
        self.submit_button.setText("⏳ Learning...")

        current_notes = self.code_editor.toPlainText()
        if not current_notes.strip():
            self.code_editor.setPlainText(f"# Learning Notes: {text}\n{'='*50}\n")
        self.chunk_counter = 1

        self.question_input.setEnabled(False)
        self.record_button.setEnabled(False)
        self.first_chunk_displayed = False

        self.main_explanation_active = True
        self.main_explanation_paused = False
        self.paused_chunk_index = 0
    
        prompt = f"""
        You are an experienced and engaging teacher creating an audio lesson for students. Your goal is to explain the topic "{text}" in a way that is clear, memorable, and easy to follow when listened to.
        OPENING REQUIREMENT:
        Begin with a warm greeting that introduces the topic, such as:
        - "Hello! Today we're going to explore {text}..."
        - "Welcome! Let's dive into {text} together..."
        - "Hi there! I'm excited to teach you about {text}..."
        - "Greetings! Today's lesson is all about {text}..."
        After the greeting, immediately provide a brief hook or real-world connection to grab attention.
        TEACHING APPROACH:
        - Explain core concepts using simple, conversational language
        - Use concrete examples and analogies that students can relate to
        - Break complex ideas into digestible steps
        - Anticipate common questions or misconceptions and address them
        - End with a brief summary of key takeaways
        STYLE REQUIREMENTS:
        - Write as if speaking directly to a student in a one-on-one tutoring session
        - Use a warm, encouraging tone that maintains student engagement
        - Avoid technical jargon unless you explain it immediately
        - Keep sentences clear and moderately short for audio comprehension
        - Include natural transitions between ideas (e.g., "Now that we understand X, let's look at Y...")
        - Vary your phrasing to maintain interest throughout
        CONTENT STRUCTURE:
        1. Opening: Greeting + topic introduction + brief explanation of what it is and why it matters
        2. Main Content: Detailed explanation with 2-4 key sections, each with examples
        3. Connections: Show how different parts relate to each other
        4. Closing: Quick recap of the most important points
        CONSTRAINTS:
        - Aim for comprehensive coverage suitable for a 6-9 minute audio explanation
        - Avoid bullet points, lists, or heavy formatting (this will be spoken aloud)
        - Do NOT use markdown symbols like *, #, -, or |
        - Write in flowing paragraphs with natural speech patterns
        - Include pauses between major ideas by ending sections with complete thoughts
        Remember: This explanation will be converted to speech and heard by students, not read. Make every sentence count and keep the explanation engaging throughout.
        """
        
        self.worker = WorkerThread(prompt, self.docs if self.pdf_loaded else [])
        self.worker.result_ready.connect(self.handle_response)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()

    def handle_response(self, text):
        """Enhanced handle_response with debugging"""        
        # Clean the text
        cleaned_lines = []
        for line in text.splitlines():
            line = line.strip()
            if line and not (line.startswith(('#', '-', '--'))):
                line = line.replace('*', '')
                cleaned_lines.append(line)
        
        cleaned_text = ' '.join(cleaned_lines)
        print(f"DEBUG: Cleaned text length: {len(cleaned_text)}")
        self.chunks = cleaned_text.split('. ')
        self.chunks = [chunk.strip() for chunk in self.chunks if chunk.strip()]
        
        self.chunks = self.apply_word_replacement(self.chunks)
        
        self.current_chunk_index = 0
        self.output_textedit.setPlainText("🎧 Starting explanation...\n")
        self.timer.start(3000)

    def display_next_chunk(self):
        """Send chunk to TTS first, display text only when speech starts"""
        if self.interrupt_requested:
            print("🛑 Chunk display interrupted for Q&A")
            return
            
        if self.current_chunk_index < len(self.chunks):
            chunk = self.chunks[self.current_chunk_index].strip()
            if chunk:
                chunk = chunk.replace('*', '').replace('#', ' ').replace('-', ' ').replace('|', ' ')
                if chunk and chunk[0] in ['#', '-']:
                    chunk = chunk[1:].lstrip()
                
                print(f"🎯 Processing chunk {self.current_chunk_index + 1}: '{chunk[:50]}...'")

                self.timer.stop()

                if hasattr(self, 'tts_thread') and self.tts_thread and self.tts_thread.isRunning():
                    self.tts_thread.stop()
                    self.tts_thread.quit()
                    self.tts_thread.wait()

                self.current_speaking_chunk = chunk

                self.tts_thread = TTSThread(chunk, enable_streaming=False)

                self.tts_thread.tts_started.connect(self.on_chunk_tts_started)
                self.tts_thread.tts_finished.connect(self.on_chunk_tts_finished)
                self.tts_thread.tts_error.connect(self.on_tts_error)

                use_streaming = False  # Set to False if you want immediate full text display
                
                if use_streaming:
                    self.tts_thread.text_ready_for_display.connect(
                        lambda text: self.start_streaming_display(text)
                    )
                else:
                    self.tts_thread.text_ready_for_display.connect(
                        lambda text: self.display_full_text_immediately(text)
                    )
                
                print("▶️ Starting TTS thread...")
                self.tts_thread.start()

                if not self.first_chunk_displayed:
                    self.question_input.setEnabled(True)
                    self.record_button.setEnabled(True)
                    self.first_chunk_displayed = True

            self.current_chunk_index += 1
        else:
            print("DEBUG: All chunks processed")
            self.timer.stop()
            self.main_explanation_active = False
            self.submit_button.setEnabled(True)
            self.submit_button.setText("📻 Submit")
    
    def on_chunk_tts_started(self):
        """Called when TTS starts - NOW we display text and start video"""
        print("🔊 Chunk TTS started - displaying text and starting video")
        self.is_speaking = True

        self.video_widget.video_paths = paths_for_read
        self.video_widget.current_video_index = 0
        self.video_widget.start_video()

        if hasattr(self, 'current_speaking_chunk'):
            chunk = self.current_speaking_chunk

            current_text = self.output_textedit.toPlainText()
            new_text = f"{current_text}\n\n👉 {chunk}."
            self.output_textedit.setPlainText(new_text)
            self.output_textedit.verticalScrollBar().setValue(
                self.output_textedit.verticalScrollBar().maximum()
            )

            current_notes = self.code_editor.toPlainText()
            numbered_chunk = f"{current_notes}\n{self.chunk_counter}. {chunk}.\n"
            self.code_editor.setPlainText(numbered_chunk)

            cursor = self.code_editor.textCursor()
            cursor.movePosition(cursor.End)
            self.code_editor.setTextCursor(cursor)
            
            self.chunk_counter += 1
            print(f"✍️ Text displayed for chunk {self.chunk_counter - 1}")
    
    def on_chunk_tts_finished(self):
        """Called when chunk TTS finishes - runs in main thread"""
        print("🔇 Chunk TTS finished")
        self.video_widget.stop_video()
        self.is_speaking = False
        
        if not self.mic_is_on:
            self.video_widget.video_paths = startup_video_path
            self.video_widget.current_video_index = 0
            self.video_widget.start_video()

        if hasattr(self, 'current_speaking_chunk'):
            delattr(self, 'current_speaking_chunk')
        
        if self.pending_function:
            self.pending_function()
            self.pending_function = None
        
        print("⏭️ Scheduling next chunk...")
        QTimer.singleShot(1000, self.proceed_to_next_chunk)
        
    def start_streaming_display(self, chunk):
        """Start streaming text display - fixed version that shows complete sentences"""
        print(f"DEBUG: Starting to stream chunk: '{chunk}' (length: {len(chunk)})")

        current_text = self.output_textedit.toPlainText()
        new_text = f"{current_text}\n\n👉 "
        self.output_textedit.setPlainText(new_text)

        current_notes = self.code_editor.toPlainText()
        notes_header = f"{current_notes}\n{self.chunk_counter}. "
        self.code_editor.setPlainText(notes_header)

        self.base_output_text = new_text
        self.base_notes_text = notes_header

        self.stream_index = 0
        self.streaming_chunk = chunk.strip()  
        self.chunk_counter += 1

        self.text_streaming_complete = False
        self.speech_complete = False
        
        print(f"DEBUG: Will stream {len(self.streaming_chunk)} characters")

        if hasattr(self, 'streaming_timer') and self.streaming_timer.isActive():
            self.streaming_timer.stop()

        self.streaming_timer = QTimer()
        self.streaming_timer.timeout.connect(self.add_next_character)
        self.streaming_timer.start(50)

    def add_next_character(self):
        """Add next character to displays - fixed version"""
        if not hasattr(self, 'streaming_chunk') or not hasattr(self, 'stream_index'):
            print("DEBUG: Missing streaming variables, stopping timer")
            if hasattr(self, 'streaming_timer'):
                self.streaming_timer.stop()
            return
        
        if not hasattr(self, 'base_output_text') or not hasattr(self, 'base_notes_text'):
            print("DEBUG: Missing base text variables, stopping timer")
            if hasattr(self, 'streaming_timer'):
                self.streaming_timer.stop()
            return

        if self.stream_index < len(self.streaming_chunk):
            char = self.streaming_chunk[self.stream_index]

            streamed_so_far = self.streaming_chunk[:self.stream_index + 1]

            full_output_text = self.base_output_text + streamed_so_far
            self.output_textedit.setPlainText(full_output_text)
            self.output_textedit.verticalScrollBar().setValue(
                self.output_textedit.verticalScrollBar().maximum()
            )

            full_notes_text = self.base_notes_text + streamed_so_far
            self.code_editor.setPlainText(full_notes_text)

            cursor = self.code_editor.textCursor()
            cursor.movePosition(cursor.End)
            self.code_editor.setTextCursor(cursor)
            
            self.stream_index += 1
            print(f"DEBUG: Streamed {self.stream_index}/{len(self.streaming_chunk)} characters")
            
        else:
            print(f"DEBUG: Finished streaming complete chunk ({len(self.streaming_chunk)} characters)")
            self.streaming_timer.stop()

            final_output_text = self.base_output_text + self.streaming_chunk + "."
            self.output_textedit.setPlainText(final_output_text)
            
            final_notes_text = self.base_notes_text + self.streaming_chunk + ".\n"
            self.code_editor.setPlainText(final_notes_text)

            cursor = self.code_editor.textCursor()
            cursor.movePosition(cursor.End)
            self.code_editor.setTextCursor(cursor)
            
            self.text_streaming_complete = True
            print("DEBUG: Streaming completed successfully")
            self.check_chunk_completion()

    def display_full_text_immediately(self, chunk):
        """Fallback method - display complete text immediately"""
        print(f"DEBUG: Displaying full text immediately: '{chunk}'")

        current_text = self.output_textedit.toPlainText()
        new_text = f"{current_text}\n\n👉 {chunk}."
        self.output_textedit.setPlainText(new_text)
        self.output_textedit.verticalScrollBar().setValue(
            self.output_textedit.verticalScrollBar().maximum()
        )

        current_notes = self.code_editor.toPlainText()
        numbered_chunk = f"{current_notes}\n{self.chunk_counter}. {chunk}.\n"
        self.code_editor.setPlainText(numbered_chunk)

        cursor = self.code_editor.textCursor()
        cursor.movePosition(cursor.End)
        self.code_editor.setTextCursor(cursor)
        
        self.chunk_counter += 1
        self.text_streaming_complete = True
        self.speech_complete = False

        self.check_chunk_completion()

    def check_chunk_completion(self):
        """Check if both text streaming and speech are complete before proceeding"""
        print(f"DEBUG: Checking completion - Text: {self.text_streaming_complete}, Speech: {self.speech_complete}")
        
        if self.text_streaming_complete and self.speech_complete:
            print("DEBUG: Both text and speech completed - ready for next chunk")
            self.text_streaming_complete = False
            self.speech_complete = False

            QTimer.singleShot(1000, self.proceed_to_next_chunk)
        else:
            print("DEBUG: Still waiting for completion")

    def proceed_to_next_chunk(self):
        """Proceed to the next chunk in the sequence"""
        print("▶️ Proceeding to next chunk")
        
        if hasattr(self, 'timer') and not self.timer.isActive():
            self.timer.start(500)
        
    def refresh_application(self):
        """Reset the entire application to initial state"""
        reply = QMessageBox.question(
            self, 
            "Refresh Application", 
            "This will stop the current task and reset the application. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        self.stop_all_processes()
        self.reset_ui_components()
        self.reset_internal_state()
        QMessageBox.information(self, "Refresh Complete", "Application has been reset successfully!")

    def stop_all_processes(self):
        """Stop all running processes and threads"""
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()

        if hasattr(self, 'auto_submit_timer') and self.auto_submit_timer.isActive():
            self.auto_submit_timer.stop()

        if hasattr(self, 'transition_timer') and self.transition_timer.isActive():
            self.transition_timer.stop()
            
        if hasattr(self, 'listening_thread') and self.listening_thread and self.listening_thread.isRunning():
            self.listening_thread.quit()
            self.listening_thread.wait()
            self.listening_thread = None

        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait()
            self.worker = None

        # Stop TTS
        if hasattr(self, 'tts_thread') and self.tts_thread and self.tts_thread.isRunning():
            self.tts_thread.stop()
            self.tts_thread.quit()
            self.tts_thread.wait()
            self.tts_thread = None
        
        tts_manager.stop_speech()

        if hasattr(self, 'video_widget'):
            self.video_widget.stop_video()

    def reset_ui_components(self):
        """Reset all UI components to initial state"""
        self.text_input.clear()
        self.question_input.clear()

        self.output_textedit.clear()
        self.clear_code_editor_completely()
        #self.code_editor.clear()

        self.submit_button.setEnabled(True)
        self.submit_button.setText("📻 Submit")
        self.record_button.setEnabled(False)
        self.record_button.setText("🎙️")

        self.question_input.setEnabled(False)

        if hasattr(self, 'video_widget'):
            self.video_widget.video_paths = startup_video_path
            self.video_widget.current_video_index = 0
            self.video_widget.start_video()

        if hasattr(self, 'is_waveform_visible'):
            if not self.is_waveform_visible:
                self.output_textedit.hide()
                self.video_widget.show()
                self.is_waveform_visible = True

    def clear_code_editor_completely(self):
        """Comprehensive method to clear the code editor and reset all its states"""
        self.code_editor.clear()
        self.code_editor.setPlainText("")
        self.code_editor.document().clear()
        
        cursor = self.code_editor.textCursor()
        cursor.movePosition(cursor.Start)
        self.code_editor.setTextCursor(cursor)
        self.code_editor.document().clearUndoRedoStacks()
        self.code_editor.setCurrentCharFormat(QTextCharFormat())

        cursor = self.code_editor.textCursor()
        cursor.clearSelection()
        self.code_editor.setTextCursor(cursor)

        if hasattr(self, 'highlighted_text'):
            self.highlighted_text.clear()
        if hasattr(self, 'text_snippets'):
            self.text_snippets.clear()
        self.update_editor_stats()        
        print("✅ Code editor cleared completely")
    
    def reset_internal_state(self):
        """Reset all internal state variables"""
        self.chunks = []
        self.current_chunk_index = 0
        self.chunk_counter = 1

        self.first_chunk_displayed = False
        self.is_speaking = False
        self.mic_is_on = False

        self.main_explanation_paused = False
        self.paused_chunk_index = 0
        self.main_explanation_active = False
        self.interrupt_requested = False
    
        self.pending_function = None
        if hasattr(self, 'highlighted_text'):
            self.highlighted_text = []
        if hasattr(self, 'text_snippets'):
            self.text_snippets = []
        if hasattr(self, 'highlighter_active'):
            self.highlighter_active = False
        if hasattr(self, 'streaming_timer') and self.streaming_timer.isActive():
            self.streaming_timer.stop()
        if hasattr(self, 'text_streaming_complete'):
            self.text_streaming_complete = False
        if hasattr(self, 'speech_complete'):
            self.speech_complete = False
            
        print("✅ Internal state reset complete")
        
    def ask_question(self):
        question = self.question_input.text().strip()
        if not question:
            QMessageBox.warning(self, "Warning", "Please type or record a question.")
            return

        if self.timer.isActive():
            self.timer.stop()
            
        prompt = f"""
        You are a helpful teacher responding to a student's question during a lesson on "{text}".
        STUDENT'S QUESTION: "{question}"
        YOUR TASK:
        Provide a clear, direct answer that:
        1. Addresses the specific question asked
        2. Connects back to the main topic "{text}" being taught
        3. Uses a concrete example or analogy to illustrate the concept
        4. Maintains continuity with the ongoing lesson
        RESPONSE GUIDELINES:
        - Get straight to the answer without unnecessary preamble
        - Use simple, conversational language as if speaking face-to-face
        - Keep your response focused and concise (aim for 30-60 seconds when spoken)
        - Include ONE brief, relatable example that clarifies the concept
        - End with a smooth transition back to the main lesson (e.g., "Does that help clarify things?" or "Now you can see how this connects to...")
        STYLE:
        - Speak naturally and warmly, like a patient tutor
        - Avoid phrases like "Great question!" or "That's interesting" - just answer directly
        - Do NOT use markdown symbols (*, #, -, |) since this will be spoken aloud
        - Write in complete, flowing sentences suitable for text-to-speech
        - Be encouraging but not overly enthusiastic
        CONSTRAINTS:
        - Maximum length: What can be spoken comfortably in 60 seconds
        - Must directly answer the question - no tangents or unnecessary background
        - If the question is unclear, provide your best interpretation and answer that
        - If the question is outside the scope of "{text}", briefly acknowledge this and provide a concise answer anyway
        Remember: This is an interruption in the main lesson. Be helpful and clear, then allow the lesson to resume smoothly.
        """
        
        self.worker = WorkerThread(prompt, self.docs if self.pdf_loaded else [])
        self.worker.result_ready.connect(self.handle_question_response)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()

    def handle_question_response(self, answer):
        """Handle Q&A response - TTS first, then display during speech"""
        
        answer_chunks = [answer]
        replaced_chunks = self.apply_word_replacement(answer_chunks)
        if replaced_chunks:
            answer = replaced_chunks[0]

        self.current_qa_answer = answer

        self.question_input.clear()
        self.in_qa_mode = True

        if hasattr(self, 'tts_thread') and self.tts_thread and self.tts_thread.isRunning():
            self.tts_thread.stop()
            self.tts_thread.quit()
            self.tts_thread.wait(1000)

        self.tts_thread = TTSThread(answer, enable_streaming=False)
        self.tts_thread.tts_started.connect(self.on_qa_tts_started_with_display)
        self.tts_thread.tts_finished.connect(self.on_qa_tts_finished)
        self.tts_thread.tts_error.connect(self.on_tts_error)
        print("▶️ Starting Q&A TTS...")
        self.tts_thread.start()

    def on_qa_tts_started_with_display(self):
        """Called when Q&A TTS starts - NOW display text and start video"""
        print("🔊 Q&A TTS started - displaying text and starting video")
        self.is_speaking = True

        self.video_widget.video_paths = paths_for_read
        self.video_widget.current_video_index = 0
        self.video_widget.start_video()

        if hasattr(self, 'current_qa_answer'):
            answer = self.current_qa_answer
            
            current_text = self.output_textedit.toPlainText()
            new_text = f"{current_text}\n\n🗣️ Q&A: {answer}"
            self.output_textedit.setPlainText(new_text)
            self.output_textedit.verticalScrollBar().setValue(
                self.output_textedit.verticalScrollBar().maximum()
            )
        
    def on_qa_tts_finished(self):
        """Called when Q&A TTS finishes"""
        print("🔇 Q&A TTS finished")
        self.video_widget.stop_video()
        self.is_speaking = False
        self.in_qa_mode = False

        if hasattr(self, 'current_qa_answer'):
            delattr(self, 'current_qa_answer')
        
        if not self.mic_is_on:
            self.video_widget.video_paths = startup_video_path
            self.video_widget.current_video_index = 0
            self.video_widget.start_video()
        
        print("🔄 Q&A finished - checking resume conditions...")
        if self.main_explanation_paused:
            if self.paused_chunk_index < len(self.chunks):
                print(f"📖 Resuming from chunk {self.paused_chunk_index}")
                QTimer.singleShot(1500, self.resume_main_explanation)
            else:
                print("✅ No more chunks")
                self.main_explanation_paused = False
                self.main_explanation_active = False
                self.submit_button.setEnabled(True)
                self.submit_button.setText("📻 Submit")
        else:
            if self.current_chunk_index < len(self.chunks):
                print(f"📖 Continuing from chunk {self.current_chunk_index}")
                QTimer.singleShot(1500, self.resume_main_content)
            else:
                print("✅ Explanation completed")
                self.main_explanation_active = False
                self.submit_button.setEnabled(True)
                self.submit_button.setText("📻 Submit")
        
    def resume_main_explanation(self):
        """Resume main explanation from where it was interrupted"""
        print(f"🔄 Resuming main explanation from chunk {self.paused_chunk_index}...")
        self.main_explanation_paused = False
        self.interrupt_requested = False
        self.current_chunk_index = self.paused_chunk_index
        transition_messages = [
            "Great! Now let me continue with the main explanation.",
            "I hope that answered your question. Let's get back to our topic.",
            "Perfect! Now let's continue where we left off.",
            "Got it! Let me resume the main explanation from where we paused.",
            "Excellent! Now, continuing with our main topic..."
        ]
        
        transition_text = random.choice(transition_messages)
        self.video_widget.video_paths = startup_video_path
        self.video_widget.current_video_index = 0
        self.video_widget.start_video()

        if hasattr(self, 'tts_thread') and self.tts_thread and self.tts_thread.isRunning():
            self.tts_thread.stop()
            self.tts_thread.quit()
            self.tts_thread.wait(1000)
        
        self.tts_thread = TTSThread(transition_text, enable_streaming=False)
        self.tts_thread.tts_started.connect(self.on_tts_started_greetings)
        self.tts_thread.tts_finished.connect(self.on_transition_tts_finished)  # Different handler!
        self.tts_thread.tts_error.connect(self.on_tts_error)
        self.tts_thread.start()

    def on_transition_tts_finished(self):
        """Called when transition message TTS finishes"""
        print("🔇 Transition TTS finished")
        self.video_widget.stop_video()
        self.is_speaking = False
        
        if not self.mic_is_on:
            self.video_widget.video_paths = startup_video_path
            self.video_widget.current_video_index = 0
            self.video_widget.start_video()

        print(f"📖 Starting main explanation from chunk {self.current_chunk_index}")
        QTimer.singleShot(1000, self.continue_main_explanation)
    
    def continue_main_explanation(self):
        """Continue the main explanation after transition"""
        print(f"📖 Continuing main explanation from chunk {self.current_chunk_index}")
        if hasattr(self, 'timer') and not self.timer.isActive():
            self.timer.start(1000)
        
    def apply_word_replacement(self, chunks):
        """Replace words in chunks based on Excel mapping"""
        try:
            if not os.path.exists(data_replace):
                print(f"WARNING: Excel file not found at {data_replace}")
                return chunks
                
            print(f"DEBUG: Reading Excel file from: {data_replace}")

            try:
                df = pd.read_excel(data_replace, engine='openpyxl')
                print(f"DEBUG: Excel file loaded successfully. Shape: {df.shape}")
            except Exception as read_error:
                print(f"ERROR: Could not read Excel file: {read_error}")
                return chunks

            if df.empty:
                print("WARNING: Excel file is empty")
                return chunks

            print(f"DEBUG: Available columns: {list(df.columns)}")

            ai_column = None
            self_column = None
            
            for col in df.columns:
                if col.strip().lower() == 'ai':
                    ai_column = col
                elif col.strip().lower() == 'self':
                    self_column = col
                    
            if ai_column is None or self_column is None:
                print(f"WARNING: Required columns 'AI' and 'Self' not found. Available: {list(df.columns)}")
                return chunks
                
            print(f"DEBUG: Using columns - AI: '{ai_column}', Self: '{self_column}'")

            replacement_dict = {}
            skipped_rows = 0
            
            for index, row in df.iterrows():
                try:
                    ai_word = str(row[ai_column]).strip() if pd.notna(row[ai_column]) else ""
                    self_word = str(row[self_column]).strip() if pd.notna(row[self_column]) else ""

                    if (ai_word and self_word and 
                        ai_word.lower() != 'nan' and self_word.lower() != 'nan' and
                        ai_word != 'None' and self_word != 'None'):

                        replacement_dict[ai_word.lower()] = self_word
                        print(f"DEBUG: Added replacement: '{ai_word}' -> '{self_word}'")
                    else:
                        skipped_rows += 1
                        if ai_word or self_word:  
                            print(f"DEBUG: Skipped row {index}: AI='{ai_word}', Self='{self_word}'")
                            
                except Exception as row_error:
                    print(f"WARNING: Error processing row {index}: {row_error}")
                    skipped_rows += 1
            
            print(f"DEBUG: Loaded {len(replacement_dict)} valid replacements, skipped {skipped_rows} rows")
            
            if not replacement_dict:
                print("WARNING: No valid replacement pairs found")
                return chunks

            updated_chunks = []
            total_replacements = 0
            
            for i, chunk in enumerate(chunks):
                print(f"DEBUG: Processing chunk {i+1}/{len(chunks)}: '{chunk[:50]}...'")
                updated_chunk, count = self.replace_words_in_text(chunk, replacement_dict)
                updated_chunks.append(updated_chunk)
                total_replacements += count

                if count > 0:
                    print(f"DEBUG: Chunk {i+1} after replacement: '{updated_chunk[:50]}...'")
            
            print(f"DEBUG: Completed - {total_replacements} total replacements made across all chunks")
            return updated_chunks
            
        except Exception as e:
            print(f"CRITICAL ERROR in word replacement: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return chunks  

    def replace_words_in_text(self, text, replacement_dict):
        """Replace words in text using the replacement dictionary with improved matching"""
        if not replacement_dict or not text:
            return text, 0
        
        updated_text = text
        replacements_made = 0

        sorted_replacements = sorted(replacement_dict.items(), key=lambda x: len(x[0]), reverse=True)
        
        for ai_word, self_word in sorted_replacements:

            original_count = replacements_made

            pattern = r'\b' + re.escape(ai_word) + r'\b'
            matches = re.findall(pattern, updated_text, re.IGNORECASE)
            if matches:
                updated_text = re.sub(pattern, self_word, updated_text, flags=re.IGNORECASE)
                replacements_made += len(matches)
                print(f"DEBUG: Word boundary - Replaced '{ai_word}' with '{self_word}' ({len(matches)} times)")

            elif ai_word.lower() in updated_text.lower():

                temp_text = updated_text.lower()
                count = temp_text.count(ai_word.lower())
                
                if count > 0:
                    pattern_exact = re.compile(re.escape(ai_word), re.IGNORECASE)
                    updated_text = pattern_exact.sub(self_word, updated_text)
                    replacements_made += count
                    print(f"DEBUG: Exact phrase - Replaced '{ai_word}' with '{self_word}' ({count} times)")

            elif len(ai_word) > 3:  
                variations = [
                    ai_word + ',',
                    ai_word + '.',
                    ai_word + '!',
                    ai_word + '?',
                    ai_word + ';',
                    '(' + ai_word + ')',
                    '"' + ai_word + '"',
                    "'" + ai_word + "'",
                ]
                
                for variation in variations:
                    if variation.lower() in updated_text.lower():
                        pattern_var = re.compile(re.escape(variation), re.IGNORECASE)
                        replacement_var = variation.replace(ai_word, self_word)
                        if pattern_var.search(updated_text):
                            updated_text = pattern_var.sub(replacement_var, updated_text)
                            replacements_made += 1
                            print(f"DEBUG: Variation - Replaced '{variation}' with '{replacement_var}'")
                            break
        
        if replacements_made > 0:
            print(f"DEBUG: Total replacements in this text: {replacements_made}")
        
        return updated_text, replacements_made

    def test_excel_file(self):
        """Test method to verify Excel file structure and content"""
        try:
            print(f"Testing Excel file: {data_replace}")
            
            if not os.path.exists(data_replace):
                print("❌ File does not exist!")
                return False
                
            df = pd.read_excel(data_replace, engine='openpyxl')
            print(f"✅ File loaded successfully")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")

            print("\nFirst 5 rows:")
            print(df.head().to_string())

            ai_col = None
            self_col = None
            
            for col in df.columns:
                if col.strip().lower() == 'ai':
                    ai_col = col
                elif col.strip().lower() == 'self':
                    self_col = col
                    
            if ai_col and self_col:
                print(f"\n✅ Found required columns: '{ai_col}' and '{self_col}'")

                valid_pairs = 0
                print("\nValid replacement pairs:")
                for idx, row in df.iterrows():
                    ai_val = str(row[ai_col]).strip() if pd.notna(row[ai_col]) else ""
                    self_val = str(row[self_col]).strip() if pd.notna(row[self_col]) else ""
                    
                    if ai_val and self_val and ai_val.lower() != 'nan' and self_val.lower() != 'nan':
                        print(f"  {valid_pairs + 1}. '{ai_val}' -> '{self_val}'")
                        valid_pairs += 1
                        
                print(f"\nTotal valid pairs: {valid_pairs}")
                return True
            else:
                print("❌ Required columns 'AI' and 'Self' not found")
                return False
                
        except Exception as e:
            print(f"❌ Error testing file: {e}")
            return False
    
    def resume_main_content(self):
        """Resume main content explanation if there are remaining chunks"""
        if self.main_explanation_paused:
            print("⚠️ Main explanation is paused, use resume_main_explanation instead")
            return
        
        if self.current_chunk_index < len(self.chunks):
            print("📖 Resuming main content explanation...")
            transition_messages = [
                "I hope you got your answer for that. Now, let's continue with the main content.",
                "Great! Now, let's get back to our main topic.",
                "I hope that helped clarify things. Let's continue where we left off.",
                "Perfect! Now, let's resume our explanation.",
            ]
            
            transition_text = random.choice(transition_messages)
            
            self.video_widget.video_paths = startup_video_path
            self.video_widget.current_video_index = 0
            self.video_widget.start_video()
            
            if hasattr(self, 'tts_thread') and self.tts_thread and self.tts_thread.isRunning():
                self.tts_thread.stop()
                self.tts_thread.quit()
                self.tts_thread.wait(1000)
            
            self.tts_thread = TTSThread(transition_text, enable_streaming=False)
            self.tts_thread.tts_started.connect(self.on_tts_started_greetings)
            self.tts_thread.tts_finished.connect(self.on_transition_tts_finished)
            self.tts_thread.tts_error.connect(self.on_tts_error)
            self.tts_thread.start()
        else:
            print("✅ Main content explanation completed")
            self.main_explanation_active = False
            self.submit_button.setEnabled(True)
            self.submit_button.setText("📻 Submit")

    def start_main_content_timer(self):
        """Start the main content timer after transition message finishes"""
        self.timer.start(3000)
        
    def handle_error(self, error_text):
        QMessageBox.critical(self, "Error", error_text)
        self.submit_button.setEnabled(True)
        self.submit_button.setText("📻 Submit")

    def speak_text(self, text):
        """Thread-safe TTS implementation"""
        if hasattr(self, 'tts_thread') and self.tts_thread and self.tts_thread.isRunning():
            self.tts_thread.stop()
            self.tts_thread.quit()
            self.tts_thread.wait(1000)
        self.tts_thread = TTSThread(text, enable_streaming=False)
        self.tts_thread.tts_started.connect(self.on_tts_started_greetings)
        self.tts_thread.tts_finished.connect(self.on_tts_finished) 
        self.tts_thread.tts_error.connect(self.on_tts_error)
        self.tts_thread.start()

    def on_tts_started_greetings(self):
        """Called when TTS starts speaking"""
        print("🔊 TTS started speaking...")
        self.is_speaking = True
        self.video_widget.video_paths = paths_for_read
        self.video_widget.current_video_index = 0
        self.video_widget.start_video()
    
    def on_tts_started(self):
        """Called when TTS starts speaking"""
        print("🔊 TTS started speaking...")
        self.is_speaking = True
    
    def on_tts_finished(self):
        """Called when TTS finishes speaking"""
        print("🔇 TTS finished speaking")
        self.video_widget.stop_video()
        self.is_speaking = False
        
        if not self.mic_is_on:
            self.video_widget.video_paths = startup_video_path
            self.video_widget.current_video_index = 0
            self.video_widget.start_video()
        
        if self.pending_function:
            pending_func = self.pending_function
            self.pending_function = None
            QTimer.singleShot(200, pending_func)
    
    def on_tts_error(self, error_message):
        """Called when TTS encounters an error"""
        print(f"❌ TTS Error: {error_message}")
        self.video_widget.stop_video()
        self.is_speaking = False
    
        self.record_button.setEnabled(True)
        self.record_button.setText("🎙️")
        
        if not self.mic_is_on:
            self.video_widget.video_paths = startup_video_path
            self.video_widget.current_video_index = 0
            self.video_widget.start_video()
        
        QMessageBox.warning(self, "Speech Error", f"Speech synthesis failed: {error_message}")
        self.pending_function = None

    def start_listening_process(self):
        """Start the listening process with hearing video"""
        print("🎙️ Microphone turning ON...")
        self.mic_is_on = True

        if self.timer.isActive():
            self.timer.stop()

        self.video_widget.video_paths = [hearing_video_path]
        self.video_widget.current_video_index = 0
        self.video_widget.start_video(loop=True)  

        self.record_button.setText("🔴")  
        self.record_button.setEnabled(False)  

        if self.listening_thread and self.listening_thread.isRunning():
            self.listening_thread.quit()
            self.listening_thread.wait()
        
        self.listening_thread = ListeningThread()
        self.listening_thread.listening_started.connect(self.on_listening_started)
        self.listening_thread.listening_finished.connect(self.on_listening_finished)
        self.listening_thread.listening_error.connect(self.on_listening_error)
        self.listening_thread.start()

    def on_listening_started(self):
        """Called when microphone starts listening"""
        print("🎙️ Microphone is ON - Listening...")

    def on_listening_finished(self, recognized_text):
        """Called when speech recognition completes successfully"""
        print(f"📝 Recognized: {recognized_text}")
        self.question_input.setText(recognized_text)
        self.stop_listening_process()
        self.auto_submit_timer.start(1000)

    def on_listening_error(self, error_message):
        """Called when speech recognition encounters an error"""
        print(f"❌ Listening error: {error_message}")
        QMessageBox.information(self, "Listening Result", error_message)
        QTimer.singleShot(500, self.resume_main_content)

    def stop_listening_process(self):
        """Stop the listening process and return to normal state"""
        print("🎙️ Microphone turning OFF...")
        self.mic_is_on = False
        self.video_widget.stop_video()

        self.video_widget.video_paths = thinking_video_path
        self.video_widget.current_video_index = 0
        self.video_widget.start_video()

        self.record_button.setText("🎙️")
        self.record_button.setEnabled(True)

        if self.listening_thread and self.listening_thread.isRunning():
            self.listening_thread.quit()
            self.listening_thread.wait(1000)
        self.listening_thread = None

    def force_stop_all_audio(self):
        """Aggressively stop all audio playback and timers"""
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        
        if hasattr(self, 'streaming_timer') and self.streaming_timer.isActive():
            self.streaming_timer.stop()
        
        if hasattr(self, 'auto_submit_timer') and self.auto_submit_timer.isActive():
            self.auto_submit_timer.stop()

        if hasattr(self, 'tts_thread') and self.tts_thread and self.tts_thread.isRunning():
            self.tts_thread.should_stop = True
            self.tts_thread.stop()
            tts_manager.stop_speech()
            self.tts_thread.quit()
            self.tts_thread.wait(300) 

        tts_manager.stop_speech()

        self.is_speaking = False

        if hasattr(self, 'text_streaming_complete'):
            self.text_streaming_complete = False
        if hasattr(self, 'speech_complete'):
            self.speech_complete = False
        
        print("✅ All audio stopped")
    
    def record_question(self):
        self.force_stop_all_audio()
        if self.main_explanation_active and not self.main_explanation_paused:
            print("🔄 Interrupting main explanation for Q&A...")
            self.interrupt_main_explanation()
        
        if self.is_speaking:
            self.record_button.setText("⏳")
            self.pending_function = self.record_question
            return

        greetings = [
            "I'm ready and listening — go ahead with your question.",
            "All ears! Ask away whenever you're ready.",
            "Just say your question whenever you're ready.",
            "Listening now — what would you like to ask?",
            "Microphone is live. Ask your question!",
            "I'm here to help. Please ask your question clearly.",
            "Take your time... then go ahead and speak your question.",
            "Ask your question clearly — I'm here to help.",
            "Go ahead. I'm paying close attention.",
        ]

        topic_related_greetings = [
            f"I'm listening — feel free to ask anything about '{text}'.",
            f"Have a question about '{text}'? Go ahead and ask.",
            f"I'm ready to help with '{text}'. What's your question?",
            f"You're learning about '{text}'. Ask me anything!",
        ]

        enthusiastic_greetings = [
            "Let's do this! I'm ready for your question.",
            "Excited to hear what you're thinking — ask away!",
            "I love great questions. Go ahead!",
            "This is your moment — ask me something cool.",
        ]
        
        all_greetings = greetings + enthusiastic_greetings + topic_related_greetings
        greeting_text = random.choice(all_greetings)

        self.video_widget.video_paths = startup_video_path
        self.video_widget.current_video_index = 0
        self.video_widget.start_video()

        self.record_button.setEnabled(False)
        self.record_button.setText("🔄")
    
        self.pending_function = self.start_listening_process
        self.speak_text(greeting_text)

    def interrupt_main_explanation(self):
        """Interrupt the current main explanation and save state"""
        self.main_explanation_paused = True
        self.paused_chunk_index = self.current_chunk_index
        self.interrupt_requested = True
        
        self.force_stop_all_audio()
        print(f"📌 Main explanation paused at chunk {self.paused_chunk_index}")

        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        
        if hasattr(self, 'streaming_timer') and self.streaming_timer.isActive():
            self.streaming_timer.stop()

        if hasattr(self, 'tts_thread') and self.tts_thread and self.tts_thread.isRunning():
            self.tts_thread.stop()
            self.tts_thread.quit()
            self.tts_thread.wait()
        
        tts_manager.stop_speech()
        self.is_speaking = False
        
        print(f"📌 Main explanation paused at chunk {self.paused_chunk_index}")
    
    def save_notes_enhanced(self):
        """Enhanced save functionality with multiple format options"""
        notes_text = self.code_editor.toPlainText()
        if not notes_text.strip():
            QMessageBox.information(self, "Save Notes", "No notes to save!")
            return

        save_dialog = QMessageBox()
        save_dialog.setWindowTitle("Save Options")
        save_dialog.setText("Choose save format:")

        txt_button = save_dialog.addButton("📄 Text File", QMessageBox.ActionRole)
        md_button = save_dialog.addButton("📝 Markdown", QMessageBox.ActionRole)
        html_button = save_dialog.addButton("🌐 HTML", QMessageBox.ActionRole)
        json_button = save_dialog.addButton("📊 JSON", QMessageBox.ActionRole)
        save_dialog.addButton("Cancel", QMessageBox.RejectRole)
        
        save_dialog.exec_()
        clicked_button = save_dialog.clickedButton()
        
        topic = self.text_input.text().replace(' ', '_') if self.text_input.text() else 'learning_notes'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            if clicked_button == txt_button:
                self._save_as_text(notes_text, topic, timestamp)
            elif clicked_button == md_button:
                self._save_as_markdown(notes_text, topic, timestamp)
            elif clicked_button == html_button:
                self._save_as_html(notes_text, topic, timestamp)
            elif clicked_button == json_button:
                self._save_as_json(notes_text, topic, timestamp)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save file:\n{str(e)}")

    def _save_as_text(self, content, topic, timestamp):
        """Save as plain text file"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save as Text", f"{topic}_{timestamp}.txt", "Text files (*.txt)"
        )
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Learning Notes: {self.text_input.text()}\n")
                f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                f.write(content)
            QMessageBox.information(self, "Success", "Text file saved successfully!")

    def _save_as_markdown(self, content, topic, timestamp):
        """Save as Markdown file with formatting"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save as Markdown", f"{topic}_{timestamp}.md", "Markdown files (*.md)"
        )
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# Learning Notes: {self.text_input.text()}\n\n")
                f.write(f"**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")
                lines = content.split('\n')
                for line in lines:
                    if line.strip().startswith('#'):
                        f.write(line + '\n')
                    elif line.strip():
                        f.write(f"- {line.strip()}\n")
                    else:
                        f.write('\n')
            QMessageBox.information(self, "Success", "Markdown file saved successfully!")

    def _save_as_html(self, content, topic, timestamp):
        """Save as HTML file with styling"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save as HTML", f"{topic}_{timestamp}.html", "HTML files (*.html)"
        )
        if filename:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Learning Notes: {self.text_input.text()}</title>
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                           margin: 40px; line-height: 1.6; background: #f8f9fa; }}
                    .container {{ background: white; padding: 30px; border-radius: 10px; 
                                box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                    h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                    .meta {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 20px; }}
                    .highlight {{ background-color: #fff3cd; padding: 2px 4px; border-radius: 3px; }}
                    pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📚 Learning Notes: {self.text_input.text()}</h1>
                    <div class="meta">Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                    <pre>{content}</pre>
                </div>
            </body>
            </html>
            """
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            QMessageBox.information(self, "Success", "HTML file saved successfully!")

    def _save_as_json(self, content, topic, timestamp):
        """Save as JSON file with metadata"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save as JSON", f"{topic}_{timestamp}.json", "JSON files (*.json)"
        )
        if filename:
            data = {
                "topic": self.text_input.text(),
                "created": datetime.now().isoformat(),
                "notes": content,
                "highlighted_snippets": self.highlighted_text,
                "text_snippets": self.text_snippets,
                "metadata": {
                    "character_count": len(content),
                    "line_count": len(content.splitlines()),
                    "app_version": "AI TeachMate v1.0"
                }
            }
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Success", "JSON file saved successfully!")

    def send_via_email(self):
        """Send notes via email using default email client or web interface"""
        notes_text = self.code_editor.toPlainText()
        if not notes_text.strip():
            QMessageBox.information(self, "Email", "No notes to send!")
            return
        
        email_dialog = QMessageBox()
        email_dialog.setWindowTitle("Send via Email")
        email_dialog.setText("Choose email method:")
        
        client_button = email_dialog.addButton("📧 Email Client", QMessageBox.ActionRole)
        web_button = email_dialog.addButton("🌐 Web Email", QMessageBox.ActionRole)
        copy_button = email_dialog.addButton("📋 Copy to Clipboard", QMessageBox.ActionRole)
        email_dialog.addButton("Cancel", QMessageBox.RejectRole)
        
        email_dialog.exec_()
        clicked_button = email_dialog.clickedButton()
        
        subject = f"Learning Notes: {self.text_input.text()}"
        body = f"""Learning Notes from AI TeachMate
        Topic: {self.text_input.text()}
        Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        {'-' * 50}

        {notes_text}

        ---
        Generated by AI TeachMate - DSR AI Scholar Engine"""
        
        try:
            if clicked_button == client_button:
                self._send_via_email_client(subject, body)
            elif clicked_button == web_button:
                self._send_via_web_email(subject, body)
            elif clicked_button == copy_button:
                self._copy_email_to_clipboard(subject, body)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not send email:\n{str(e)}")
            
    def _send_via_email_client(self, subject, body):
        """Send email automatically using SMTP (with GUI input for recipient)"""
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = "***************************"
        sender_password = "******************"

        recipient_email, ok = QInputDialog.getText(
            self, 
            'Send Email', 
            'Enter recipient email address:',
            QLineEdit.Normal,
            ''
        )
        
        if not ok or not recipient_email.strip():
            return
        
        recipient_email = recipient_email.strip()

        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, recipient_email):
            QMessageBox.warning(self, "Invalid Email", "Please enter a valid email address!")
            return

        try:
            msg = MIMEMultipart()
            msg["From"] = sender_email
            msg["To"] = recipient_email
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()  
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()

            QMessageBox.information(self, "Email", f"Email sent successfully to {recipient_email}!")

        except Exception as e:
            QMessageBox.critical(self, "Email Error", f"Failed to send email:\n{str(e)}")

    def _send_via_web_email(self, subject, body):
        """Open web email interface"""
        gmail_url = f"https://mail.google.com/mail/?view=cm&fs=1&su={quote(subject)}&body={quote(body)}"
        webbrowser.open(gmail_url)
        QMessageBox.information(self, "Email", "Web email opened in your browser!")

    def _copy_email_to_clipboard(self, subject, body):
        """Copy email content to clipboard"""
        clipboard = QApplication.clipboard()
        email_content = f"Subject: {subject}\n\n{body}"
        clipboard.setText(email_content)
        QMessageBox.information(self, "Email", "Email content copied to clipboard!")

    def toggle_highlighter(self):
        """Toggle highlighter mode for text selection"""
        if not hasattr(self, 'highlighter_active'):
            self.highlighter_active = False

        self.highlighter_active = not self.highlighter_active

        if self.highlighter_active:
            QMessageBox.information(self, "Highlighter",
                "🖍️ Highlighter ON!\nSelect text and press Ctrl+H to highlight it.")

            if not hasattr(self, 'highlight_shortcut'):
                self.highlight_shortcut = QShortcut(QKeySequence("Ctrl+H"), self)
                self.highlight_shortcut.activated.connect(self.apply_highlight)

        else:
            QMessageBox.information(self, "Highlighter", "🖍️ Highlighter OFF!")

    def apply_highlight(self):
        """Apply yellow background highlight to selected text"""
        cursor = self.code_editor.textCursor()
        if cursor.hasSelection():
            fmt = QTextCharFormat()
            fmt.setBackground(QColor("yellow")) 
            fmt.setForeground(QColor("black")) 
            cursor.mergeCharFormat(fmt)
        else:
            QMessageBox.warning(self, "Highlighter", "⚠️ No text selected to highlight!")

    def save_highlight(self):
        """Save currently selected text as highlight"""
        cursor = self.code_editor.textCursor()
        if cursor.hasSelection():
            selected_text = cursor.selectedText()
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            highlight_entry = {
                "text": selected_text,
                "timestamp": timestamp,
                "position": cursor.selectionStart()
            }
            
            self.highlighted_text.append(highlight_entry)
            QMessageBox.information(self, "Highlight Saved", 
                f"✅ Highlighted text saved!\nTotal highlights: {len(self.highlighted_text)}")
        else:
            QMessageBox.warning(self, "No Selection", "Please select text first!")

    def create_text_snippet(self):
        """Create and manage text snippets"""
        cursor = self.code_editor.textCursor()
        selected_text = cursor.selectedText() if cursor.hasSelection() else ""

        dialog = QDialog(self)
        dialog.setWindowTitle("Create Text Snippet")
        dialog.setFixedSize(400, 300)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2d2d30;
                color: #cccccc;
            }
            QLineEdit, QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
        """)
        
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel("Snippet Name:"))
        name_input = QLineEdit()
        name_input.setPlaceholderText("Enter snippet name...")
        layout.addWidget(name_input)

        layout.addWidget(QLabel("Snippet Content:"))
        content_input = QTextEdit()
        content_input.setPlaceholderText("Enter or paste snippet content...")
        if selected_text:
            content_input.setPlainText(selected_text)
        layout.addWidget(content_input)

        button_layout = QHBoxLayout()
        save_btn = QPushButton("💾 Save Snippet")
        view_btn = QPushButton("👁️ View All")
        cancel_btn = QPushButton("❌ Cancel")
        
        button_layout.addWidget(save_btn)
        button_layout.addWidget(view_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        save_btn.clicked.connect(lambda: self._save_snippet(name_input.text(), content_input.toPlainText(), dialog))
        view_btn.clicked.connect(lambda: self._view_snippets(dialog))
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec_()

    def _save_snippet(self, name, content, dialog):
        """Save a text snippet"""
        if not name.strip() or not content.strip():
            QMessageBox.warning(dialog, "Invalid Input", "Please provide both name and content!")
            return
        
        snippet = {
            "name": name.strip(),
            "content": content.strip(),
            "created": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "topic": self.text_input.text()
        }
        
        self.text_snippets.append(snippet)
        QMessageBox.information(dialog, "Success", 
            f"✅ Snippet '{name}' saved!\nTotal snippets: {len(self.text_snippets)}")
        dialog.accept()

    def _view_snippets(self, parent_dialog):
        """View all saved snippets"""
        if not self.text_snippets:
            QMessageBox.information(parent_dialog, "No Snippets", "No snippets saved yet!")
            return
        
        viewer = QDialog(parent_dialog)
        viewer.setWindowTitle(f"Text Snippets ({len(self.text_snippets)} saved)")
        viewer.setFixedSize(500, 400)
        viewer.setStyleSheet(parent_dialog.styleSheet())
        
        layout = QVBoxLayout(viewer)
        
        snippets_text = QTextEdit()
        snippets_text.setReadOnly(True)
        
        content = "📝 SAVED TEXT SNIPPETS\n" + "=" * 40 + "\n\n"
        for i, snippet in enumerate(self.text_snippets, 1):
            content += f"{i}. {snippet['name']}\n"
            content += f"   Topic: {snippet['topic']}\n"
            content += f"   Created: {snippet['created']}\n"
            content += f"   Content: {snippet['content'][:100]}{'...' if len(snippet['content']) > 100 else ''}\n"
            content += "-" * 40 + "\n\n"
        
        snippets_text.setPlainText(content)
        layout.addWidget(snippets_text)
        
        btn_layout = QHBoxLayout()
        export_btn = QPushButton("📤 Export All")
        clear_btn = QPushButton("🗑️ Clear All")
        close_btn = QPushButton("✅ Close")
        
        btn_layout.addWidget(export_btn)
        btn_layout.addWidget(clear_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        export_btn.clicked.connect(lambda: self._export_snippets(viewer))
        clear_btn.clicked.connect(lambda: self._clear_snippets(viewer, snippets_text))
        close_btn.clicked.connect(viewer.accept)
        
        viewer.exec_()

    def _export_snippets(self, parent):
        """Export snippets to file"""
        if not self.text_snippets:
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            parent, "Export Snippets", 
            f"snippets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON files (*.json);;Text files (*.txt)"
        )
        
        if filename:
            try:
                if filename.endswith('.json'):
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(self.text_snippets, f, indent=2, ensure_ascii=False)
                else:
                    with open(filename, 'w', encoding='utf-8') as f:
                        for snippet in self.text_snippets:
                            f.write(f"Name: {snippet['name']}\n")
                            f.write(f"Topic: {snippet['topic']}\n")
                            f.write(f"Created: {snippet['created']}\n")
                            f.write(f"Content:\n{snippet['content']}\n")
                            f.write("-" * 50 + "\n\n")
                
                QMessageBox.information(parent, "Success", "Snippets exported successfully!")
            except Exception as e:
                QMessageBox.critical(parent, "Error", f"Could not export snippets:\n{str(e)}")

    def _clear_snippets(self, parent, text_widget):
        """Clear all snippets"""
        reply = QMessageBox.question(parent, "Clear Snippets", 
            "Are you sure you want to clear all snippets?",
            QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.text_snippets.clear()
            self.highlighted_text.clear()
            text_widget.setPlainText("All snippets cleared!")
            QMessageBox.information(parent, "Cleared", "All snippets have been cleared!")

    def update_editor_stats(self):
        """Update character count and other stats in status bar"""
        text = self.code_editor.toPlainText()
        char_count = len(text)
        highlights_count = len(getattr(self, 'highlighted_text', []))
        snippets_count = len(getattr(self, 'text_snippets', []))
        
        stats_text = f"{char_count} chars"
        if highlights_count > 0:
            stats_text += f" | {highlights_count} highlights"
        if snippets_count > 0:
            stats_text += f" | {snippets_count} snippets"
            
        self.char_count_label.setText(stats_text)

def apply_dark_theme_to_app(app):
    """Apply dark theme to the entire application"""
    app.setStyle('Fusion')

    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(DARK_COLORS['background']))
    dark_palette.setColor(QPalette.WindowText, QColor(DARK_COLORS['text_primary']))
    
    app.setPalette(dark_palette)
    
def main_to_RUN_GUI():
    app = QApplication(sys.argv)
    apply_dark_theme_to_app(app)
    url_1 = "https://images.prismic.io/tefl-iberia/Zi-aA93JpQ5PTPaB_BestAItoolsforTEFLteachers-featuredimage-2.png?auto=format,compress"
    try:
        r = requests.get(url_1, timeout=5)
        r.raise_for_status()
        pixmap = QPixmap()
        pixmap.loadFromData(r.content)
        app.setWindowIcon(QIcon(pixmap))  
    except Exception as e:
        print(f"⚠️ Could not load icon: {e}")
    window = TextInputGUI()
    window.show()
    sys.exit(app.exec_())
    
def main_authentication():
    licence_work_by_server()
         
if __name__ == '__main__':
   main_authentication()
