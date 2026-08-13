import sys
import os
import webbrowser
import threading
import time
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, make_response

from config import config
from database import VectorDatabase
from ingest import run_ingestion
from rag_engine import RAGEngine
from llm_provider import get_llm_provider, FoundryLocalProvider, LocalOpenAIProvider, OfflineFallbackProvider

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

app = Flask(__name__)
db = VectorDatabase()
engine = RAGEngine()

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local RAG AI Assistant - Microsoft Foundry Local</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --card-border: #334155;
            --accent-color: #38bdf8;
            --accent-hover: #0284c7;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --user-msg-bg: #0369a1;
            --assistant-msg-bg: #1e293b;
            --danger-color: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            display: flex;
            height: 100vh;
            overflow: hidden;
        }

        .sidebar {
            width: 330px;
            background-color: #1e293b;
            border-right: 1px solid var(--card-border);
            display: flex;
            flex-direction: column;
            padding: 20px;
            gap: 16px;
            overflow-y: auto;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-icon {
            font-size: 28px;
        }

        .brand-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--accent-color);
        }

        .brand-subtitle {
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        .section-title {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }

        .stats-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 14px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            text-align: center;
        }

        .stat-value {
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--accent-color);
        }

        .stat-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        .control-group {
            background: rgba(15, 23, 42, 0.4);
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .control-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: var(--text-primary);
            font-weight: 500;
        }

        .slider {
            width: 100%;
            accent-color: var(--accent-color);
        }

        .btn {
            background-color: var(--card-border);
            color: var(--text-primary);
            border: none;
            border-radius: 8px;
            padding: 10px 14px;
            font-weight: 600;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s ease;
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .btn:hover {
            background-color: var(--accent-hover);
            color: white;
        }

        .btn-primary {
            background-color: #0284c7;
            color: white;
        }

        .btn-secondary {
            background-color: #0f172a;
            border: 1px dashed var(--accent-color);
            color: var(--accent-color);
        }

        .btn-danger {
            background-color: rgba(239, 68, 68, 0.15);
            color: var(--danger-color);
            border: 1px solid rgba(239, 68, 68, 0.3);
            padding: 4px 8px;
            font-size: 0.75rem;
            width: auto;
        }

        .btn-danger:hover {
            background-color: var(--danger-color);
            color: white;
        }

        .doc-list {
            display: flex;
            flex-direction: column;
            gap: 6px;
            max-height: 140px;
            overflow-y: auto;
        }

        .doc-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #0f172a;
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 0.78rem;
            border: 1px solid var(--card-border);
        }

        .doc-name {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 170px;
        }

        .main-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }

        .header-banner {
            padding: 16px 28px;
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-bottom: 1px solid var(--card-border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header-info h1 {
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--accent-color);
        }

        .header-info p {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        .chat-box {
            flex: 1;
            padding: 20px 28px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .message {
            display: flex;
            flex-direction: column;
            max-width: 80%;
            border-radius: 12px;
            padding: 14px 18px;
            line-height: 1.6;
            font-size: 0.95rem;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .message.user {
            align-self: flex-end;
            background-color: var(--user-msg-bg);
            color: white;
            border-bottom-right-radius: 2px;
        }

        .message.assistant {
            align-self: flex-start;
            background-color: var(--assistant-msg-bg);
            border: 1px solid var(--card-border);
            color: var(--text-primary);
            border-bottom-left-radius: 2px;
        }

        .message-meta {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 8px;
        }

        .chunks-container {
            margin-top: 12px;
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 12px;
            white-space: normal;
        }

        .chunks-title {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--accent-color);
            margin-bottom: 8px;
        }

        .chunk-item {
            background: #1e293b;
            border-left: 3px solid var(--accent-color);
            padding: 10px 14px;
            margin-bottom: 8px;
            border-radius: 6px;
            font-size: 0.85rem;
            white-space: pre-wrap;
        }

        .chunk-meta {
            display: flex;
            justify-content: space-between;
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 0.78rem;
            margin-bottom: 6px;
            white-space: normal;
        }

        .score-badge {
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent-color);
            border: 1px solid rgba(56, 189, 248, 0.3);
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
        }

        .input-bar {
            padding: 16px 28px;
            background-color: #1e293b;
            border-top: 1px solid var(--card-border);
            display: flex;
            gap: 12px;
        }

        .input-bar input {
            flex: 1;
            background-color: var(--bg-color);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 12px 16px;
            color: white;
            font-size: 0.95rem;
            outline: none;
        }

        .input-bar input:focus {
            border-color: var(--accent-color);
        }

        .input-bar button {
            width: 120px;
        }
    </style>
</head>
<body>

    <!-- Sidebar -->
    <div class="sidebar">
        <div class="brand">
            <span class="brand-icon">⚡</span>
            <div>
                <div class="brand-title">Foundry Local</div>
                <div class="brand-subtitle">Offline RAG Pro Studio</div>
            </div>
        </div>

        <hr style="border-color: var(--card-border);">

        <div>
            <div class="section-title">Knowledge Base Stats</div>
            <div class="stats-card">
                <div>
                    <div class="stat-value" id="doc-count">{{ stats.total_documents }}</div>
                    <div class="stat-label">Documents</div>
                </div>
                <div>
                    <div class="stat-value" id="chunk-count">{{ stats.total_chunks }}</div>
                    <div class="stat-label">Chunks</div>
                </div>
            </div>
        </div>

        <!-- RAG Hyperparameters Tuning -->
        <div>
            <div class="section-title">⚙️ RAG Hyperparameters</div>
            <div class="control-group">
                <div>
                    <div class="control-label">
                        <span>Top-K Passages:</span>
                        <span id="top-k-val">3</span>
                    </div>
                    <input type="range" class="slider" id="top-k-slider" min="1" max="6" value="3" oninput="document.getElementById('top-k-val').innerText = this.value">
                </div>
                <div>
                    <div class="control-label">
                        <span>Min Match Threshold:</span>
                        <span id="thresh-val">15%</span>
                    </div>
                    <input type="range" class="slider" id="thresh-slider" min="5" max="40" value="15" oninput="document.getElementById('thresh-val').innerText = this.value + '%'">
                </div>
            </div>
        </div>

        <!-- Document Library Manager -->
        <div>
            <div class="section-title">📄 Manage Library</div>
            <div class="doc-list" id="doc-list">
                {% for doc in stats.documents %}
                <div class="doc-item">
                    <span class="doc-name" title="{{ doc.filename }}">📄 {{ doc.filename }}</span>
                    <button class="btn btn-danger" onclick="deleteDoc('{{ doc.filename }}')">🗑️ Delete</button>
                </div>
                {% else %}
                <div style="font-size: 0.75rem; color: var(--text-secondary); text-align: center;">No documents loaded</div>
                {% endfor %}
            </div>
        </div>

        <div>
            <div class="section-title">Upload & Actions</div>
            <button class="btn btn-secondary" onclick="document.getElementById('file-upload').click()">📁 Upload Files (.md, .txt, .pdf)</button>
            <input type="file" id="file-upload" accept=".txt,.md,.pdf" multiple style="display: none;" onchange="uploadFiles()">
            
            <div style="height: 8px;"></div>
            
            <button class="btn btn-primary" onclick="reingestDocs()">🔄 Re-Ingest Database</button>
        </div>

        <div style="margin-top: auto; font-size: 0.72rem; color: var(--text-secondary); text-align: center;">
            Zero Internet Cloud Dependency<br>Powered by Microsoft Foundry Local
        </div>
    </div>

    <!-- Main Container -->
    <div class="main-container">
        <div class="header-banner">
            <div class="header-info">
                <h1>Local RAG Q&A Assistant</h1>
                <p>Grounding local documents with zero-cloud AI inference & live RAG tuning</p>
            </div>
            <div>
                <button class="btn btn-secondary" onclick="exportChat()">📥 Export Chat (.md)</button>
            </div>
        </div>

        <div class="chat-box" id="chat-box">
            <div class="message assistant">Hello! I am your <b>Local RAG AI Assistant</b> running offline. Ask me any question about your ingested documents (course FAQ, Foundry Local guide, RAG architecture, or Python notes)!</div>
        </div>

        <div class="input-bar">
            <input type="text" id="user-input" placeholder="Type your question here (e.g. What is Microsoft Foundry Local?)..." onkeydown="if(event.key==='Enter' || event.keyCode===13) sendMessage()">
            <button class="btn btn-primary" id="send-btn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        var chatHistory = [];

        async function sendMessage() {
            try {
                var inputField = document.getElementById('user-input');
                if (!inputField) return;
                
                var question = inputField.value.trim();
                if (!question) return;

                var topKElem = document.getElementById('top-k-slider');
                var threshElem = document.getElementById('thresh-slider');
                var topK = topKElem ? parseInt(topKElem.value) : 3;
                var threshold = threshElem ? parseFloat(threshElem.value) / 100.0 : 0.15;

                var chatBox = document.getElementById('chat-box');

                var userMsg = document.createElement('div');
                userMsg.className = 'message user';
                userMsg.innerText = question;
                chatBox.appendChild(userMsg);
                inputField.value = '';
                chatBox.scrollTop = chatBox.scrollHeight;

                chatHistory.push('User: ' + question);

                var loadingMsg = document.createElement('div');
                loadingMsg.className = 'message assistant';
                loadingMsg.innerText = 'Searching local documents & generating answer...';
                chatBox.appendChild(loadingMsg);
                chatBox.scrollTop = chatBox.scrollHeight;

                var response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question, top_k: topK, threshold: threshold })
                });
                
                if (!response.ok) {
                    throw new Error('Server status ' + response.status);
                }
                
                var data = await response.json();

                loadingMsg.innerText = data.answer || 'No answer returned.';
                chatHistory.push('Assistant: ' + (data.answer || ''));

                var meta = document.createElement('div');
                meta.className = 'message-meta';
                meta.innerText = 'Latency: ' + (data.latency_seconds || 0) + 's | Engine: ' + (data.llm_provider || 'Offline');
                loadingMsg.appendChild(meta);

                if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
                    var chunksContainer = document.createElement('div');
                    chunksContainer.className = 'chunks-container';
                    
                    var title = document.createElement('div');
                    title.className = 'chunks-title';
                    title.innerText = '📌 Grounded Context & Vector Search Scores:';
                    chunksContainer.appendChild(title);

                    data.retrieved_chunks.forEach(function(c) {
                        var item = document.createElement('div');
                        item.className = 'chunk-item';
                        var pct = ((c.score || 0) * 100).toFixed(1);
                        
                        var metaHeader = document.createElement('div');
                        metaHeader.className = 'chunk-meta';
                        metaHeader.innerHTML = '<span>📄 ' + c.filename + ' (Chunk #' + c.chunk_index + ')</span><span class="score-badge">Match Score: ' + pct + '%</span>';
                        
                        var contentBody = document.createElement('div');
                        contentBody.innerText = c.content || '';

                        item.appendChild(metaHeader);
                        item.appendChild(contentBody);
                        chunksContainer.appendChild(item);
                    });

                    loadingMsg.appendChild(chunksContainer);
                }

                chatBox.scrollTop = chatBox.scrollHeight;
            } catch (err) {
                alert('Query Error: ' + err.message);
            }
        }

        async function reingestDocs() {
            if (confirm('Re-ingest documents into SQLite vector store?')) {
                var res = await fetch('/api/ingest', { method: 'POST' });
                var data = await res.json();
                updateStats(data.database_stats);
                alert('Successfully ingested ' + data.processed_documents + ' documents (' + data.processed_chunks + ' chunks)!');
            }
        }

        async function deleteDoc(filename) {
            if (confirm('Delete document "' + filename + '" from database and storage?')) {
                var res = await fetch('/api/documents/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filename: filename })
                });
                var data = await res.json();
                updateStats(data.database_stats);
            }
        }

        function updateStats(stats) {
            document.getElementById('doc-count').innerText = stats.total_documents;
            document.getElementById('chunk-count').innerText = stats.total_chunks;
            
            var docList = document.getElementById('doc-list');
            if (!stats.documents || stats.documents.length === 0) {
                docList.innerHTML = '<div style="font-size: 0.75rem; color: var(--text-secondary); text-align: center;">No documents loaded</div>';
            } else {
                var html = '';
                stats.documents.forEach(function(d) {
                    html += '<div class="doc-item"><span class="doc-name" title="' + d.filename + '">📄 ' + d.filename + '</span><button class="btn btn-danger" onclick="deleteDoc(\'' + d.filename + '\')">🗑️ Delete</button></div>';
                });
                docList.innerHTML = html;
            }
        }

        async function uploadFiles() {
            var input = document.getElementById('file-upload');
            if (input.files.length === 0) return;

            var formData = new FormData();
            for (var i = 0; i < input.files.length; i++) {
                formData.append('files', input.files[i]);
            }

            try {
                var res = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                var data = await res.json();
                updateStats(data.database_stats);
                alert('Uploaded and ingested ' + data.uploaded_count + ' files successfully!');
            } catch (err) {
                alert('Upload error: ' + err);
            }
        }

        function exportChat() {
            if (chatHistory.length === 0) {
                alert('No chat messages to export yet.');
                return;
            }
            var blob = new Blob([chatHistory.join('\n')], { type: 'text/markdown' });
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = 'rag_chat_history.md';
            a.click();
            URL.revokeObjectURL(url);
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    stats = db.get_stats()
    response = make_response(render_template_string(HTML_TEMPLATE, stats=stats))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/query', methods=['POST'])
def query():
    data = request.get_json() or {}
    question = data.get('question', '').strip()
    top_k = data.get('top_k')
    threshold = data.get('threshold')
    
    if not question:
        return jsonify({'answer': 'Please enter a valid question.'})
        
    response = engine.ask(question, top_k=top_k, similarity_threshold=threshold)
    return jsonify({
        'question': response.question,
        'answer': response.answer,
        'retrieved_chunks': response.retrieved_chunks,
        'llm_provider': response.llm_provider,
        'latency_seconds': response.latency_seconds
    })

@app.route('/api/ingest', methods=['POST'])
def ingest():
    res = run_ingestion()
    return jsonify(res)

@app.route('/api/upload', methods=['POST'])
def upload():
    uploaded_files = request.files.getlist('files')
    saved_count = 0
    for file in uploaded_files:
        if file.filename:
            save_path = config.docs_dir / file.filename
            file.save(str(save_path))
            saved_count += 1
            
    res = run_ingestion()
    res['uploaded_count'] = saved_count
    return jsonify(res)

@app.route('/api/documents/delete', methods=['POST'])
def delete_doc():
    data = request.get_json() or {}
    filename = data.get('filename')
    if filename:
        db.delete_document(filename)
        file_path = config.docs_dir / filename
        if file_path.exists():
            try:
                os.remove(file_path)
            except Exception:
                pass
                
    stats = db.get_stats()
    return jsonify({'status': 'deleted', 'database_stats': stats})

def open_browser():
    time.sleep(1.2)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    stats = db.get_stats()
    if stats['total_chunks'] == 0:
        print("[INFO] Initial database setup: Ingesting sample documents...")
        run_ingestion()

    print("\n=======================================================")
    print("[SERVER] Starting Local RAG AI Assistant Web Server...")
    print("[SERVER] Opening Web Interface at: http://127.0.0.1:5000")
    print("[SERVER] Press Ctrl+C to stop the server.")
    print("=======================================================\n")
    
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host='127.0.0.1', port=5000, debug=False)
