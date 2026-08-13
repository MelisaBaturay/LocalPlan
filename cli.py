import sys
import time
from config import config
from database import VectorDatabase
from ingest import run_ingestion
from rag_engine import RAGEngine

# Optional Rich formatting
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt
    from rich.markdown import Markdown
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

def print_banner():
    if HAS_RICH:
        banner = (
            "[bold cyan]⚡ Microsoft Foundry Local - Offline RAG Assistant CLI[/bold cyan]\n"
            "[dim]Type your question to search documents, or type 'ingest', 'stats', 'help', or 'exit'.[/dim]"
        )
        console.print(Panel(banner, border_style="cyan", expand=False))
    else:
        print("\n=======================================================")
        print("⚡ Microsoft Foundry Local - Offline RAG Assistant CLI")
        print("Type your question, or type 'ingest', 'stats', 'help', or 'exit'.")
        print("=======================================================\n")

def show_stats(db: VectorDatabase):
    stats = db.get_stats()
    if HAS_RICH:
        table = Table(title="Local Knowledge Base Statistics", border_style="bright_blue")
        table.add_column("Metric", style="bold yellow")
        table.add_column("Value", style="bold green")
        table.add_row("Total Ingested Documents", str(stats["total_documents"]))
        table.add_row("Total Text Chunks / Vectors", str(stats["total_chunks"]))
        table.add_row("Database File Path", stats["db_path"])
        table.add_row("Ingested Files", ", ".join(stats["filenames"]) if stats["filenames"] else "None")
        console.print(table)
    else:
        print("\n--- Local Knowledge Base Statistics ---")
        print(f"Total Ingested Documents : {stats['total_documents']}")
        print(f"Total Text Chunks/Vectors: {stats['total_chunks']}")
        print(f"Database File Path       : {stats['db_path']}")
        print(f"Ingested Files           : {', '.join(stats['filenames']) if stats['filenames'] else 'None'}\n")

def main():
    db = VectorDatabase()
    engine = RAGEngine()
    
    stats = db.get_stats()
    if stats["total_chunks"] == 0:
        print("Vector store is empty. Running initial ingestion...")
        run_ingestion()
        
    print_banner()
    
    while True:
        try:
            if HAS_RICH:
                user_input = Prompt.ask("\n[bold green]RAG-AI>[/bold green]").strip()
            else:
                user_input = input("\nRAG-AI> ").strip()
                
            if not user_input:
                continue
                
            cmd = user_input.lower()
            if cmd in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
                
            elif cmd == "ingest":
                print("Ingesting local documents...")
                res = run_ingestion()
                print(f"✓ Ingestion complete! Processed {res['processed_documents']} files ({res['processed_chunks']} chunks).")
                
            elif cmd == "stats":
                show_stats(db)
                
            elif cmd == "help":
                print("\nAvailable Commands:")
                print("  ingest: Re-ingest documents from data/sample_documents")
                print("  stats : Display vector database statistics")
                print("  help  : Show this help message")
                print("  exit  : Quit the CLI assistant")
                print("  Or simply type any question to query your local documents!\n")
                
            else:
                response = engine.ask(user_input)
                
                if HAS_RICH:
                    console.print(Panel(
                        Markdown(response.answer),
                        title=f"[bold green]Answer[/bold green] [dim]({response.llm_provider} | {response.latency_seconds}s)[/dim]",
                        border_style="green"
                    ))
                    if response.retrieved_chunks:
                        console.print("\n[bold cyan]Retrieved Grounded Passages:[/bold cyan]")
                        for idx, c in enumerate(response.retrieved_chunks, 1):
                            console.print(f"  {idx}. [yellow]{c['filename']}[/yellow] (Chunk #{c['chunk_index']}) - Score: [bold green]{c['score']}[/bold green]")
                    else:
                        console.print("[dim yellow]No relevant passages matched the similarity threshold.[/dim yellow]")
                else:
                    print(f"\n--- Answer ({response.llm_provider} | {response.latency_seconds}s) ---")
                    print(response.answer)
                    if response.retrieved_chunks:
                        print("\nRetrieved Grounded Passages:")
                        for idx, c in enumerate(response.retrieved_chunks, 1):
                            print(f"  {idx}. {c['filename']} (Chunk #{c['chunk_index']}) - Score: {c['score']}")
                    else:
                        print("\nNo relevant passages matched the similarity threshold.")

        except KeyboardInterrupt:
            print("\nSession terminated.")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
