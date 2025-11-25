import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
import sys
import time

# Add the directory containing the compiled C++ module to the Python path
# This assumes the module is in the build directory or installed
# For development, we might need to adjust this path

try:
    # Attempt to import the bindings directly if installed or in PYTHONPATH
    import vector_database_bindings as vdb
except ImportError:
    # Fallback for development: try to find it in the build directory
    script_dir = os.path.dirname(__file__)
    build_dir = os.path.join(script_dir, '..', '..', 'lightweight_vector_database', 'build') # Corrected path
    sys.path.insert(0, os.path.abspath(build_dir))
    try:
        import vector_database_bindings as vdb
    except ImportError as e:
        messagebox.showerror("Import Error", f"Could not import vector_database_bindings. Make sure it's built and accessible. Error: {e}")
        sys.exit(1)

from embedder import ImageEmbedder

class ImageSimilarityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Similarity Search")

        self.embedder = ImageEmbedder()
        self.db = None
        self.db_path = "image_database.bin"
        self.image_paths = {} # id -> path

        self.vector_dimension = 1000 # MobileNetV2 output feature size

        self._create_widgets()
        self._load_database()

    def _create_widgets(self):
        # Frame for database operations
        db_frame = tk.LabelFrame(self.root, text="Database Operations", padx=10, pady=10)
        db_frame.pack(pady=10, padx=10, fill="x")

        self.load_db_button = tk.Button(db_frame, text="Load Database", command=self._load_database)
        self.load_db_button.pack(side="left", padx=5)

        self.save_db_button = tk.Button(db_frame, text="Save Database", command=self._save_database)
        self.save_db_button.pack(side="left", padx=5)

        self.index_button = tk.Button(db_frame, text="Index Images from Folder", command=self._index_images_from_folder)
        self.index_button.pack(side="left", padx=5)

        self.status_label = tk.Label(self.root, text="Status: Ready", bd=1, relief="sunken", anchor="w")
        self.status_label.pack(side="bottom", fill="x")

        # Frame for search operations
        search_frame = tk.LabelFrame(self.root, text="Search Operations", padx=10, pady=10)
        search_frame.pack(pady=10, padx=10, fill="x")

        self.query_image_button = tk.Button(search_frame, text="Select Query Image", command=self._select_query_image)
        self.query_image_button.pack(side="left", padx=5)

        self.num_results_label = tk.Label(search_frame, text="Number of Results (k):")
        self.num_results_label.pack(side="left", padx=5)
        self.num_results_entry = tk.Entry(search_frame, width=5)
        self.num_results_entry.insert(0, "5")
        self.num_results_entry.pack(side="left", padx=5)

        # Display frames
        display_frame = tk.Frame(self.root)
        display_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.query_image_panel = tk.Label(display_frame, text="Query Image")
        self.query_image_panel.pack(side="left", padx=5, fill="both", expand=True)

        self.results_frame = tk.Frame(display_frame)
        self.results_frame.pack(side="right", padx=5, fill="both", expand=True)

        self.results_labels = []
        for i in range(int(self.num_results_entry.get())):
            label = tk.Label(self.results_frame, text=f"Result {i+1}")
            label.pack(side="top", pady=2, fill="x")
            self.results_labels.append(label)

    def _update_status(self, message):
        self.status_label.config(text=f"Status: {message}")
        self.root.update_idletasks()

    def _load_database(self):
        self._update_status("Loading database...")
        try:
            print(f"DEBUG: Initializing database with path='{self.db_path}', dimension={self.vector_dimension}, read_only=False")
            self.db = vdb.Database(self.db_path, self.vector_dimension, read_only=False)
            self.db.load()
            # Reconstruct image_paths from metadata if possible, or clear
            self.image_paths = {} # For simplicity, clear for now. A real app would store paths in metadata.
            self._update_status(f"Database loaded from {self.db_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load database: {e}")
            print(f"DEBUG: Initializing empty database with path='{self.db_path}', dimension={self.vector_dimension}, read_only=False after error")
            self.db = vdb.Database(self.db_path, self.vector_dimension, read_only=False) # Initialize an empty one
            self._update_status("Database initialized (empty)")

    def _save_database(self):
        if self.db:
            self._update_status("Saving database...")
            try:
                self.db.save()
                self._update_status(f"Database saved to {self.db_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save database: {e}")
        else:
            messagebox.showinfo("Info", "No database to save.")

    def _index_images_from_folder(self):
        folder_selected = filedialog.askdirectory()
        if not folder_selected:
            return

        self._update_status(f"Indexing images from {folder_selected}...")
        image_files = [os.path.join(folder_selected, f) for f in os.listdir(folder_selected) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        if not image_files:
            messagebox.showinfo("Info", "No image files found in the selected folder.")
            self._update_status("Ready")
            return

        # Re-initialize database to ensure it's fresh for new indexing
        print(f"DEBUG: Re-initializing database for indexing with path='{self.db_path}', dimension={self.vector_dimension}, read_only=False")
        self.db = vdb.Database(self.db_path, self.vector_dimension, read_only=False)
        self.image_paths = {}

        start_time = time.time()
        for i, img_path in enumerate(image_files):
            self._update_status(f"Indexing image {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
            try:
                embedding = self.embedder.embed_image(img_path)
                # Store image path in metadata
                metadata = {"path": img_path}
                print(f"DEBUG: Inserting into database: embedding_length={len(embedding.tolist())}, metadata={metadata}")
                node_id = self.db.insert(embedding.tolist(), metadata)
                self.image_paths[node_id] = img_path
            except Exception as e:
                print(f"Error embedding or inserting {img_path}: {e}")
        
        end_time = time.time()
        self._update_status(f"Indexing complete. {len(image_files)} images indexed in {end_time - start_time:.2f} seconds.")
        self._save_database() # Automatically save after indexing

    def _select_query_image(self):
        if not self.db:
            messagebox.showinfo("Info", "Please index images or load a database first.")
            return

        query_image_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        if not query_image_path:
            return

        self._update_status(f"Querying for similar images to {os.path.basename(query_image_path)}...")
        
        # Display query image
        self._display_image(query_image_path, self.query_image_panel, "Query Image")

        try:
            query_embedding = self.embedder.embed_image(query_image_path)
            k = int(self.num_results_entry.get())
            
            # Query the database, requesting metadata to get the original image paths
            print(f"DEBUG: Querying database with query_embedding_length={len(query_embedding.tolist())}, k={k}, include={{vdb.Include.ID, vdb.Include.DISTANCE, vdb.Include.METADATA}}")
            results = self.db.query(query_embedding.tolist(), k, include={vdb.Include.ID, vdb.Include.DISTANCE, vdb.Include.METADATA})

            self._update_status(f"Search complete. Found {len(results)} results.")
            self._display_results(results)

        except Exception as e:
            messagebox.showerror("Error", f"Error during query: {e}")
            self._update_status("Ready")

    def _display_image(self, image_path, panel, text_label):
        try:
            img = Image.open(image_path)
            img.thumbnail((200, 200)) # Resize for display
            img_tk = ImageTk.PhotoImage(img)
            panel.config(image=img_tk, text=text_label, compound="top")
            panel.image = img_tk # Keep a reference!
        except Exception as e:
            panel.config(image="", text=f"Error loading {os.path.basename(image_path)}: {e}")
            panel.image = None

    def _display_results(self, results):
        for i, label in enumerate(self.results_labels):
            if i < len(results):
                result = results[i]
                result_path = result.metadata.get("path", "N/A")
                distance = result.distance
                
                label.config(text=f"Result {i+1} (Dist: {distance:.4f})")
                self._display_image(result_path, label, f"Result {i+1}")
            else:
                label.config(image="", text=f"Result {i+1}")
                label.image = None


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSimilarityApp(root)
    root.geometry("1000x700") # Set initial window size
    root.mainloop()
