import customtkinter as ctk
from tkinter import filedialog

class VideoImport(ctk.CTkFrame):
    def __init__(self, parent, import_func):
        super().__init__(master=parent)
        self.grid(column=0, columnspan=2, row=0, sticky='nsew')
        self.import_func = import_func

        # Configure grid layout
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.columnconfigure(0, weight=1)

        # Center card
        self.card = ctk.CTkFrame(self, corner_radius=20, fg_color="#2D2D2D")
        self.card.grid(row=0, column=0, pady=50, sticky='n')
        self.card.grid_rowconfigure((0, 1, 2, 3, 4), weight=1)
        self.card.grid_columnconfigure(0, weight=1)

        # Title
        self.title_label = ctk.CTkLabel(self.card, text="InterpFlow", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.grid(row=0, column=0, pady=(20, 10))

        # Button Row
        button_frame = ctk.CTkFrame(self.card, fg_color="transparent")
        button_frame.grid(row=1, column=0, pady=10)

        self.select_video_btn = ctk.CTkButton(button_frame, text='🎞️ Select Video', command=self.select_video, width=150, height=40, corner_radius=10)
        self.select_video_btn.grid(row=0, column=0, padx=5)

        self.select_output_btn = ctk.CTkButton(button_frame, text='📁 Select Output Folder', command=self.select_output_folder, width=180, height=40, corner_radius=10)
        self.select_output_btn.grid(row=0, column=1, padx=5)

        self.start_btn = ctk.CTkButton(button_frame, text='🚀 Start Processing', command=self.start_processing, width=150, height=40, corner_radius=10)
        self.start_btn.grid(row=0, column=2, padx=5)

        # Inputs Row
        input_frame = ctk.CTkFrame(self.card, fg_color="transparent")
        input_frame.grid(row=2, column=0, pady=(10, 20))

        self.batch_var = ctk.IntVar(value=12)
        self.width_var = ctk.IntVar(value=680)
        self.height_var = ctk.IntVar(value=480)
        self.trt_var = ctk.BooleanVar(value=False)

        ctk.CTkLabel(input_frame, text="Batch:").grid(row=0, column=0, padx=(5, 0))
        self.batch_entry = ctk.CTkEntry(input_frame, textvariable=self.batch_var, width=60)
        self.batch_entry.grid(row=0, column=1, padx=(0, 10))

        ctk.CTkLabel(input_frame, text="Width:").grid(row=0, column=2, padx=(5, 0))
        self.width_entry = ctk.CTkEntry(input_frame, textvariable=self.width_var, width=60)
        self.width_entry.grid(row=0, column=3, padx=(0, 10))

        ctk.CTkLabel(input_frame, text="Height:").grid(row=0, column=4, padx=(5, 0))
        self.height_entry = ctk.CTkEntry(input_frame, textvariable=self.height_var, width=60)
        self.height_entry.grid(row=0, column=5, padx=(0, 10))

        self.trt_checkbox = ctk.CTkCheckBox(input_frame, text="TRT", variable=self.trt_var)
        self.trt_checkbox.grid(row=0, column=6, padx=(10, 0))

        # Status label
        self.status_label = ctk.CTkLabel(self.card, text="", text_color="red")
        self.status_label.grid(row=3, column=0, pady=10)

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=0)
        self.progress_bar.grid_remove()

        # Store paths
        self.video_path = None
        self.output_folder = None

    def select_video(self):
        video_file = filedialog.askopenfile(title="Select Video")
        if video_file:
            self.video_path = video_file.name
            self.select_video_btn.configure(text="✔️ Video Selected")
            self.update_status(f"🎥 Video Selected", success=True)

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder = folder
            self.select_output_btn.configure(text="✔️ Output Selected")
            self.update_status(f"📂 Output Folder Selected", success=True)

    def start_processing(self):
        if not self.video_path or not self.output_folder:
            self.update_status("⚠️ Please select both video and output folder.", success=False)
            return

        self.update_status("🚀 Processing started...", success=True)

        self.progress_bar.grid()
        self.progress_bar.set(0)

        batch = self.batch_var.get()
        width = self.width_var.get()
        height = self.height_var.get()
        path_to_trt = "use_trt" if self.trt_var.get() else None

        self.import_func(
            video_dr=self.video_path,
            output_folder=self.output_folder,
            batch=batch,
            output_width=width,
            output_height=height,
            use_to_trt=path_to_trt,
            progress_callback=self.update_progress
        )

    def update_status(self, message, success=True):
        color = "green" if success else "red"
        self.status_label.configure(text=message, text_color=color)

    def update_progress(self, current, total):
        percent = current / total
        self.progress_bar.set(percent)
        self.progress_bar.update()

        if percent == 1.0:
            self.update_status("🎬 Video Exported!", success=True)
